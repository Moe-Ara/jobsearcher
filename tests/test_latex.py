"""LaTeX tool: CV + DIN letter templates compile with the real engine (Tectonic in Docker, xelatex/pdflatex locally),
fit on one page and produce a clean text layer. Skipped where no TeX engine is installed."""


import pytest

from app.config_files import get_profile
from app.services import latex, toolbox
from app.services.cv import FIT_LEVELS, build_cv, get_master_cv
from app.services.latex import cv_tex, esc, letter_tex, split_letter

LETTER = ("Sehr geehrter Herr Lang,\n\nan der Stelle reizt mich die Verbindung von Simulation und Robotik.\n\n"
          "Bei Nordlicht Mobility Software habe ich ein Validierungsframework entwickelt.\n\n"
          "Über eine Einladung zu einem Vorstellungsgespräch freue ich mich sehr.\n\nMit freundlichen Grüßen\nAlex Beispiel")


@pytest.fixture
def with_latex(monkeypatch):
    monkeypatch.delenv("JOBSEARCHER_DISABLE_LATEX", raising=False)
    if not latex.available():
        pytest.skip("no TeX engine installed")


def text_of(pdf: bytes) -> tuple[str, int]:
    from app.services.ats import pdf_text

    return pdf_text(pdf)  # the ATS view (pdftotext when installed)


def test_split_letter_finds_salutation_body_and_closing():
    opening, body, closing = split_letter(LETTER)
    assert opening == "Sehr geehrter Herr Lang," and closing == "Mit freundlichen Grüßen"
    assert len(body) == 3 and not any("Alex Beispiel" in p for p in body)


def test_templates_escape_special_characters():
    cv = build_cv(get_master_cv(), "de", [], "Profil mit C# & 100 % Python_3.")
    assert r"C\# \& 100 \% Python\_3." in cv_tex(cv)
    assert esc("{x}") == r"\{x\}"
    tex = letter_tex(LETTER, "Bewerbung", cv, ["Acme & Co. KG"], "19.10.2026", ["Lebenslauf"], "de")
    assert r"Acme \& Co. KG" in tex and r"\opening{Sehr geehrter Herr Lang,}" in tex and "Anlagen" in tex


def test_cv_compiles_to_one_clean_page(with_latex):
    master = get_master_cv()
    cv = toolbox.fit_cv_to_page(lambda lv: build_cv(master, "de", [], "Softwareentwickler mit Java/Spring.", None, lv),
                                len(FIT_LEVELS))
    pdf, engine = toolbox.render_cv(cv)
    assert engine.startswith("latex/")
    text, pages = text_of(pdf)
    assert pages == 1 and toolbox.verify_pdf(pdf, 1).ok
    for part in ("ALEX BEISPIEL", "alex.beispiel@example.org", "06/2023 - 05/2025", "Werkstudent", "Staatsangehörigkeit",
                 "großer Datenmengen"):  # umlauts and ß survive (XeTeX needs fontspec, not T1/inputenc)
        assert part.lower() in text.lower(), part


def test_letter_compiles_with_din_layout(with_latex):
    cv = build_cv(get_master_cv(), "de", [], "")
    pdf, engine = toolbox.render_letter(LETTER, "Bewerbung um die Stelle als Entwickler", cv,
                                        ["Wandelbots GmbH", "Herrn Lang", "Dresden"], "19.10.2026",
                                        ["Lebenslauf", "Masterzeugnis"], "de")
    assert engine.startswith("latex/")
    text, pages = text_of(pdf)
    assert pages == 1
    for part in ("Alex Beispiel", "Wandelbots GmbH", "Herrn Lang", "19.10.2026", "Bewerbung um die Stelle als Entwickler",
                 "Sehr geehrter Herr Lang,", "Mit freundlichen Grüßen", "Anlagen", "Masterzeugnis"):
        assert part in text, part
    assert text.count("Sehr geehrt") == 1


def test_broken_latex_falls_back_to_the_builtin_renderer(with_latex, monkeypatch):
    monkeypatch.setattr(latex, "cv_tex", lambda cv: r"\documentclass{article}\begin{document}\undefinedmacro\end{document}")
    toolbox._pdf_cache.clear()
    pdf, engine = toolbox.render_cv(build_cv(get_master_cv(), "en", [], "Statement for the fallback test."))
    assert engine == "builtin" and pdf.startswith(b"%PDF")


def test_tool_calls_are_logged():
    with toolbox.recording() as calls:
        toolbox.verify_pdf(toolbox.render_cv(build_cv(get_master_cv(), "de", [], "x"))[0], 1)
    assert [c["tool"] for c in calls] == ["render_cv", "verify_pdf"] and all("seconds" in c for c in calls)
    assert {"render_cv", "render_letter", "company_research", "fact_check"} <= set(toolbox.REGISTRY)
    assert get_profile() is not None


def test_a_long_letter_with_four_attachments_still_fits_one_page(with_latex):
    """The 'Anlagen' list (one per line) used to push the closing of ~250-word letters onto page 2."""
    cv = build_cv(get_master_cv(), "de", [], "Profil.")
    para = ("In meiner Tätigkeit als Werkstudent bei Nordlicht Mobility Software habe ich ein Java/Spring-basiertes "
            "Validierungsframework mit über 200 automatisierten Testfällen mitentwickelt und CI/CD-Pipelines mit Jenkins "
            "und Docker automatisiert. ")
    letter = ("Sehr geehrte Damen und Herren,\n\n" + "\n\n".join([para * 2] * 5) +
              "\n\nIch freue mich auf ein persönliches Gespräch.\n\nMit freundlichen Grüßen\nAlex Beispiel")
    attachments = ["Lebenslauf", "Masterzeugnis mit Notenübersicht", "Bachelorzeugnis",
                   "Arbeitszeugnis Nordlicht Mobility Software (Werkstudent)"]
    assert len(letter.split()) > 270
    loose = latex.compile_tex(letter_tex(letter, "Bewerbung als Softwareentwickler", cv, ["Firma GmbH", "Ulm"],
                                         "Ulm, 20.10.2026", attachments, "de"))
    assert toolbox.page_count(loose.pdf) == 2  # the old layout spills
    pdf, engine = toolbox.render_letter(letter, "Bewerbung als Softwareentwickler", cv, ["Firma GmbH", "Ulm"],
                                        "Ulm, 20.10.2026", attachments, "de")
    text, pages = text_of(pdf)
    assert pages == 1 and "fit=" in engine
    assert "Mit freundlichen Grüßen" in text and "Bachelorzeugnis" in text
