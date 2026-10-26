"""Application agent (tailored CV, letter, e-mail) + reviewer agent (ATS check, recruiter review, SEND/REVISE loop)."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config_files import get_prefs, get_profile
from app.db import session_scope
from app.llm.schemas import CvProfileStatement, RecruiterReview
from app.main import app
from app.models import ApplicationDocument, ApplicationPackage
from app.services import application_agent as agent
from app.services import ats
from app.services.cv import build_cv, check_statement, cv_vocabulary, get_master_cv, render_text, tailor_cv
from app.services.cv_render import render_letter_pdf, render_pdf
from app.services.latex import cv_tex as render_tex
from app.services.keywords import job_keywords, literal_in, term_in
from app.services.reviewer import decide, revision_plan
from tests.test_letters import job_id, make_setup  # noqa: F401 - fixture

AD = ("We are looking for a Software Engineer for test automation. You build Java and Spring services, automate "
      "tests and work with Jenkins and Docker. ")


def fake_job(**kw):
    analysis = kw.pop("analysis", None)
    base = dict(id=None, title="Software Engineer Test Automation", company="Acme GmbH", location="Ulm", country="Germany",
                description=AD * 3, requirements="Required: Java, Spring, OpenShift and REST APIs.", responsibilities="",
                employment_type=None, salary_min=None, salary_max=None, currency=None, language_requirements=[],
                years_experience_required=None,
                current_analysis=SimpleNamespace(result=analysis, provider="local") if analysis else None)
    base.update(kw)
    return SimpleNamespace(**base)


ANALYSIS = {"strong_matches": [{"skill": "3+ years of Java and Spring experience"}, {"skill": "Python"}],
            "partial_matches": [{"skill": "Experience with OpenShift"}], "critical_gaps": ["OpenShift"],
            "nice_to_have_gaps": ["Azure"]}


# ------------------------------------------------------------------ keywords


def test_literal_matching_handles_symbols_and_boundaries():
    assert literal_in("C++", "Entwicklung in C++ und Python")
    assert literal_in("CI/CD", "Jenkins CI/CD pipelines")
    assert not literal_in("Java", "JavaScript developer")
    assert term_in("machine learning", "Erfahrung in maschinellem Lernen und Machine Learning")


def test_job_keywords_reduce_requirements_to_core_terms():
    profile = get_profile()
    kws = {k.term.lower(): k for k in job_keywords(fake_job(analysis=ANALYSIS), cv_vocabulary(get_master_cv(), profile))}
    assert kws["java"].priority == "required" and kws["spring"].priority == "required"
    assert kws["openshift"].priority == "required"
    assert kws["azure"].priority == "preferred"
    assert kws["docker"].priority == "preferred"  # only in the description
    assert "c" not in kws and "go" not in kws  # one/two-letter terms match too much


# ------------------------------------------------------------------ CV


def test_master_cv_builds_in_both_languages():
    master = get_master_cv()
    de, en = build_cv(master, "de", [], ""), build_cv(master, "en", [], "")
    assert de["sections"]["experience"] == "Berufserfahrung" and en["sections"]["experience"] == "Professional Experience"
    assert "Geboren am 01.01.1995" in de["personal"] and not any("1998" in p for p in en["personal"])
    assert de["experience"][0]["dates"] == "02/2026 - 06/2026"  # ASCII range, reverse-chronological order kept


def test_tailoring_orders_by_relevance_and_surfaces_only_real_skills():
    job = fake_job(analysis=ANALYSIS)
    with session_scope() as s:
        cv, calls = tailor_cv(s, job, "en", None, get_profile())
    assert calls == [] and cv["meta"]["statement_source"] == "fallback"
    mbti = next(r for r in cv["experience"] if r["id"] == "mbti")
    assert "Java/Spring" in mbti["bullets"][0]
    text = render_text(cv)
    assert "REST APIs" in cv["meta"]["extra_skills"]  # the profile has it, the CV didn't say it
    assert "OpenShift" not in text and "Azure" not in text  # gaps are never stuffed in
    assert cv["skills"][-1]["id"] == "extra"


def test_tailored_statement_from_the_writer_is_checked(make_setup_cv):
    setup = make_setup_cv(statement="Software engineer with Java/Spring test automation and Jenkins CI/CD. "
                                    "Five years of OpenShift operations at scale. 7 years of experience in Azure.")
    with session_scope() as s:
        cv, calls = tailor_cv(s, fake_job(analysis=ANALYSIS), "en", setup, get_profile())
    assert cv["meta"]["statement_source"] == "writer" and len(calls) == 1
    assert "OpenShift" not in cv["profile_statement"] and "Azure" not in cv["profile_statement"]
    assert len(cv["meta"]["statement_removed"]) == 2


def test_check_statement_drops_invented_numbers_and_year_claims():
    master_text = get_master_cv().full_text("de")
    clean, removed = check_statement("Softwareentwickler mit Java/Spring. Über 500 Testfälle automatisiert. "
                                     "Mehrjährige Erfahrung in KI.", master_text, get_profile())
    assert clean == "Softwareentwickler mit Java/Spring." and len(removed) == 2


def test_renderers_produce_pdf_and_escaped_latex():
    cv = build_cv(get_master_cv(), "de", [], "Profil mit C# & Python.")
    pdf = render_pdf(cv)
    assert pdf.startswith(b"%PDF")
    tex = render_tex(cv)
    assert r"\documentclass" in tex and r"C\# \& Python" in tex and r"\section{Berufserfahrung}" in tex
    letter = render_letter_pdf("Sehr geehrte Damen und Herren,\n\nText mit Umlauten: äöüß.\n\nMit freundlichen Grüßen\nAlex Beispiel",
                               "Bewerbung", cv, ["Acme GmbH", "Ulm"], "19.10.2026")
    assert letter.startswith(b"%PDF")


# ------------------------------------------------------------------ ATS


def test_ats_reads_the_pdf_text_layer():
    profile = get_profile()
    job = fake_job(analysis=ANALYSIS)
    with session_scope() as s:
        cv, _ = tailor_cv(s, job, "de", None, profile)
    keywords = job_keywords(job, cv_vocabulary(get_master_cv(), profile))
    report = ats.check(cv, render_pdf(cv), keywords, job, profile)
    assert all(c["ok"] for c in report["parse_checks"]), report["parse_checks"]
    status = {r["term"].lower(): r["status"] for r in report["keywords"]}
    assert status["java"] == "covered" and status["rest apis"] == "covered"
    assert status["openshift"] == "missing_gap"
    assert 0 < report["score"] <= 100 and "OpenShift" in ats.summary(report)


def test_ats_knockouts_for_languages_and_years():
    profile = get_profile()
    job = fake_job(language_requirements=["Fluent French required", "German (C1)"], years_experience_required=8)
    cv = build_cv(get_master_cv(), "en", [], "")
    report = ats.check(cv, render_pdf(cv), [], job, profile)
    assert any("French" in k for k in report["knockouts"]) and any("8+ years" in k for k in report["knockouts"])
    assert not any("German" in k for k in report["knockouts"])


# ------------------------------------------------------------------ verdict


def ats_report(score=85, knockouts=(), missing_have=()):
    return {"score": score, "parse_checks": [{"id": "email", "ok": True, "label": "E-mail", "detail": ""}],
            "knockouts": list(knockouts), "missing_have": list(missing_have), "missing_gap": [], "keywords": []}


def recruiter(verdict="SEND", cv=8, letter=8, red_flags=()):
    return {"verdict": verdict, "cv_score": cv, "letter_score": letter, "fit_score": 8, "red_flags": list(red_flags),
            "verdict_reason": "ok", "cv_changes": ["Name MLOps"], "letter_changes": ["Be specific"]}


def test_send_needs_ats_recruiter_and_no_red_flags():
    assert decide(ats_report(), recruiter(), {"warnings": []})["verdict"] == "SEND"
    assert decide(ats_report(score=60), recruiter(), None)["verdict"] == "REVISE"
    assert decide(ats_report(), recruiter(letter=5), None)["verdict"] == "REVISE"
    flagged = decide(ats_report(), recruiter(red_flags=["Letter names the wrong company"]), None)
    assert flagged["verdict"] == "REVISE" and any("wrong company" in r for r in flagged["reasons"])
    blocked = decide(ats_report(), recruiter(), {"warnings": [{"type": "placeholder", "detail": "[Firma] left"}]})
    assert blocked["verdict"] == "REVISE" and blocked["fixable"]


def test_knockouts_alone_are_not_fixable_by_revising():
    d = decide(ats_report(knockouts=["Requires fluent French"]), recruiter(), None)
    assert d["verdict"] == "REVISE" and d["fixable"] is False


def test_revision_plan_only_uses_supported_keywords():
    from app.services.keywords import Keyword

    plan = revision_plan(ats_report(missing_have=["REST APIs"]), recruiter("REVISE"),
                         [Keyword("REST APIs", "required", "posting")], "Letter without it")
    assert plan["add_keywords"] == ["REST APIs"] and plan["letter_keywords"] == ["REST APIs"]
    assert plan["cv_changes"] == ["Name MLOps"] and plan["letter_changes"] == ["Be specific"]


def test_application_email_reuses_the_letter_salutation():
    job = fake_job()
    mail = agent.draft_email(job, "Sehr geehrte Frau Weber,\n\nText", "de", "Alex Beispiel",
                             {"email": "a@b.de", "phone": "+49 1"})
    assert mail["body"].startswith("Sehr geehrte Frau Weber,") and "Software Engineer Test Automation" in mail["subject"]
    assert mail["attachments"] == ["Anschreiben_Alex_Beispiel.pdf", "Lebenslauf_Alex_Beispiel.pdf"]


# ------------------------------------------------------------------ end to end


@pytest.fixture
def make_setup_cv():
    def build(statement="Softwareentwickler mit Erfahrung in Java-/Spring-Testautomatisierung und Jenkins-CI/CD.",
              verdicts=("REVISE", "SEND")):
        setup = make_setup()
        setup.provider.responses["CvProfileStatement"] = CvProfileStatement(statement=statement, focus_terms=["Java"])
        queue = list(verdicts)

        def review(_user):
            v = queue.pop(0) if queue else verdicts[-1]
            return RecruiterReview(
                first_impression="Solid test-automation profile.", cv_score=8 if v == "SEND" else 6,
                letter_score=8 if v == "SEND" else 5, fit_score=7, strengths=["Java/Spring framework"],
                concerns=[] if v == "SEND" else ["Letter is generic"], red_flags=[],
                cv_changes=[] if v == "SEND" else ["Lead with the test automation work"],
                letter_changes=[] if v == "SEND" else ["Name the posting's Jenkins pipelines explicitly"],
                verdict=v, verdict_reason="ok" if v == "SEND" else "letter too generic")

        setup.provider.responses["RecruiterReview"] = review
        return setup

    return build


def test_full_run_revises_until_send(job_id, make_setup_cv, monkeypatch):  # noqa: F811
    monkeypatch.setattr(get_prefs().application, "ats_send_threshold", 0)
    setup = make_setup_cv()
    setup.provider.texts = [
        "Sehr geehrte Damen und Herren,\n\nbei Nordlicht Mobility Software habe ich ein Java/Spring-basiertes "
        "Validierungsframework mitentwickelt.\n\nMit freundlichen Grüßen\n[Name]",  # German editor
        "Sehr geehrte Damen und Herren,\n\nbei Nordlicht Mobility Software habe ich ein Java/Spring-basiertes "
        "Validierungsframework mitentwickelt; Ihre Jenkins-Pipelines kenne ich aus dieser Arbeit.\n\n"
        "Mit freundlichen Grüßen\n[Name]",  # revision
    ]
    result = agent.run_application(job_id, language="de", setup=setup)
    assert result["verdict"] == "SEND" and result["rounds"] == 2
    with session_scope() as s:
        pkg = s.get(ApplicationPackage, result["package"])
        assert pkg.status == "done" and pkg.verdict == "SEND" and len(pkg.reviews) == 2
        assert pkg.reviews[0]["verdict"] == "REVISE" and pkg.reviews[0]["revision"]["letter_changes"]
        stages = [d.stage for d in s.query(ApplicationDocument).filter_by(group_id=pkg.letter_group_id).order_by(ApplicationDocument.id)]
        assert "revised" in stages and stages[-1] == "final"
        assert pkg.email["subject"] == "Bewerbung um die Stelle als Softwareentwickler Testautomatisierung"
        cv_pdf, name, media = agent.package_file(s, pkg, "cv.pdf")
        assert cv_pdf.startswith(b"%PDF") and name == "Lebenslauf_Alex_Beispiel.pdf"
        assert agent.package_file(s, pkg, "letter.pdf")[0].startswith(b"%PDF")


def test_round_limit_ends_with_revise(job_id, make_setup_cv, monkeypatch):  # noqa: F811
    monkeypatch.setattr(get_prefs().application, "max_revision_rounds", 1)
    setup = make_setup_cv(verdicts=("REVISE",))
    result = agent.run_application(job_id, language="de", setup=setup)
    assert result["verdict"] == "REVISE" and result["rounds"] == 2


def test_failed_run_marks_the_package(job_id, monkeypatch):  # noqa: F811
    setup = make_setup()

    def boom(*a, **k):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(agent, "generate_letter", boom)
    with pytest.raises(RuntimeError):
        agent.run_application(job_id, language="de", setup=setup)
    with session_scope() as s:
        pkg = s.query(ApplicationPackage).one()
        assert pkg.status == "failed" and "model crashed" in pkg.error


def test_pages_and_downloads(job_id, make_setup_cv, monkeypatch):  # noqa: F811
    monkeypatch.setattr(get_prefs().application, "ats_send_threshold", 0)
    result = agent.run_application(job_id, language="de", setup=make_setup_cv(verdicts=("SEND",)))
    client = TestClient(app)
    page = client.get(f"/applications/{result['package']}")
    assert page.status_code == 200 and "Good — send it." in page.text and "ATS check" in page.text
    assert client.get("/applications").status_code == 200
    job_page = client.get(f"/jobs/{job_id}").text
    assert "Complete application (PDF)" in job_page and "Mark as applied" in job_page
    pdf = client.get(f"/applications/{result['package']}/file/cv.pdf")
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
    assert client.get(f"/applications/{result['package']}/file/cv.tex").text.startswith("% Tailored CV")
    assert client.get(f"/applications/{result['package']}/file/secret.sh").status_code == 404


def test_letter_buttons_wait_for_the_agent(job_id, monkeypatch):  # noqa: F811
    monkeypatch.setattr(agent.tasks, "is_running", lambda name: name == f"apply-{job_id}")
    r = TestClient(app).post(f"/jobs/{job_id}/letters/generate", data={"language": "de"}, follow_redirects=False)
    assert r.status_code == 303 and "application+agent" in r.headers["location"].replace("%20", "+")


def test_interrupted_packages_are_marked_failed(job_id):  # noqa: F811
    with session_scope() as s:
        s.add(ApplicationPackage(job_id=job_id, status="running", language="de", reviews=[]))
    assert agent.mark_interrupted() == 1
    with session_scope() as s:
        assert s.query(ApplicationPackage).one().status == "failed"


# ------------------------------------------------------------------ letter structure (the applicant's instruction)


def test_writer_prompts_follow_the_letter_outline():
    from app.config_files import get_prefs as prefs
    from app.llm.letter_prompts import GLIEDERUNG_DE, editor_system_de, writer_system_de, writer_system_en

    assert GLIEDERUNG_DE in writer_system_de(prefs().writing) and GLIEDERUNG_DE in writer_system_de(prefs().writing, examples=True)
    order = ["1. Persönliche Anrede", "2. Einstieg", "3. Persönliche Vorstellung", "4. Abschluss", "5. Bitte um Rückmeldung",
             "6. Grußformel"]
    assert [GLIEDERUNG_DE.index(x) for x in order] == sorted(GLIEDERUNG_DE.index(x) for x in order)
    assert "Vorstellungsgespräch" in GLIEDERUNG_DE and "Anlagenliste" in GLIEDERUNG_DE
    assert "Request for a reply" in writer_system_en(prefs().writing)
    assert "paragraph ORDER" in editor_system_de(prefs().writing)


def test_subject_and_contact_person_for_the_letter_layout():
    job = fake_job(title="Softwareentwickler Testautomatisierung (m/w/d)")
    assert agent.subject_line(job, "de") == "Bewerbung um die Stelle als Softwareentwickler Testautomatisierung"
    assert agent.subject_line(fake_job(title="AI Engineer (f/m/x)"), "en") == "Application for the position of AI Engineer"
    assert agent.contact_person("Sehr geehrter Herr Lang,\n\nText") == "Herrn Lang"
    assert agent.contact_person("Sehr geehrte Frau Dr. Weber,\n\nText") == "Frau Dr. Weber"
    assert agent.contact_person("Sehr geehrte Damen und Herren,\n\nText") == ""


def test_letter_pdf_has_sender_block_recipient_and_attachments():
    cv = build_cv(get_master_cv(), "de", [], "")
    cv["contact"] = {**cv["contact"], "street": "Musterweg 1", "postal_code": "88045"}
    pdf = render_letter_pdf("Sehr geehrter Herr Lang,\n\nText.\n\nMit freundlichen Grüßen\nAlex Beispiel",
                            "Bewerbung um die Stelle als Entwickler", cv, ["Acme GmbH", "", "Herrn Lang", "Ulm"],
                            "19.10.2026", ["Lebenslauf", "Zeugnisse"])
    text, pages = ats.pdf_text(pdf)
    assert pages == 1
    for part in ("Musterweg 1", "88045 Friedrichshafen", "Alex Beispiel, Musterweg 1, 88045 Friedrichshafen",
                 "Herrn Lang", "19.10.2026", "Bewerbung um die Stelle als Entwickler", "Anlagen", "Zeugnisse"):
        assert part in text, part
    # without street / postcode (not filled in yet) nothing is invented
    empty = render_letter_pdf("Text", None, build_cv(get_master_cv(), "de", [], ""), ["Acme GmbH"], "19.10.2026")
    assert "Friedrichshafen" in ats.pdf_text(empty)[0] and "None" not in ats.pdf_text(empty)[0]


def test_return_address_line_never_contains_the_phone_number():
    text = ats.pdf_text(render_letter_pdf("Text", None, build_cv(get_master_cv(), "de", [], ""), ["Acme"], "19.10.2026"))[0]
    assert "Alex Beispiel, Friedrichshafen\n" in text or text.count("+49") == 1


# ------------------------------------------------------------------ regressions from the first live run (job 262)

SIM_JOB = ("Simulation Engineer NVIDIA Omniverse. You build high-fidelity simulation scenes for virtual commissioning, "
           "prepare CAD data and handle the kinematization of production equipment with Isaac Sim and OpenUSD.")


def test_statement_copying_the_postings_duties_is_dropped():
    master_text = get_master_cv().full_text("en")
    statement = ("Experienced in virtual commissioning and simulation for industrial automation, with a focus on "
                 "preparing engineering data. Developed a Java/Spring validation framework with automated test cases.")
    clean, removed = check_statement(statement, master_text, get_profile(), SIM_JOB)
    assert clean == "Developed a Java/Spring validation framework with automated test cases."
    assert removed and "virtual commissioning" in removed[0]
    kinem = "Built simulation environments including kinematization of production equipment from CAD data."
    assert check_statement(kinem, master_text, get_profile(), SIM_JOB)[0] == ""


def test_a_genuine_gap_alone_does_not_force_revise():
    report = {**ats_report(), "missing_gap": ["Omniverse", "Isaac Sim"]}
    rec = {**recruiter("REVISE", cv=8, letter=7),
           "verdict_reason": "Candidate lacks direct experience with required simulation environments",
           "red_flags": ["Missing core requirement: no Omniverse/Isaac Sim experience",
                         "ATS report shows genuine gap in required simulation environments"]}
    d = decide(report, rec, None)
    assert d["verdict"] == "SEND", d["reasons"]
    assert any("Omniverse" in n for n in d["gap_notes"])
    # a problem with the documents themselves still counts
    rec["red_flags"].append("Cover letter doesn't connect the AI skills to the simulation tooling")
    assert decide(report, rec, None)["verdict"] == "REVISE"


def test_cv_changes_never_add_a_gap_skill():
    rec = {**recruiter("REVISE"), "cv_changes": ["Add 'NVIDIA Omniverse' and 'Isaac Sim' to the skills section",
                                                  "Move the test automation work to the top"]}
    plan = revision_plan({**ats_report(), "missing_gap": ["Omniverse", "Isaac Sim"]}, rec, [], "letter")
    assert plan["cv_changes"] == ["Move the test automation work to the top"]


def test_dangling_clause_after_removal_is_tidied():
    from app.services.letters import tidy_after_removal

    text = "Dear Hiring Team,\n\n, and I am eager to learn these tools. My degree helps.\n\nKind regards"
    assert tidy_after_removal(text) == "Dear Hiring Team,\n\nI am eager to learn these tools. My degree helps.\n\nKind regards"
    assert tidy_after_removal("Gut. , so lerne ich schnell.") == "Gut. So lerne ich schnell."


def test_the_best_round_is_kept_and_a_worse_revision_stops_the_loop(job_id, make_setup_cv, monkeypatch):  # noqa: F811
    monkeypatch.setattr(get_prefs().application, "max_revision_rounds", 3)
    setup = make_setup_cv(verdicts=("REVISE",))
    scores = iter([(7, 6), (6, 5), (6, 5), (6, 5)])  # round 1 best, round 2 worse → stop

    def review(_user):
        cv_s, letter_s = next(scores)
        return RecruiterReview(first_impression="ok", cv_score=cv_s, letter_score=letter_s, fit_score=6, strengths=[],
                               concerns=[], red_flags=[], cv_changes=["Lead with testing"], letter_changes=["Be concrete"],
                               verdict="REVISE", verdict_reason="needs work")

    setup.provider.responses["RecruiterReview"] = review
    result = agent.run_application(job_id, language="de", setup=setup)
    assert result["rounds"] == 2 and result["best_round"] == 1 and result["recruiter"] == 6.5
    with session_scope() as s:
        pkg = s.get(ApplicationPackage, result["package"])
        assert pkg.reviews[0]["best"] and pkg.reviews[1]["stopped"]
        letter = agent.package_letter(s, pkg.job_id, pkg.letter_group_id, pkg.reviews)
        assert letter.id == pkg.reviews[0]["letter_document_id"]  # not the later (worse) revision


@pytest.mark.parametrize("invented", [
    "Erfahrung in der Entwicklung von Simulationsumgebungen für industrielle Automatisierung sowie Kenntnisse im "
    "Umgang mit CAD-Daten und deren Vorbereitung für Simulationen.",
    "Verwaltung von CAD-Daten und Erstellung strukturierter Szenen für industrielle Anwendungen.",
])
def test_german_statement_inventing_work_is_dropped_even_for_english_postings(invented):
    master_text = get_master_cv().full_text("de")
    assert check_statement(invented, master_text, get_profile(), SIM_JOB)[0] == ""


@pytest.mark.parametrize("lang,honest", [
    ("de", "Softwareentwickler mit M.Sc. Intelligent Systems und Berufserfahrung in Java-/Spring-Entwicklung, "
           "automatisierten Tests und Jenkins-CI/CD-Pipelines."),
    ("de", "Masterarbeit zu einer modularen Pipeline mit Klassifikationsmodellen und LLM-basierter Analyse und Planung."),
    ("en", "Software engineer who built a Java/Spring validation framework for ECUs and a Flask/Plotly dashboard."),
])
def test_honest_statements_survive_the_check(lang, honest):
    assert check_statement(honest, get_master_cv().full_text(lang), get_profile())[0] == honest


def test_fallback_statements_pass_their_own_check():
    from app.services.cv import fallback_statement

    for lang in ("de", "en"):
        s = fallback_statement(lang)
        assert check_statement(s, get_master_cv().full_text(lang), get_profile())[0] == s


def test_closing_has_no_comma_and_the_writer_knows_today():
    from app.services.letters import ensure_signature, today_line

    assert ensure_signature("Text\n\nMit freundlichen Grüßen,\nAlex Beispiel", "Alex Beispiel").endswith(
        "Mit freundlichen Grüßen\nAlex Beispiel")
    assert "Heutiges Datum" in today_line("de") and "past tense" in today_line("en")


def test_single_invented_detail_drops_the_sentence():
    s = ("Softwareentwickler mit Schwerpunkt auf Validierungsframeworks und automatisierten Testprozessen für Steuergeräte "
         "sowie Erfahrung in der Integration von Hardware in Simulationsumgebungen.")
    assert check_statement(s, get_master_cv().full_text("de"), get_profile())[0] == ""


# ------------------------------------------------------------------ certificates, one-page fit, two Mercedes roles


def test_master_cv_matches_the_references():
    master = get_master_cv()
    roles = {r.id: (r.start, r.end) for r in master.experience}
    assert roles["mbti"] == ("06/2023", "05/2025") and roles["mbti-intern"] == ("12/2022", "05/2023")
    msc = next(e for e in master.education if e.id == "msc")
    assert msc.start == "09/2023" and "1,3" in str(msc.thesis.label) and msc.project is not None


def test_cv_is_cut_to_one_page_by_relevance():
    from app.services.cv import fit_to_page
    from app.services.cv_render import page_count

    job = fake_job(analysis=ANALYSIS)
    kws = job_keywords(job, cv_vocabulary(get_master_cv(), get_profile()))
    cv = fit_to_page(get_master_cv(), "de", kws, "Softwareentwickler mit Java/Spring.", ["REST APIs"])
    assert page_count(cv) == 1
    text = render_text(cv)
    assert "Java-/Spring-Validierungsframeworks" in text and "Praktikant" in text  # core content never cut


def test_complete_application_pdf_bundles_letter_cv_and_certificates(tmp_path, monkeypatch):
    import io

    from pypdf import PdfReader, PdfWriter

    from app.config_files import ApplicationDocumentRef

    def pdf(pages: int) -> bytes:
        w = PdfWriter()
        for _ in range(pages):
            w.add_blank_page(width=200, height=200)
        buf = io.BytesIO()
        w.write(buf)
        return buf.getvalue()

    (tmp_path / "Zeugnis.pdf").write_bytes(pdf(4))
    monkeypatch.setattr(agent, "documents_dir", lambda: tmp_path)
    monkeypatch.setattr(get_prefs().application, "documents", [
        ApplicationDocumentRef(file="Zeugnis.pdf", label_de="Arbeitszeugnis", skip_pages=[1]),
        ApplicationDocumentRef(file="missing.pdf", label_de="Fehlt")])
    found, missing = agent.available_documents()
    assert [r.file for r, _ in found] == ["Zeugnis.pdf"] and missing == ["missing.pdf"]
    assert agent.attachment_labels("de") == ["Lebenslauf", "Arbeitszeugnis"]
    bundle = agent.bundle_pdf([pdf(1), pdf(1)], found)
    assert len(PdfReader(io.BytesIO(bundle)).pages) == 1 + 1 + 3  # letter + CV + reference without its cover note
    mail = agent.draft_email(fake_job(), "Sehr geehrte Damen und Herren,", "de", "Alex Beispiel", {})
    assert mail["attachments"] == ["Bewerbung_Alex_Beispiel_Acme_GmbH.pdf"] and "Zeugnisse" in mail["body"]


def test_gender_marker_is_removed_anywhere_in_the_subject():
    assert agent.subject_line(fake_job(title="Simulation Engineer (gn) NVIDIA Omniverse"), "de") == \
        "Bewerbung um die Stelle als Simulation Engineer NVIDIA Omniverse"


def test_old_packages_use_the_current_contact_details():
    old_cv = build_cv(get_master_cv(), "de", [], "")
    old_cv["contact"] = {"email": "old@example.org", "phone": "", "street": "", "postal_code": "", "location": "", "links": []}
    cv = agent.current_cv(SimpleNamespace(cv=old_cv))
    assert cv["contact"]["email"] == get_master_cv().contact.email and cv["experience"] == old_cv["experience"]


def test_core_terms_keep_skills_listed_in_two_casings():
    """cv.yaml keywords are lower-case ("java"), profile skills are not ("Java"): both used to cancel each other out."""
    from app.services.keywords import _core_terms, job_keywords

    vocab = ["java", "Java", "spring", "Spring", "python", "Python"]
    assert sorted(_core_terms("Java/Spring", vocab)) == ["Java", "Spring"]
    assert _core_terms("3+ years of Python experience", vocab) == ["Python"]
    analysis = SimpleNamespace(result={"strong_matches": [{"skill": "Java/Spring backend development"}]})
    job = SimpleNamespace(current_analysis=analysis, title="Backend Developer", requirements="", responsibilities="",
                          description="")
    terms = {k.term for k in job_keywords(job, vocab)}
    assert {"Java", "Spring"} <= terms


def test_export_writes_the_papers_and_zips_them(tmp_path, monkeypatch):
    import zipfile
    from io import BytesIO

    from app.services import export
    from app.services.cv import build_cv, get_master_cv
    from app.services.ingest import ingest_posting
    from app.sources.base import JobPosting
    from app.enums import SourceType

    monkeypatch.setattr(export, "export_root", lambda: tmp_path)
    monkeypatch.setattr(agent, "package_file", lambda s, pkg, kind: (b"%PDF " + kind.encode(), f"{kind}", "application/pdf"))
    with session_scope() as s:
        job = ingest_posting(s, JobPosting(source="manual", source_type=SourceType.MANUAL, title="Data Engineer (m/w/d)",
                                           company="Export GmbH", location="Ulm", description="Bewerbung an jobs@export.example")).job
        pkg = ApplicationPackage(job_id=job.id, status="done", language="de", verdict="REVISE", rounds=1,
                                 reviews=[{"best": True, "reasons": ["Cover letter: too generic"]}],
                                 cv=build_cv(get_master_cv(), "de", [], "Profil."),
                                 email={"subject": "Bewerbung", "body": "Sehr geehrte Damen und Herren, …"})
        s.add(pkg)
        s.flush()
        pkg_id = pkg.id
    folder = export.export_package(pkg_id)
    assert folder.parent.name == "needs review" and "Export GmbH - Data Engineer" in folder.name
    names = sorted(p.name for p in folder.iterdir())
    assert names == ["E-Mail.txt", "bundle.pdf", "cv.pdf", "info.txt", "letter.pdf"]
    assert "too generic" in (folder / "info.txt").read_text(encoding="utf-8")
    with session_scope() as s:  # verdict changes → the folder moves, no stale copy
        s.get(ApplicationPackage, pkg_id).verdict = "SEND"
    moved = export.export_package(pkg_id)
    assert moved.parent.name == "ready to send" and not folder.exists()
    with zipfile.ZipFile(BytesIO(export.zip_all())) as z:
        assert any(n.startswith("ready to send/") and n.endswith("bundle.pdf") for n in z.namelist())


def test_letters_written_before_the_guards_render_without_their_header():
    from types import SimpleNamespace as NS

    old = ("Sehr geehrte Damen und Herren,\n\nAlex Beispiel\nUlm, Deutschland\n+49 123 456 789 · alex.beispiel@example.com"
           "\n\nBROSIS Engineering GmbH\nFrau Weber\nLudwigstraße 12\n88099 Neukirch\n\nNeukirch, 19.10.2026\n\n"
           "Bewerbung als Softwareentwickler (m/w/d)\n\nSehr geehrte Frau Weber,\n\nIhre Stellenanzeige hat mich "
           "angesprochen, da Sie einen Softwareentwickler mit Erfahrung in Java suchen.\n\nMit freundlichen Grüßen\nAlex Beispiel")
    job = NS(title="Softwareentwickler", company="BROSIS Engineering GmbH", description="Wir suchen Verstärkung.",
             requirements="", responsibilities="", location="Neukirch")
    text = agent.layout_text(job, NS(content=old, language="de"))
    assert text.startswith("Sehr geehrte Damen und Herren,\n\nIhre Stellenanzeige")  # "Frau Weber" is not in the posting
    assert "example.com" not in text and "Ludwigstraße" not in text and text.count("Sehr geehrte") == 1
    assert agent.contact_person(text) == ""


def test_keyword_matching_tolerates_plurals():
    from app.services.keywords import literal_in, plural_variants, term_in

    assert term_in("api", "Built REST APIs with Flask") and term_in("APIs", "a REST API")
    assert term_in("databases", "SQLite database") and term_in("pipeline", "CI/CD pipelines")
    assert not literal_in("api", "REST APIs")  # the strict matcher (fact checks) stays strict
    assert plural_variants("C#") == [] and plural_variants("Go") == []


def test_partial_matches_with_professional_evidence_for_own_skills_become_strong():
    from app.services.analysis import promote_direct_matches

    result = {"strong_matches": [], "partial_matches": [
        {"skill": "Java", "evidence": "direct_professional", "profile_evidence": "Java"},
        {"skill": "Spring", "evidence": "direct_professional", "profile_evidence": "Spring"},
        {"skill": "Kotlin", "evidence": "transferable", "profile_evidence": "Java"}]}
    assert promote_direct_matches(result, get_profile()) == ["Java", "Spring"]
    assert [m["skill"] for m in result["strong_matches"]] == ["Java", "Spring"]
    assert [m["skill"] for m in result["partial_matches"]] == ["Kotlin"]


def test_a_finished_run_keeps_its_export_record_next_to_the_run_log(job_id, make_setup_cv, monkeypatch, tmp_path):  # noqa: F811
    from app.services import export

    monkeypatch.setattr(export, "export_root", lambda: tmp_path)
    monkeypatch.setattr(get_prefs().application, "ats_send_threshold", 0)
    result = agent.run_application(job_id, language="de", setup=make_setup_cv(verdicts=("SEND",)))
    with session_scope() as s:
        meta = s.get(ApplicationPackage, result["package"]).meta
    assert meta["export"]["dir"].startswith("ready to send/") and (tmp_path / meta["export"]["dir"]).is_dir()
    assert meta["stages"] and "seconds_total" in meta


# ------------------------------------------------------------------ your feedback → revision + memory


@pytest.fixture
def memory_file(tmp_path, monkeypatch):
    from app.services import agent_memory

    monkeypatch.setattr(agent_memory, "memory_path", lambda: tmp_path / "agent_memory.yaml")
    return tmp_path / "agent_memory.yaml"


def test_agent_memory_dedupes_and_steers_the_prompts(memory_file):
    from app.services import agent_memory

    added = agent_memory.add_rules([{"rule": "Keep letters under 300 words", "applies_to": "letter"},
                                    {"rule": "keep letters under 300 words!", "applies_to": "letter"},
                                    {"rule": "Never list Unity", "applies_to": "cv", "cv_exclude": ["Unity"]}])
    assert [r.text for r in added] == ["Keep letters under 300 words", "Never list Unity"]
    assert "under 300 words" in agent_memory.instructions_block("letter")
    assert "Never list Unity" not in agent_memory.instructions_block("letter")
    assert "Never list Unity" in agent_memory.instructions_block("review") and agent_memory.cv_excludes() == ["Unity"]
    assert agent_memory.delete_rule(added[0].id) and len(agent_memory.load_rules()) == 1
    assert memory_file.read_text(encoding="utf-8").startswith("# What the application agent learned")


def test_cv_items_can_be_left_out_and_emphasised():
    master = get_master_cv()
    skill = master.skills[0].items[0].split("|")[-1].strip()
    full = render_text(build_cv(master, "en", [], "Statement here."))
    assert skill in full
    assert skill not in render_text(build_cv(master, "en", [], "Statement here.", exclude=[skill]))


def test_feedback_revises_the_package_saves_notes_and_learns(job_id, make_setup_cv, monkeypatch, memory_file):  # noqa: F811
    from app.services.agent_memory import FeedbackPlan, GeneralRule
    from app.services.application_notes import get_notes

    monkeypatch.setattr(get_prefs().application, "ats_send_threshold", 0)
    setup = make_setup_cv(verdicts=("SEND", "REVISE"))
    result = agent.run_application(job_id, language="de", setup=setup)
    setup.provider.responses["FeedbackPlan"] = FeedbackPlan(
        letter_changes=["Make the letter shorter"], cv_changes=["Lead with test automation"], cv_exclude=["Unity"],
        job_avoid="Don't mention the salary", general_rules=[GeneralRule(rule="Keep letters short", applies_to="letter")])
    setup.provider.texts = ["Sehr geehrte Damen und Herren,\n\nkürzer.\n\nMit freundlichen Grüßen\n[Name]"]
    out = agent.revise_with_feedback(result["package"], "Zu lang. Unity bitte weglassen.", learn=True, setup=setup)
    assert out["learned"] == ["Keep letters short"]
    with session_scope() as s:
        pkg = s.get(ApplicationPackage, result["package"])
        last = pkg.reviews[-1]
        assert pkg.status == "done" and last["user_revision"] and last["best"] and pkg.verdict == last["verdict"] == "REVISE"
        assert pkg.cv["meta"]["revision"]["exclude"] == ["Unity"]  # your round wins over the higher-scoring SEND round
        fb = pkg.meta["feedback"][-1]
        assert fb["text"].startswith("Zu lang") and fb["learned"] == ["Keep letters short"]
        assert "salary" in get_notes(s, job_id).avoid
        assert "The applicant asks: Make the letter shorter" in setup.provider.last_user["text"] or any(
            d.stage == "revised" for d in s.query(ApplicationDocument).filter_by(group_id=pkg.letter_group_id))
    page = TestClient(app).get(f"/applications/{result['package']}")
    assert "Tell the agent what to change" in page.text and "Keep letters short" in page.text
    assert "Keep letters short" in TestClient(app).get("/agent-memory").text


def test_a_failed_revision_keeps_the_package(job_id, make_setup_cv, monkeypatch, memory_file):  # noqa: F811
    monkeypatch.setattr(get_prefs().application, "ats_send_threshold", 0)
    setup = make_setup_cv(verdicts=("SEND",))
    result = agent.run_application(job_id, language="de", setup=setup)
    monkeypatch.setattr(agent, "tailor_cv", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        agent.revise_with_feedback(result["package"], "Shorter please", learn=False, setup=setup)
    with session_scope() as s:
        pkg = s.get(ApplicationPackage, result["package"])
        assert pkg.status == "done" and pkg.verdict == "SEND" and pkg.meta["feedback"][-1]["error"] == "boom"


def test_feedback_terms_match_cv_items_by_their_core_words():
    from app.services.cv import _excluded, core_words

    assert core_words("Unity-Projekt") == ["unity"] and core_words("Robotik-Erfahrung") == ["robotik"]
    assert _excluded("Published several Unity Asset Store packages", ["Unity-Projekt"])
    assert not _excluded("Community management", ["Unity"])  # whole words, not inside other words
