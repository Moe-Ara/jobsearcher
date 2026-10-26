"""ATS check — the mechanical half of the reviewer agent.

An applicant tracking system reads the PDF's embedded text layer, not the rendered page. This check extracts the
tailored CV PDF's text (pypdf) and verifies it the way a parser sees it, then scores the posting's keywords against
it (the honest way: a keyword the candidate does not have is reported as a gap, never stuffed in).

  parse checks   text extracted, no garbage glyphs, name/email/phone present as literal text, standard section
                 headings in reading order, every role/degree has an ASCII start-end date range, page count
  keywords       covered / missing but supported by the profile ("add it") / missing and a genuine gap
  knock-outs     required languages, years of experience, critical gaps from the fit analysis
  score          0-100: 65 % keyword match (required ×2), 35 % parse checks, minus knock-out penalties
"""

from __future__ import annotations

import io
import re
from typing import Any

from app.config_files import CandidateProfile
from app.services.cv import profile_skill_for
from app.services.keywords import Keyword, term_in

MAX_PAGES = 2
_GARBAGE = re.compile(r"\(cid:\d+\)|�")
_DATE_RANGE = re.compile(r"\b(0[1-9]|1[0-2])/(19|20)\d{2}\s+-\s+((0[1-9]|1[0-2])/(19|20)\d{2}|heute|present|today)\b", re.I)
_UNICODE_DASH_DATE = re.compile(r"(19|20)\d{2}\s*[‐-―−]\s*\d")
_LEVELS = {"a1": 1, "a2": 2, "b1": 3, "b2": 4, "c1": 5, "c2": 6, "native": 7, "muttersprache": 7}
_REQ_LEVEL = [(re.compile(r"native|muttersprach", re.I), 7), (re.compile(r"\bc2\b|verhandlungssicher|business[- ]fluent", re.I), 6),
              (re.compile(r"\bc1\b|fluent|fließend|sehr gut|excellent", re.I), 5), (re.compile(r"\bb2\b|good|gut", re.I), 4)]
_LANG_NAMES = {"german": ("german", "deutsch"), "english": ("english", "englisch"), "arabic": ("arabic", "arabisch"),
               "french": ("french", "französisch"), "spanish": ("spanish", "spanisch"), "italian": ("italian", "italienisch"),
               "dutch": ("dutch", "niederländisch"), "polish": ("polish", "polnisch"), "chinese": ("chinese", "chinesisch",
               "mandarin"), "japanese": ("japanese", "japanisch"), "russian": ("russian", "russisch"),
               "turkish": ("turkish", "türkisch"), "portuguese": ("portuguese", "portugiesisch")}


def pdf_text(data: bytes) -> tuple[str, int]:
    """(text, pages) as an ATS parser sees it. Poppler's pdftotext (what many ATS products build on) when installed;
    pypdf otherwise — its spacing heuristic splits some XeTeX glyph runs ("W erkstudent"), pdftotext does not."""
    import shutil
    import subprocess
    import tempfile

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = len(reader.pages)
    if shutil.which("pdftotext"):
        with tempfile.TemporaryDirectory(prefix="js-ats-") as tmp:
            src = f"{tmp}/doc.pdf"
            with open(src, "wb") as fh:
                fh.write(data)
            try:
                proc = subprocess.run(["pdftotext", "-enc", "UTF-8", src, "-"], capture_output=True, timeout=60)
                if proc.returncode == 0 and proc.stdout.strip():
                    return proc.stdout.decode("utf-8", errors="replace"), pages
            except (OSError, subprocess.TimeoutExpired):
                pass
    return "\n".join((p.extract_text() or "") for p in reader.pages), pages


def _parse_checks(text: str, pages: int, cv: dict[str, Any]) -> list[dict[str, Any]]:
    c = cv["contact"]
    low = text.lower()
    checks = [
        ("text", "Text layer extracted", len(text.strip()) > 300, "No readable text — an ATS would see an empty CV"),
        ("glyphs", "No garbled characters", not _GARBAGE.search(text), "Garbled glyphs (cid / replacement characters)"),
        ("name", "Name as text", cv["name"].lower() in low or cv["name"].upper().lower() in low, "Name not found in the text layer"),
        ("email", "E-mail as literal text", bool(c.get("email")) and c["email"].lower() in low, "E-mail address missing from the text layer"),
        ("phone", "Phone number as text", bool(c.get("phone")) and re.sub(r"\D", "", c["phone"])[-8:] in re.sub(r"\D", "", text),
         "Phone number missing from the text layer"),
        ("pages", f"At most {MAX_PAGES} pages", pages <= MAX_PAGES, f"{pages} pages"),
        ("ascii_dates", "Date ranges use an ASCII hyphen", not _UNICODE_DASH_DATE.search(text),
         "A year is joined by a Unicode dash — some ATS imports drop such dates"),
    ]
    headings = [cv["sections"].get(k, "") for k in ("experience", "education", "skills")]
    positions = [low.find(h.lower()) for h in headings if h]
    checks.append(("headings", "Standard section headings, in order", all(p >= 0 for p in positions) and positions == sorted(positions),
                   "Section headings missing or out of order: " + ", ".join(headings)))
    entries = len(cv["experience"]) + len(cv["education"])
    found = len(_DATE_RANGE.findall(text))
    checks.append(("dates", "Every role and degree has a start-end date", found >= entries, f"{found} of {entries} date ranges recognised"))
    return [{"id": i, "label": label, "ok": bool(ok), "detail": "" if ok else detail} for i, label, ok, detail in checks]


def _language_knockouts(job, profile: CandidateProfile) -> list[str]:
    have: dict[str, int] = {}
    for lang in profile.languages:
        name = str(getattr(lang, "language", "")).lower()
        level = str(getattr(lang, "level", "")).lower()
        have[name] = max((v for k, v in _LEVELS.items() if k in level), default=4)
    out = []
    for req in job.language_requirements or []:
        r = str(req)
        name = next((n for n, aliases in _LANG_NAMES.items() if any(a in r.lower() for a in aliases)), None)
        if not name:
            continue
        need = next((lvl for rx, lvl in _REQ_LEVEL if rx.search(r)), 0)
        if re.search(r"plus|advantage|vorteil|wünschenswert|nice", r, re.I):
            continue
        if name not in have:
            out.append(f"Requires {r} — not in your languages")
        elif need and have[name] < need:
            out.append(f"Requires {r} — your level is lower")
    return out


def check(cv: dict[str, Any], pdf: bytes, keywords: list[Keyword], job, profile: CandidateProfile) -> dict[str, Any]:
    try:
        text, pages = pdf_text(pdf)
    except Exception as exc:  # noqa: BLE001 - an unreadable PDF is itself the finding
        text, pages = "", 0
        extract_error = str(exc)
    else:
        extract_error = None
    parse = _parse_checks(text, pages, cv)

    rows = []
    for k in keywords:
        if term_in(k.term, text):
            status = "covered"
        elif profile_skill_for(k.term, profile) or term_in(k.term, profile.evidence_corpus()):
            status = "missing_have"
        else:
            status = "missing_gap"
        rows.append({**k.as_dict(), "status": status})
    total = sum(2 if r["priority"] == "required" else 1 for r in rows)
    covered = sum(2 if r["priority"] == "required" else 1 for r in rows if r["status"] == "covered")
    match = round(100 * covered / total) if total else 100
    parse_score = round(100 * sum(c["ok"] for c in parse) / len(parse))

    knockouts = _language_knockouts(job, profile)
    years = getattr(job, "years_experience_required", None)
    have_years = profile.years_professional_experience
    if years and have_years is not None and years > have_years + 2:
        knockouts.append(f"Requires {years:g}+ years of experience — your profile shows about {have_years:g}")
    a = job.current_analysis.result if getattr(job, "current_analysis", None) and job.current_analysis.result else {}
    gaps = [str(g) for g in (a.get("critical_gaps") or [])]
    score = max(0, round(0.65 * match + 0.35 * parse_score) - min(30, 15 * len(knockouts)))
    return {
        "score": score, "keyword_match": match, "parse_score": parse_score, "pages": pages,
        "parse_checks": parse, "keywords": rows, "knockouts": knockouts, "critical_gaps": gaps,
        "missing_have": [r["term"] for r in rows if r["status"] == "missing_have"],
        "missing_gap": [r["term"] for r in rows if r["status"] == "missing_gap"],
        "extract_error": extract_error, "text_chars": len(text),
    }


def summary(report: dict[str, Any]) -> str:
    """Compact text version for the recruiter prompt."""
    failed = [c["label"] + (f" ({c['detail']})" if c["detail"] else "") for c in report["parse_checks"] if not c["ok"]]
    lines = [
        f"ATS score {report['score']}/100 (keyword match {report['keyword_match']}%, parse checks {report['parse_score']}%).",
        "Covered keywords: " + (", ".join(r["term"] for r in report["keywords"] if r["status"] == "covered") or "none"),
        "Missing but the candidate has them: " + (", ".join(report["missing_have"]) or "none"),
        "Missing (genuine gaps): " + (", ".join(report["missing_gap"]) or "none"),
        "Failed parse checks: " + ("; ".join(failed) or "none"),
        "Knock-outs: " + ("; ".join(report["knockouts"]) or "none"),
    ]
    return "\n".join(lines)
