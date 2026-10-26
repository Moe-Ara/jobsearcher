"""Reviewer agent — checks a complete application before you submit it, the way it will be read:

  1. as an ATS:        app.services.ats.check on the CV PDF's text layer (parse checks, keyword coverage, knock-outs)
  2. as a recruiter:   the validator model reads posting + CV + letter + the ATS report (fresh context, no drafts
                       history) and scores CV, letter and fit, lists red flags and concrete changes
  3. verdict:          SEND (good — send it) or REVISE (with the revision instructions the application agent applies)

The verdict is advisory: nothing is ever submitted automatically.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.config_files import CandidateProfile, get_prefs
from app.llm.application_prompts import RECRUITER_SYSTEM, RECRUITER_VERSION, recruiter_user
from app.llm.factory import LLMSetup, Role
from app.llm.schemas import RecruiterReview
from app.services import ats, tasks
from app.services.cv import render_text
from app.services.keywords import Keyword, term_in
from app.services.llm_calls import CallResult, llm_call

log = logging.getLogger(__name__)

_BLOCKING_LETTER_WARNINGS = {"unsupported", "placeholder"}
_GAP_WORDS = re.compile(r"\b(lacks?|lacking|missing|no (direct |hands-on |prior )?(experience|exposure)|without "
                        r"(direct )?experience|gap|not (have|has)|keine (direkte )?erfahrung|fehlt|fehlende)\b", re.I)
_DOCUMENT_WORDS = re.compile(r"\b(letter|anschreiben|cv|lebenslauf|typo|generic|tone|wording|phrase|frame|framing|"
                             r"structure|format|inconsisten)", re.I)


def is_gap_finding(text: str, gaps: list[str]) -> bool:
    """A finding about the candidate lacking something (fixable only by learning it), not about the documents."""
    if _DOCUMENT_WORDS.search(text):
        return False
    low = text.lower()
    return any(g.lower() in low for g in gaps) or bool(_GAP_WORDS.search(text))


def _clamp10(v: Any) -> int:
    try:
        return max(0, min(10, int(round(float(v)))))
    except (TypeError, ValueError):
        return 0


def recruiter_review(session, job, cv_text: str, letter: str, ats_report: dict[str, Any], letter_warnings: list[str],
                     setup: LLMSetup | None, research_text: str = "") -> tuple[dict[str, Any] | None, CallResult | None]:
    target = setup.target(Role.VALIDATOR) if setup else None
    if target is None:
        return None, None
    tasks.progress("Reviewer agent: reading the application as a recruiter")
    try:
        res = llm_call(session, target, kind="recruiter_review", job_id=job.id, prompt_version=RECRUITER_VERSION,
                       profile_hash="-",
                       system=RECRUITER_SYSTEM + __import__("app.services.agent_memory", fromlist=["x"]).instructions_block("review"),
                       user=recruiter_user(job, cv_text, letter, ats.summary(ats_report), letter_warnings, research_text),
                       schema=RecruiterReview, use_cache=False)
    except Exception as exc:  # noqa: BLE001 - the ATS half still gives a verdict
        log.warning("Recruiter review failed (%s) — verdict from the ATS check only", exc)
        return None, None
    data = RecruiterReview.model_validate(res.data).model_dump()
    for k in ("cv_score", "letter_score", "fit_score"):
        data[k] = _clamp10(data.get(k))
    for k in ("strengths", "concerns", "red_flags", "cv_changes", "letter_changes"):
        data[k] = [str(x).strip() for x in (data.get(k) or []) if str(x).strip()][:8]
    data["model"] = res.target.label
    return data, res


def decide(ats_report: dict[str, Any], recruiter: dict[str, Any] | None, letter_validation: dict[str, Any] | None,
           personalization: dict[str, Any] | None = None) -> dict[str, Any]:
    """Combine the ATS check, the recruiter read and the letter's own validation into SEND / REVISE."""
    p = get_prefs().application
    reasons: list[str] = []
    blocking: list[str] = []
    if ats_report["score"] < p.ats_send_threshold:
        reasons.append(f"ATS score {ats_report['score']} is below {p.ats_send_threshold}")
    for c in ats_report["parse_checks"]:
        if not c["ok"] and c["id"] in ("text", "email", "name", "glyphs"):
            blocking.append(f"ATS parse problem: {c['label']} — {c['detail']}")
    warnings = (letter_validation or {}).get("warnings") or []
    for w in warnings:
        if w.get("type") in _BLOCKING_LETTER_WARNINGS:
            blocking.append(f"Cover letter: {w.get('detail')}")
        elif w.get("type") == "tone":  # a negative self-statement is worth a revision round
            reasons.append(f"Cover letter tone: {w.get('detail')}")
    quality = None
    gap_notes: list[str] = []
    gaps = list(ats_report.get("missing_gap") or [])
    if recruiter:
        quality = round((recruiter["cv_score"] + recruiter["letter_score"]) / 2, 1)
        why = recruiter.get("verdict_reason") or "would not forward it yet"
        if recruiter["verdict"] != "SEND":
            # "candidate lacks X" is a fact about you, not about the documents: revising can't fix it honestly
            (gap_notes if is_gap_finding(why, gaps) else reasons).append(f"Recruiter: {why}")
        if quality < p.recruiter_send_threshold:
            reasons.append(f"Recruiter quality {quality}/10 is below {p.recruiter_send_threshold}")
        for flag in recruiter.get("red_flags", [])[:6]:
            (gap_notes if is_gap_finding(flag, gaps) else reasons).append(f"Red flag: {flag}")
    elif personalization and personalization.get("personalization_quality") == "LOW":
        reasons.append("Cover letter reads generic (heuristic check; no reviewer model available)")
    fixable_blocks = list(blocking)
    blocking += [f"Knock-out: {k}" for k in ats_report["knockouts"]]
    verdict = "SEND" if not reasons and not blocking else "REVISE"
    # only knock-outs left (e.g. a required language you don't speak): revising the documents cannot fix them
    fixable = bool(reasons or fixable_blocks)
    if gaps:
        gap_notes.insert(0, "Genuine gaps (not in your profile — never added, at most addressed honestly in the letter): "
                         + ", ".join(gaps))
    return {"verdict": verdict, "reasons": blocking + reasons, "quality": quality, "fixable": fixable, "gap_notes": gap_notes}


def revision_plan(ats_report: dict[str, Any], recruiter: dict[str, Any] | None, keywords: list[Keyword],
                  letter: str, letter_validation: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the application agent changes in the next round (only things the profile supports)."""
    required = {k.term for k in keywords if k.priority == "required"}
    add = sorted(ats_report["missing_have"], key=lambda t: t not in required)
    letter_keywords = [t for t in add if t in required and not term_in(t, letter)][:6]
    gaps = [g.lower() for g in ats_report.get("missing_gap") or []]
    # the CV never gets a skill you don't have; the letter may still bridge a gap honestly
    cv_changes = [c for c in (recruiter or {}).get("cv_changes", []) if not any(g in c.lower() for g in gaps)]
    letter_changes = list((recruiter or {}).get("letter_changes", []))
    letter_changes += [f"Fix: {f}" for f in (recruiter or {}).get("red_flags", []) if "letter" in f.lower() or "anschreiben" in f.lower()]
    # the letter's own tone check: negative self-statements become positive learning steps
    letter_changes += [f"Rephrase positively, as a learning step (no negative self-statement): «{w.get('sentence')}»"
                       for w in (letter_validation or {}).get("warnings", []) if w.get("type") == "tone"]
    return {
        "add_keywords": add[:8],
        "cv_changes": cv_changes[:8],
        "letter_changes": letter_changes[:10],
        "letter_keywords": letter_keywords,
    }


def review_application(session, job, cv: dict[str, Any], pdf: bytes, keywords: list[Keyword], letter_doc,
                       setup: LLMSetup | None, profile: CandidateProfile,
                       research_text: str = "") -> tuple[dict[str, Any], list[CallResult]]:
    tasks.progress("Reviewer agent: ATS check of the CV PDF")
    ats_report = ats.check(cv, pdf, keywords, job, profile)
    letter = letter_doc.content if letter_doc is not None else ""
    validation = letter_doc.validation if letter_doc is not None else None
    warnings = [str(w.get("detail")) for w in ((validation or {}).get("warnings") or [])]
    recruiter, call = recruiter_review(session, job, render_text(cv), letter, ats_report, warnings, setup, research_text)
    decision = decide(ats_report, recruiter, validation, (validation or {}).get("personalization"))
    review = {"ats": ats_report, "recruiter": recruiter, **decision,
              "revision": (revision_plan(ats_report, recruiter, keywords, letter, validation)
                           if decision["verdict"] == "REVISE" else None),
              "letter_document_id": letter_doc.id if letter_doc is not None else None}
    return review, [call] if call else []
