"""Application material: cover-letter pipeline. Runs ONLY on an explicit user action (Generate / Regenerate / editor
button). Status changes never trigger it.

  writer (Qwen3-14B)  →  [German style editor (Mistral Small 3.1), optional]  →  [humanizer, optional]
                      →  factual validation (+ correction, + deterministic removal)
Every stage is stored as its own ApplicationDocument row (draft / edited / humanized / final); nothing is overwritten.
Local GPU work is done under GPU_LOCK so a running batch can't interleave model swaps with this pipeline.
"""

from __future__ import annotations

import contextlib
import logging
import re
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config_files import CandidateProfile, WritingPrefs, get_prefs, get_profile
from app.db import session_scope
from app.llm.application_prompts import LETTER_REVISE_VERSION, letter_revise_system, letter_revise_user
from app.llm.factory import LLMSetup, Role, RoleTarget, get_llm
from app.llm.letter_prompts import (
    AVOID_PHRASES_DE,
    AVOID_PHRASES_EN,
    CORRECTION_SYSTEM,
    CORRECTION_VERSION,
    EDITOR_DE_VERSION,
    FACTCHECK_SYSTEM,
    FACTCHECK_VERSION,
    REVIEW_SYSTEM,
    REVIEW_VERSION,
    WRITER_DE_VERSION,
    WRITER_EN_VERSION,
    correction_user,
    editor_system_de,
    editor_user,
    factcheck_user,
    writer_system_de,
    writer_system_en,
    writer_user,
    choose_variant,
    review_user,
)
from app.llm.local_provider import GPU_LOCK, clean_text_output, interactive
from app.llm.schemas import SUPPORTED_STATUSES, CoverLetterDraft, FactCheckResult, LetterReview
from app.services.evidence import (
    find_level_violations,
    find_personal_circumstances,
    find_unknown_technologies,
    reconcile_claim,
    split_sentences,
)
from app.models import ApplicationDocument, Job, utcnow
from app.services.humanizer import PROMPT_VERSION as HUMANIZER_VERSION
from app.services.humanizer import HumanizerUnavailable, humanize
from app.services import tasks
from app.services.application_notes import NOTE_FIELDS, PersonalNotes, get_notes, notes_block, record_generation, strategy_for
from app.services.letter_quality import assess
from app.services.letter_templates import LetterTemplate, get_template, select_templates, target_words
from app.services.letter_templates import leakage as template_leakage
from app.services.llm_calls import CallResult, llm_call

log = logging.getLogger(__name__)


CLAIM_STATUSES = ("SUPPORTED_PROFESSIONAL", "SUPPORTED_ACADEMIC", "SUPPORTED_PROJECT", "SUPPORTED_GENERAL", "AMBIGUOUS", "UNSUPPORTED")


# A first draft below this share of the minimum length gets one fuller rewrite; anything above stays as written —
# shorter beats padded.
SHORT_DRAFT_FACTOR = 0.6


class LetterError(ValueError):
    pass


# --------------------------------------------------------------------------- language


_DE_WORDS = {"und", "der", "die", "das", "wir", "für", "mit", "bei", "ist", "eine", "ein", "zu", "von", "auf", "du",
             "dein", "deine", "ihre", "unsere", "unser", "oder", "im", "als", "werden", "sind", "dich", "sie", "über", "aufgaben"}
_EN_WORDS = {"the", "and", "we", "you", "with", "for", "our", "are", "is", "of", "to", "your", "will", "on", "as",
             "be", "team", "experience", "what", "who"}
_GERMAN_WORD = re.compile(r"\b(german|deutsch\w*)\b", re.IGNORECASE)
_LEVEL_BEFORE = re.compile(
    r"(fluent|fluency in|excellent|very good|business[- ]fluent|native|proficient|strong|sehr gute|fließende?s?|"
    r"verhandlungssichere?s?|ausgezeichnete)\s*(command of|knowledge of|skills in|written and spoken|spoken and written)?\s*$",
    re.IGNORECASE,
)
_LEVEL_AFTER = re.compile(r"^\W*(\(?\s*(c1|c2|native|fluent)|is (required|mandatory|a must)|required|mandatory|erforderlich|vorausgesetzt)", re.IGNORECASE)
_OPTIONAL_AFTER = re.compile(r"^\W*(is |would be |are )?(a plus|nice[- ]to[- ]have|an advantage|advantageous|beneficial|optional|von vorteil|wünschenswert|ein plus)", re.IGNORECASE)


def german_expected_in(text: str) -> bool:
    """German explicitly required (level stated in the same clause), not merely 'a plus'."""
    for m in _GERMAN_WORD.finditer(text):
        after = text[m.end() : m.end() + 50]
        if _OPTIONAL_AFTER.search(after):
            continue
        # only the clause right before "German" (stop at another language or list/sentence boundary)
        clause = re.split(r"english|englisch|arabic|französisch|french|[;•\n.]", text[max(0, m.start() - 60) : m.start()], flags=re.I)[-1]
        if _LEVEL_BEFORE.search(clause) or _LEVEL_AFTER.search(after):
            return True
    return False


def detect_language(job) -> tuple[str, str]:
    """Default letter language: German ad → de; Germany role where German is clearly expected → de; else en."""
    text = " ".join(x for x in (job.title, job.description, job.requirements, job.responsibilities) if x)[:8000]
    words = re.findall(r"[a-zäöüß]+", text.lower())
    de = sum(w in _DE_WORDS for w in words)
    en = sum(w in _EN_WORDS for w in words)
    if de >= 5 and de > en:
        return "de", "German job advertisement"
    german_expected = german_expected_in(text) or any(
        str(req).lower().startswith("german") and re.search(r"c1|c2|native|fluent|flie|verhandlung|sehr gut", str(req), re.I)
        for req in (job.language_requirements or [])
    )
    if (job.country or "") == "Germany" and german_expected:
        return "de", "Role in Germany where German is clearly expected"
    if en >= 5:
        return "en", "English job advertisement"
    if (job.country or "") == "Germany":
        return "de", "Role in Germany (little text to judge)"
    return "en", "Default"


# --------------------------------------------------------------------------- helpers


def today_line(lang: str) -> str:
    """The model doesn't know the date: without it a degree finished 05/2026 is described in the future tense."""
    today = utcnow()
    if lang == "de":
        return (f"\n\nHeutiges Datum: {today:%d.%m.%Y}. Zeiträume, die vor heute enden, sind abgeschlossen "
                "(Vergangenheitsform).")
    return f"\n\nToday's date: {today:%d %B %Y}. Periods that end before today are completed (past tense)."


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def normalize_letter(text: str) -> str:
    text = clean_text_output(text or "").replace("\r\n", "\n")
    lines = text.split("\n")
    # editors sometimes prepend "Hier ist die überarbeitete Fassung:" — drop such a lead-in line
    if lines and lines[0].rstrip().endswith(":") and len(lines[0]) < 120 and not re.match(r"^(sehr geehrte|dear|hallo|guten tag)", lines[0], re.I):
        lines = lines[1:]
    text = "\n".join(line.rstrip() for line in lines).strip().strip('"„“').strip()
    return _lowercase_after_salutation(re.sub(r"\n{3,}", "\n\n", text))


_CLOSING_LINE = re.compile(
    r"^(mit (den )?(freundlichen|besten) grüßen|freundliche grüße|viele grüße|beste grüße|kind regards|best regards|"
    r"regards|sincerely|yours sincerely|yours faithfully)\s*,?\s*$", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"\[[^\]\n]{1,40}\]")


def ensure_signature(text: str, name: str) -> str:
    """The line after the closing is always the applicant's name (models drop it or write "[Name]")."""
    if not name:
        return text
    lines = text.rstrip().split("\n")
    idx = next((i for i in range(len(lines) - 1, max(-1, len(lines) - 5), -1) if _CLOSING_LINE.match(lines[i].strip())), None)
    if idx is None:
        return text
    lines[idx] = lines[idx].rstrip().rstrip(",")  # "Mit freundlichen Grüßen" / "Kind regards" — no comma
    rest = [ln for ln in lines[idx + 1:] if ln.strip()]
    if rest and not all(_PLACEHOLDER.fullmatch(ln.strip()) or ln.strip().lower() == name.lower() for ln in rest):
        return text  # something else (e.g. a real name typed by hand) — leave it
    return "\n".join(lines[:idx + 1] + [name])


# After "Sehr geehrte …," German continues in lower case ("ich bewerbe mich …"). Only function words are lowered —
# nouns and the formal "Sie/Ihr/Ihnen" keep their capital letter.
_LOWER_AFTER_COMMA = {
    "ich", "mein", "meine", "meinen", "meiner", "meinem", "mich", "mir", "mit", "als", "seit", "nach", "bei", "durch",
    "in", "im", "gerne", "gern", "hiermit", "über", "auf", "vielen", "da", "wie", "während", "nachdem", "aktuell",
    "derzeit", "zurzeit", "bereits", "schon", "die", "der", "das", "den", "dem", "ein", "eine", "einen", "zu", "zum",
    "zur", "für", "an", "am", "von", "vom", "nun", "aufgrund", "dank", "besonders", "gerade", "wenn", "weil",
}
_GERMAN_SALUTATION = re.compile(r"^(sehr geehrte[rs]?|liebe[rs]?|hallo|guten tag)\b[^\n]*,\s*$", re.IGNORECASE)


def _lowercase_after_salutation(text: str) -> str:
    head, sep, rest = text.partition("\n")
    if not sep or not _GERMAN_SALUTATION.match(head):
        return text
    m = re.match(r"(\s*)([A-ZÄÖÜ][a-zäöüß]*)\b", rest)
    if m and m.group(2).lower() in _LOWER_AFTER_COMMA:
        rest = rest[:m.start(2)] + m.group(2).lower() + rest[m.end(2):]
    return head + sep + rest


def numeric_warnings(letter: str, profile: CandidateProfile, job) -> list[dict[str, str]]:
    """Deterministic guard: numbers in the letter must come from the profile (or the vacancy, for company facts)."""
    profile_text = profile.evidence_corpus()
    job_text = " ".join(x for x in (job.title, job.description, job.requirements, job.responsibilities) if x)
    warnings = []
    def present(value: str, text: str) -> bool:  # whole number only: "7" must not match inside "07/2023"
        return re.search(rf"(?<![\d.,]){re.escape(value)}(?![\d]|[.,]\d)", text) is not None

    dates = re.compile(r"(?<![\d.])\d{1,2}\.\d{1,2}\.\d{2,4}(?![\d])|(?<![\d/])\d{1,2}/\d{4}(?![\d])")
    for d in sorted(set(dates.findall(letter))):  # a date is one token: "05.05.2026" must not become "05.05"
        if d not in profile_text and d not in job_text:
            warnings.append({"type": "number", "detail": f"Date “{d}” is not in your profile — check it."})
    letter = dates.sub(" ", letter)
    for n in sorted(set(re.findall(r"(?<![\w.,])\d+(?:[.,]\d+)?(?![\w])", letter))):
        variants = {n, n.replace(",", "."), n.replace(".", ",")}
        if not any(present(v, profile_text) or present(v, job_text) for v in variants):
            warnings.append({"type": "number", "detail": f"Number “{n}” is not in your profile — check it."})
    return warnings


def _sum_usage(results: list[CallResult]) -> dict[str, Any]:
    return {
        "input_tokens": sum(r.record.input_tokens or 0 for r in results),
        "output_tokens": sum(r.record.output_tokens or 0 for r in results),
        "cost_usd": round(sum(r.record.cost_usd or 0 for r in results), 6),
        "duration_ms": sum(r.record.duration_ms or 0 for r in results),
    }


def _writing_for(w: WritingPrefs, templates: list[LetterTemplate]) -> WritingPrefs:
    """With templates, the length target follows the examples (their body length ±15 %)."""
    if not templates:
        return w
    lo, hi = target_words(templates, (w.target_words_min, w.target_words_max))
    return w.model_copy(update={"target_words_min": lo, "target_words_max": hi})


def fit_from_analysis(job: Job, max_strong: int = 6, max_transferable: int = 5) -> dict[str, Any] | None:
    """Strong + transferable/partial matches and critical gaps from the job's current fit analysis (None if absent)."""
    a = job.current_analysis.result if job.current_analysis and job.current_analysis.result else None
    if not a or not (a.get("strong_matches") or a.get("partial_matches")):
        return None

    def items(key: str, limit: int) -> list[dict[str, str]]:
        return [{"skill": str(m.get("skill", ""))[:120], "evidence": str(m.get("evidence", "")),
                 "profile_evidence": str(m.get("profile_evidence", ""))[:240]}
                for m in (a.get(key) or [])[:limit] if m.get("skill")]

    return {"strong": items("strong_matches", max_strong), "transferable": items("partial_matches", max_transferable),
            "gaps": [str(g)[:160] for g in (a.get("critical_gaps") or [])[:6]]}


# Requirements come from the (English) analysis, letters are often German.
_EN_DE = {
    "ai": ("ki",), "development": ("entwicklung", "entwickl"), "developer": ("entwickler",), "software": ("software",),
    "infrastructure": ("infrastruktur",), "automation": ("automatisier",), "automated": ("automatisier",),
    "testing": ("test",), "tests": ("test",), "test": ("test",), "machine": ("maschinell",), "learning": ("lernen",),
    "workflows": ("workflow", "abläuf", "prozess"), "workflow": ("workflow", "ablauf", "prozess"),
    "pipelines": ("pipeline",), "integration": ("integration", "integrier"), "data": ("daten",),
    "visualization": ("visualisier",), "security": ("sicherheit",), "requirements": ("anforderung",),
    "engineering": ("entwicklung", "engineering"), "app": ("app", "anwendung"), "applications": ("anwendung",),
    "systems": ("system",), "embedded": ("eingebettet", "embedded"), "analysis": ("analyse",),
    "communication": ("kommunikation",), "teamwork": ("team",), "english": ("englisch",), "german": ("deutsch",),
}

_COVER_STOP = {"and", "with", "for", "the", "und", "mit", "der", "die", "das", "von", "experience", "erfahrung",
               "knowledge", "kenntnisse", "skills", "basic", "gute", "sehr", "strong", "solid", "using", "based"}


def skill_coverage(letter: str, fit: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Heuristic: a match counts as visible when the requirement (or most of its key words) appears in the letter."""
    text = letter.lower()
    letter_words = set(re.findall(r"[a-zäöüß0-9+#./-]{2,}", text.replace("-", " ")))

    def present(word: str) -> bool:
        options = {word, *_EN_DE.get(word, ())}
        return any(o in text if len(o) < 5 else any(lw.startswith(o[:5]) for lw in letter_words) or o in text
                   for o in options)

    def visible(skill: str) -> bool:
        s = skill.lower().strip()
        if s and s in text:
            return True
        words = [w for w in re.findall(r"[a-zäöüß0-9+#./-]{2,}", s) if w not in _COVER_STOP]
        return bool(words) and sum(present(w) for w in words) / len(words) >= 0.5

    return {kind: [{"skill": m["skill"], "evidence": m.get("evidence", ""), "covered": visible(m["skill"])}
                   for m in fit.get(kind, [])]
            for kind in ("strong", "transferable")}


def _group_notes(session: Session, group_id: str) -> PersonalNotes | None:
    """The notes snapshot the draft of this letter group was written with (None for pre-notes letters)."""
    draft = session.scalars(select(ApplicationDocument).where(ApplicationDocument.group_id == group_id,
                                                              ApplicationDocument.stage == "draft")).first()
    snapshot = (draft.meta or {}).get("personal_notes") if draft else None
    return PersonalNotes(**{k: snapshot.get(k, "") or "" for k in NOTE_FIELDS}) if snapshot is not None else None


def _review(session: Session, target: RoleTarget, job: Job, letter: str,
            notes: PersonalNotes | None) -> tuple[dict[str, Any] | None, CallResult | None]:
    """Content-specificity review ("could this be sent elsewhere?") on the already-loaded validator model.
    Advisory only: if it fails, the deterministic quality heuristics stand in."""
    tasks.progress(f"Reviewing personalization ({_short(target)})")
    try:
        res = llm_call(session, target, kind="letter_review", job_id=job.id, prompt_version=REVIEW_VERSION,
                       profile_hash="-", system=REVIEW_SYSTEM, user=review_user(job, letter, notes_block(notes)),
                       schema=LetterReview, use_cache=False)
        return LetterReview.model_validate(res.data).model_dump(), res
    except Exception as exc:  # noqa: BLE001 - advisory step must never sink a validated letter
        log.warning("Letter review failed (%s) — using heuristics", exc)
        return None, None


def _group_fit(session: Session, group_id: str) -> dict[str, Any] | None:
    draft = session.scalars(select(ApplicationDocument).where(ApplicationDocument.group_id == group_id,
                                                              ApplicationDocument.stage == "draft")).first()
    return (draft.meta or {}).get("fit") if draft else None


def _group_templates(session: Session, group_id: str) -> list[LetterTemplate]:
    """Templates the draft of this letter group was written from (still on disk)."""
    draft = session.scalars(select(ApplicationDocument).where(ApplicationDocument.group_id == group_id,
                                                              ApplicationDocument.stage == "draft")).first()
    refs = ((draft.meta or {}).get("templates") or []) if draft else []
    return [t for r in refs if (t := get_template(r.get("id", "")))]


_CITIZENSHIP_RX = re.compile(
    r"staatsbürger|staatsangehörig|arbeitserlaubnis|aufenthaltstitel|arbeitsgenehmigung|citizen|work permit|"
    r"work authori[sz]ation|right to work|visa", re.IGNORECASE)
_RELOCATION_RX = re.compile(
    r"umzug|umzieh|umzuziehen|deutschlandweit|ortsflexibel|ortsungebunden|arbeitsort\w* flexibel|flexibel\w* (beim|im|was den|bezüglich des) (arbeits|einsatz)?ort|"
    r"relocat|anywhere in germany|flexible about the location|move to|umzug nach|nach luxemburg", re.IGNORECASE)
_LETTER_CLOSING = re.compile(r"^(mit (den )?(freundlichen|besten) grüßen|freundliche grüße|beste grüße|viele grüße|"
                             r"kind regards|best regards|regards|sincerely|yours sincerely)\b", re.IGNORECASE)
_CONVERSATION = re.compile(r"gespräch|kennenlernen|rückmeldung|interview|conversation|hearing from you|talk", re.IGNORECASE)


def mobility_applies(job, w: WritingPrefs) -> bool:
    """The work-permit/relocation statement belongs in letters for jobs in the configured countries."""
    countries = {c.lower() for c in w.mobility_countries}
    return bool(countries) and (job.country or "").lower() in countries


_MONTHS_DE = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober",
              "November", "Dezember"]
_MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
_DEGREE_RX = re.compile(r"(master|m\.\s?sc|bachelor|b\.\s?sc|studium|studies|degree|abschluss)", re.IGNORECASE)
_DONE_RX = re.compile(r"(abgeschlossen|abschließe|abschliessen|beendet|completed|graduated|finish|schließe .* ab)",
                      re.IGNORECASE)
_CLOSING_PARA = re.compile(r"^(mit (den )?(freundlichen|besten) grüßen|freundliche grüße|beste grüße|viele grüße|"
                           r"kind regards|best regards|regards|sincerely|yours sincerely|yours faithfully)\b", re.IGNORECASE)
_REPLY_VARIANTS = {
    "de": ["Über die Gelegenheit, mich Ihnen in einem persönlichen Gespräch vorzustellen, freue ich mich sehr.",
           "Ich freue mich darauf, Sie in einem persönlichen Gespräch von mir zu überzeugen.",
           "Über eine Einladung zu einem Vorstellungsgespräch freue ich mich sehr."],
    "en": ["I would be delighted to introduce myself in a personal interview.",
           "I look forward to convincing you in a personal conversation.",
           "I would very much welcome an invitation to an interview."],
}


def degree_sentence(language: str, today=None) -> str:
    """Point 4 of the letter outline, from the master CV (config/cv.yaml): time and type of the highest degree."""
    from app.services.cv import CvError, get_master_cv, tr

    try:
        edu = get_master_cv().education
    except CvError:
        return ""
    if not edu:
        return ""
    e = edu[0]
    try:
        month, year = (int(x) for x in e.end.split("/"))
    except ValueError:
        return ""
    title = tr(e.title, language).replace("M.Sc. ", "Master of Science in ").replace("B.Sc. ", "Bachelor of Science in ")
    done = (today or utcnow()).date() >= __import__("datetime").date(year, month, 1)
    if language == "de":
        when = f"{_MONTHS_DE[month - 1]} {year}"
        # the institution in parentheses: no German case ending to get wrong ("an der Technischen Hochschule")
        return (f"Meinen {title} ({e.org}) habe ich im {when} abgeschlossen." if done
                else f"Meinen {title} ({e.org}) schließe ich voraussichtlich im {when} ab.")
    when = f"{_MONTHS_EN[month - 1]} {year}"
    return (f"In {when} I completed my {title} at {e.org}." if done
            else f"I expect to complete my {title} at {e.org} in {when}.")


# Finishing the degree described as still ahead: "Ich schließe im Mai 2026 meinen M.Sc. … ab", "werde … abschließen",
# "I will complete / expect to complete / am completing my M.Sc."
_FUTURE_DEGREE = re.compile(
    r"\b(schlie(ß|ss)e\b.*\bab\b|werde\b.*\babschlie(ß|ss)en|voraussichtlich\b.*\babschl|bald\b.*\babschl|"
    r"will (complete|finish|graduate)|expect to (complete|finish|graduate)|am (completing|finishing)|"
    r"(complete|finish) my .* in (january|february|march|april|may|june|july|august|september|october|november|december) \d{4})",
    re.IGNORECASE)


def degree_done(today=None) -> bool:
    from app.services.cv import CvError, get_master_cv

    try:
        e = get_master_cv().education[0]
        month, year = (int(x) for x in e.end.split("/"))
    except (CvError, IndexError, ValueError, AttributeError):
        return False
    return (today or utcnow()).date() >= __import__("datetime").date(year, month, 1)


def fix_degree_tense(text: str, language: str, today=None) -> tuple[str, bool]:
    """Once the degree's end date has passed, a sentence that still talks about finishing it (a model's tense slip, or
    a draft written before the date) is replaced by the correct past-tense degree sentence from cv.yaml."""
    if not degree_done(today):
        return text, False
    sentence = degree_sentence(language, today)
    if not sentence:
        return text, False
    changed = False
    paragraphs = []
    for para in text.split("\n\n"):
        # split only before a capital letter: "M.Sc. in Intelligent Systems" stays one sentence
        parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ„\"])", para.strip()) if "\n" not in para.strip() else [para]
        out = []
        for part in parts:
            if _DEGREE_RX.search(part) and _FUTURE_DEGREE.search(part):
                if sentence not in out:
                    out.append(sentence)
                changed = True
            else:
                out.append(part)
        paragraphs.append(" ".join(out) if parts != [para] else out[0])
    return "\n\n".join(paragraphs), changed


_NAMED_SALUTATION = re.compile(r"^(sehr geehrte[rn]?|liebe[rn]?|dear|hello)\s+(frau|herr|herrn|ms\.?|mrs\.?|mr\.?|dr\.?)"
                                r"\s+(?:(?:dr|prof)\.?\s+)*([^,\n]+?)\s*,\s*$", re.IGNORECASE)


def ensure_salutation(text: str, language: str, job_text: str) -> tuple[str, str | None]:
    """A salutation that names a person must name someone the posting names — otherwise (an invented name or a
    template placeholder like "[Nachname]") it becomes the neutral salutation. Returns (text, replaced line)."""
    first, _, rest = text.strip().partition("\n")
    neutral = "Sehr geehrte Damen und Herren," if language == "de" else "Dear Hiring Team,"
    if re.match(r"^sehr geehrte damen und \w+,?\s*$", first.strip(), re.I) and first.strip() != neutral:
        return f"{neutral}\n{rest}" if rest else neutral, first.strip()  # "... Damen und Herlen" (model typo)
    m = _NAMED_SALUTATION.match(first.strip())
    if not m and not _PLACEHOLDER.search(first):
        return text, None
    surname = (m.group(3).split()[-1] if m else "").strip("[]()")
    if m and surname and not _PLACEHOLDER.search(first) and surname.lower() in (job_text or "").lower():
        return text, None
    return f"{neutral}\n{rest}" if rest else neutral, first.strip()


def gap_claims(text: str, gaps: list[str]) -> list[dict[str, Any]]:
    """First-person sentences that claim experience with a gap the fit analysis found (e.g. "Erfahrung mit
    Omniverse") — unless they frame it as learning or say it's missing. Deterministic backstop to the fact check."""
    from app.services.evidence import _DISCLAIMED, _FIRST_PERSON, _LEARNING
    from app.services.keywords import literal_in

    terms = [g for g in gaps if g and len(g) >= 3]
    out = []
    for sentence in split_sentences(text):
        if not _FIRST_PERSON.search(sentence) or _LEARNING.search(sentence) or _DISCLAIMED.search(sentence):
            continue
        hits = [t for t in terms if literal_in(t, sentence)]
        if hits:
            out.append({"sentence": sentence, "claim": f"experience with {', '.join(hits)}", "claimed_level": "professional",
                        "status": "UNSUPPORTED", "profile_evidence": "",
                        "note": f"{', '.join(hits)} is a gap the fit analysis found — address it as a learning step instead.",
                        "source": "gap-guard"})
    return out


def job_gap_terms(job) -> list[str]:
    """Core terms of the fit analysis' critical gaps ("3 years of NVIDIA Omniverse" → "NVIDIA Omniverse")."""
    from app.services.keywords import _core_terms

    a = job.current_analysis.result if getattr(job, "current_analysis", None) and job.current_analysis.result else {}
    terms: list[str] = []
    for gap in a.get("critical_gaps") or []:
        terms += _core_terms(str(gap), [])
    return [t for t in dict.fromkeys(terms) if len(t.split()) <= 3]


def strip_header(text: str) -> tuple[str, bool]:
    """Sender block, recipient address, place/date and subject lines above the salutation are removed — the letter
    layout adds them from your CV and the posting (a model-written header has invented contact data, template
    placeholders and wrong dates). Also when the header follows a stray salutation line."""
    from app.services.letter_templates import salutation_line

    lines = text.strip().split("\n")
    sal = salutation_line(lines, lambda ln: bool(_GERMAN_SALUTATION.match(ln)
                                                or re.match(r"^(dear|hello|hi)\b.*,\s*$", ln, re.IGNORECASE)))
    if sal:
        return "\n".join(lines[sal:]), True
    return text, False


_ONLY_PLACEHOLDERS = re.compile(r"^[\s,.;:·|/–-]*(\[[^\]\n]{1,40}\][\s,.;:·|/–-]*)+$")
# Phone numbers and e-mail addresses in the letter text: the layout prints your real ones; a model-written one is
# invented ("+49 152 0000000", "name@example.com").
_CONTACT_DATA = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\+\d{2}[\d \t/()-]{6,}\d|\b0\d{2,4}[ \t/-]?\d{5,}\b")


def drop_placeholders(text: str) -> tuple[str, list[str]]:
    """Template placeholders ("[Ansprechpartner/in]", "[Datum]", "[Firma]") never reach a finished letter: lines that
    are only placeholders go, a sentence with a placeholder goes (its content is unknown). The salutation line is
    left to ensure_salutation, which replaces a placeholder name with the neutral salutation."""
    removed: list[str] = []
    paragraphs = []
    for para in text.strip().split("\n\n"):
        lines = []
        for ln in para.split("\n"):
            header_like = len(ln.strip()) <= 110 and not ln.strip().endswith((".", "!", "?"))  # "+49 … · a@b.de"
            if _ONLY_PLACEHOLDERS.match(ln) and ln.strip() or _CONTACT_DATA.search(ln) and header_like:
                removed.append(ln.strip())
                continue
            lines.append(ln)
        para = "\n".join(lines).strip()
        if para and (_PLACEHOLDER.search(para) or _CONTACT_DATA.search(para)) and not _GERMAN_SALUTATION.match(para) \
                and not _NAMED_SALUTATION.match(para):
            kept = []
            for sentence in split_sentences(para):
                if _PLACEHOLDER.search(sentence) or _CONTACT_DATA.search(sentence):
                    removed.append(sentence)
                else:
                    kept.append(sentence)
            para = " ".join(kept)
        if para:
            paragraphs.append(para)
    return tidy_after_removal("\n\n".join(paragraphs)), removed


def ensure_outline(text: str, language: str, seed: str = "") -> tuple[str, list[str]]:
    """The applicant's letter outline, enforced on every finished letter whatever stage produced it:
    salutation → motivation/company → why you fit → degree → (mobility, added later) → interview request → closing.
    Adds a missing salutation, degree sentence, interview request or closing; never rewrites the model's text."""
    de = language == "de"
    added: list[str] = []
    text, header = strip_header(text)
    if header:
        added.append("removed header")
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    if not paragraphs or not _GERMAN_SALUTATION.match(paragraphs[0].split("\n")[0]) and not re.match(
            r"^(dear|hello|hi)\b.*,\s*$", paragraphs[0].split("\n")[0], re.IGNORECASE):
        paragraphs.insert(0, "Sehr geehrte Damen und Herren," if de else "Dear Hiring Team,")
        added.append("salutation")
    closing_at = next((i for i, p in enumerate(paragraphs) if _CLOSING_PARA.match(p)), None)
    if closing_at is None:
        paragraphs.append(("Mit freundlichen Grüßen" if de else "Kind regards") + "\n[Name]")
        closing_at = len(paragraphs) - 1
        added.append("closing")
    body = paragraphs[1:closing_at]
    reply_at = next((i for i in range(len(body) - 1, max(-1, len(body) - 3), -1) if _CONVERSATION.search(body[i])), None)
    if reply_at is None:
        variants = _REPLY_VARIANTS["de" if de else "en"]
        body.append(variants[sum(map(ord, seed)) % len(variants)])
        reply_at = len(body) - 1
        added.append("interview request")
    joined, tense_fixed = fix_degree_tense("\n\n".join(body), language)
    if tense_fixed:
        body = [b for b in joined.split("\n\n") if b.strip()]
        added.append("degree tense")
    if not any(_DEGREE_RX.search(p) and _DONE_RX.search(p) for p in body):
        sentence = degree_sentence(language)
        if sentence:  # before the citizenship / relocation paragraph if there is one, else before the request
            mobility_at = next((i for i, p in enumerate(body) if _CITIZENSHIP_RX.search(p) and len(p.split()) <= 40), reply_at)
            body.insert(min(mobility_at, reply_at), sentence)
            added.append("degree")
    return "\n\n".join([paragraphs[0], *body, *paragraphs[closing_at:]]), added


def ensure_mobility_statement(text: str, language: str, job, w: WritingPrefs) -> tuple[str, list[str]]:
    """Insert the configured sentences for whatever the letter left out (citizenship / relocation).
    Placed right before the closing "conversation" paragraph, else before the sign-off."""
    if not mobility_applies(job, w):
        return text, []
    de = language == "de"
    own = (w.country_sentences or {}).get(job.country or "", {})  # e.g. Luxembourg: EU citizen, move to Luxembourg
    missing = []
    if not _CITIZENSHIP_RX.search(text):
        missing.append(own.get("work_permit_de" if de else "work_permit_en")
                       or (w.work_permit_sentence_de if de else w.work_permit_sentence_en))
    if not _RELOCATION_RX.search(text):
        missing.append(own.get("relocation_de" if de else "relocation_en")
                       or (w.relocation_sentence_de if de else w.relocation_sentence_en))
    missing = [m.strip() for m in missing if m and m.strip()]
    if not missing:
        return text, []
    sentence = " ".join(missing)
    paragraphs = text.split("\n\n")
    closing = next((i for i, p in enumerate(paragraphs) if _LETTER_CLOSING.match(p.strip())), len(paragraphs))
    insert_at = closing
    if closing >= 2 and _CONVERSATION.search(paragraphs[closing - 1]) and len(paragraphs[closing - 1].split()) <= 40:
        insert_at = closing - 1
    paragraphs.insert(insert_at, sentence)
    return "\n\n".join(paragraphs), missing


def _short(target: RoleTarget | None) -> str:
    return target.model.split("/")[-1].split(":")[0] if target else "?"


@contextlib.contextmanager
def _gpu_guard(*targets: RoleTarget | None):
    if not any(t is not None and t.is_local for t in targets):
        yield
        return
    with interactive():  # you are waiting for this: background batches pause
        with _gpu_lock_with_progress():
            yield


@contextlib.contextmanager
def _gpu_lock_with_progress():
    if not GPU_LOCK.acquire(blocking=False):  # e.g. an analysis batch is using the GPU → say so instead of hanging silently
        tasks.progress("Waiting for the GPU — another model job (e.g. job analysis) is running")
        GPU_LOCK.acquire()
    try:
        yield
    finally:
        GPU_LOCK.release()


# --------------------------------------------------------------------------- validation


def _factcheck(session: Session, target: RoleTarget, job: Job, letter: str, profile: CandidateProfile,
               only: list[str] | None = None) -> CallResult:
    return llm_call(
        session, target, kind="factcheck", job_id=job.id, prompt_version=FACTCHECK_VERSION,
        profile_hash=profile.fingerprint(), system=FACTCHECK_SYSTEM, user=factcheck_user(profile, job, letter, only),
        schema=FactCheckResult,
    )


def sentence_key(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence or "").strip().lower()


def _owner(claim_sentence: str, sentences: list[str]) -> str | None:
    """The letter sentence a claim belongs to (the checker copies it verbatim, but not always exactly)."""
    key = sentence_key(claim_sentence)
    keys = [sentence_key(s) for s in sentences]
    if key in keys:
        return key
    return next((k for k in keys if key and (key in k or k in key)), keys[0] if keys else None)


def factcheck_incremental(session: Session, target: RoleTarget, job: Job, text: str, profile: CandidateProfile,
                          known: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], CallResult | None]:
    """Fact-check only the sentences not verified before (`known`: sentence key -> its claims, updated in place).
    A revision usually changes 2-4 sentences: the model reads the whole letter for context but lists claims only for
    those, which cuts the most expensive call of the pipeline (output tokens) to a fraction."""
    sentences = split_sentences(text)
    new = [s for s in sentences if sentence_key(s) not in known]
    call = None
    if new:
        call = _factcheck(session, target, job, text, profile, only=new if known else None)
        for s in new:
            known[sentence_key(s)] = []
        for c in call.data.get("claims", []):
            owner = _owner(c.get("sentence", ""), new)
            if owner is not None:
                known.setdefault(owner, []).append(reconcile_claim(c))
    return [c for s in sentences for c in known.get(sentence_key(s), [])], call


def _checked_claims(claims: list[dict[str, Any]], text: str, profile: CandidateProfile,
                    gaps: list[str] | None = None) -> list[dict[str, Any]]:
    """LLM claims (already reconciled with the claimed evidence level) plus deterministic guards over the full text."""
    claims = list(claims)
    flagged = {c.get("sentence", "").strip() for c in claims if c.get("status") == "UNSUPPORTED"}
    for v in find_level_violations(text, profile):
        if v.sentence not in flagged:
            claims.append(v.as_claim())
            flagged.add(v.sentence)
    for c in find_unknown_technologies(text, profile) + find_personal_circumstances(text, profile) + gap_claims(text, gaps or []):
        if c["sentence"] not in flagged:
            claims.append(c)
            flagged.add(c["sentence"])
    return claims


def _remove_sentence(text: str, sentence: str) -> str | None:
    """Remove `sentence` from `text` (exact match, then whitespace-insensitive). None if not found."""
    if sentence and sentence in text:
        return text.replace(sentence, "")
    pattern = r"\s+".join(re.escape(w) for w in sentence.split())
    if pattern and re.search(pattern, text):
        return re.sub(pattern, "", text, count=1)
    return None


_NEGATIVE_SELF = re.compile(
    r"\bleider\b|\b(noch )?keine (praktische |direkte |berufliche |eigene )?erfahrung|\bnoch (nicht|nie)\b|\bfehlt mir\b|"
    r"\bmir fehl(t|en)\b|\bnur (wenig|begrenzt|geringe)\w*\b|\bunfortunately\b|\bi (have|had) (no|never|not)\b|"
    r"\bi haven'?t\b|\bi (do not|don'?t) have\b|\bi lack\b|\black(ing)? (of )?(direct |hands-on )?experience|"
    r"\bonly (limited|little|basic) (experience|knowledge)|\bnot (yet )?(familiar|experienced)\b", re.I)
_SELF = re.compile(r"\b(ich|mir|mich|mein\w*|i|my|me)\b", re.I)


def negative_self_talk(text: str) -> list[str]:
    """First-person sentences that talk the applicant down ("Leider habe ich keine Erfahrung mit …") — the letter's
    tone must stay positive; a gap is phrased as a learning step instead."""
    return [s for s in split_sentences(text) if _NEGATIVE_SELF.search(s) and _SELF.search(s)]


def tidy_after_removal(text: str) -> str:
    """A removed clause can leave ', and I am eager …' or '. , so …': drop the dangling joiner, capitalise again."""
    def fix(m: re.Match[str]) -> str:
        return m.group(1) + m.group(2).upper()

    text = re.sub(r"(^|\n\n|[.!?]\s+)[,;:–—-]+\s*(?:(?:and|but|und|aber)\s+)?(\w)", fix, text)
    return re.sub(r"[ \t]+([,.;:])", r"\1", text)


def validate_letter(session: Session, target: RoleTarget, job: Job, letter: str, profile: CandidateProfile,
                    rewrite: bool = True, target_words: tuple[int, int] | None = None,
                    known: dict[str, list[dict[str, Any]]] | None = None) -> tuple[str, dict[str, Any], list[CallResult]]:
    """Fact-check (LLM + evidence-level guard) → correct UNSUPPORTED/AMBIGUOUS claims → re-check →
    remove any UNSUPPORTED sentence that is still left. Returns (final_text, validation_report, call_results).
    rewrite=False skips the LLM correction (used for humanized text, which an LLM rewrite would make machine-like
    again): UNSUPPORTED sentences are then only removed, AMBIGUOUS ones only flagged.
    `known`: sentence verdicts of an earlier version of this letter — only new or changed sentences are checked."""
    calls: list[CallResult] = []
    known = {k: list(v) for k, v in (known or {}).items()}
    tasks.progress(f"Checking facts against your profile ({_short(target)})")
    llm_claims, first = factcheck_incremental(session, target, job, letter, profile, known)
    if first is not None:
        calls.append(first)
    gaps = job_gap_terms(job)
    claims = _checked_claims(llm_claims, letter, profile, gaps)
    rounds = [{"stage": "initial", "claims": claims}]
    text = letter
    corrected = False
    removed: list[str] = []

    issues = [c for c in claims if c.get("status") in ("UNSUPPORTED", "AMBIGUOUS")]
    if issues and rewrite:
        tasks.progress(f"Correcting {len(issues)} unsupported/ambiguous claim(s) ({_short(target)})")
        fix = llm_call(
            session, target, kind="letter_fix", job_id=job.id, prompt_version=CORRECTION_VERSION,
            profile_hash=profile.fingerprint(), system=CORRECTION_SYSTEM, user=correction_user(profile, text, issues),
            use_cache=False,
        )
        calls.append(fix)
        text = normalize_letter(fix.data)
        corrected = True
        tasks.progress(f"Re-checking the corrected sentences ({_short(target)})")
        for c in issues:  # a corrected sentence must be checked again even if the model left it unchanged
            known.pop(sentence_key(c.get("sentence", "")), None)
        llm_claims, second = factcheck_incremental(session, target, job, text, profile, known)
        if second is not None:
            calls.append(second)
        claims = _checked_claims(llm_claims, text, profile, gaps)
        rounds.append({"stage": "after correction", "claims": claims})

    # Anything still UNSUPPORTED is removed deterministically (sentence-level), never shown as final text.
    unresolved = []
    for c in claims:
        if c.get("status") != "UNSUPPORTED":
            continue
        sentence = (c.get("sentence") or "").strip()
        new_text = _remove_sentence(text, sentence)
        if new_text is not None:
            text = new_text
            removed.append(sentence)
        else:
            unresolved.append(c)
    if removed:
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r" +\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)  # a removed sentence must not leave an indented paragraph
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        text = tidy_after_removal(text)

    warnings: list[dict[str, str]] = []
    for c in claims:
        if c.get("status") == "AMBIGUOUS":
            warnings.append({"type": "ambiguous", "detail": f"{c.get('claim')}: {c.get('note') or 'check wording'}",
                             "sentence": c.get("sentence", "")})
    for c in unresolved:
        warnings.append({"type": "unsupported", "detail": f"Unsupported claim could not be removed automatically: {c.get('claim')}",
                         "sentence": c.get("sentence", "")})
    for c in claims:
        if c.get("status") == "UNSUPPORTED" and c.get("sentence", "").strip() in removed:
            warnings.append({"type": "removed", "detail": f"Removed: {c.get('note') or c.get('claim')}", "sentence": c["sentence"]})
    warnings += numeric_warnings(text, profile, job)
    warnings += [{"type": "tone", "detail": f"Negative about yourself — phrase it positively: „{s}“", "sentence": s}
                 for s in negative_self_talk(text)]
    for ph in sorted(set(_PLACEHOLDER.findall(text))):  # e.g. "[Firma]" copied from a template
        warnings.append({"type": "placeholder", "detail": f"Placeholder {ph} left in the letter — fill it in or remove it"})
    lowered = text.lower()
    for phrase in AVOID_PHRASES_DE + AVOID_PHRASES_EN:  # models don't always obey the style rules → make it visible
        if phrase.lower() in lowered:
            warnings.append({"type": "style", "detail": f"Stock phrase: “{phrase}” — consider rewording"})
    w = get_prefs().writing
    lo, hi = target_words or (w.target_words_min, w.target_words_max)
    wc = word_count(text)
    if wc < lo * SHORT_DRAFT_FACTOR or wc > hi * 1.3:
        warnings.append({"type": "length", "detail": f"{wc} words (target {lo}–{hi})"})

    status = "needs_review" if unresolved or any(x["type"] in ("ambiguous", "number", "placeholder") for x in warnings) else (
        "corrected" if corrected or removed else "clean")
    counts = {s: sum(1 for c in claims if c.get("status") == s) for s in CLAIM_STATUSES}
    counts["SUPPORTED"] = sum(v for k, v in counts.items() if k in SUPPORTED_STATUSES)
    report = {
        "status": status,
        "claims": claims,
        "rounds": rounds,
        "corrected": corrected,
        "removed_sentences": removed,
        "warnings": warnings,
        "validator": f"{target.provider.name}:{target.model}",
        "counts": counts,
        # verdicts per sentence of the final text: the next version (revision) only checks what changed
        "checked_sentences": {sentence_key(x): known.get(sentence_key(x), []) for x in split_sentences(text)
                              if sentence_key(x) in known},
    }
    return text, report, calls


# --------------------------------------------------------------------------- pipeline stages


def _store(session: Session, *, job: Job, group_id: str, stage: str, language: str, content: str, subject: str | None,
           source: ApplicationDocument | None, target: RoleTarget, prompt_version: str, calls: list[CallResult],
           meta: dict[str, Any], validation: dict[str, Any] | None = None) -> ApplicationDocument:
    answered_by = calls[0] if calls else None
    content = ensure_signature(content, get_prefs().writing.applicant_name.strip())
    doc = ApplicationDocument(
        job_id=job.id, group_id=group_id, stage=stage, language=language, subject=subject, content=content,
        source_document_id=source.id if source else None,
        provider=answered_by.target.provider.name if answered_by else target.provider.name,
        model=answered_by.record.model if answered_by else target.model,
        prompt_version=prompt_version, meta={**meta, "word_count": word_count(content)}, validation=validation,
        **_sum_usage(calls),
    )
    session.add(doc)
    session.flush()
    return doc


def _previous_verdicts(session: Session, source: ApplicationDocument) -> dict[str, list[dict[str, Any]]]:
    """Sentence verdicts of the finished version this one was derived from (a revision of a final letter)."""
    doc, seen = source, set()
    while doc is not None and doc.source_document_id and doc.id not in seen:
        seen.add(doc.id)
        doc = session.get(ApplicationDocument, doc.source_document_id)
        if doc is not None and doc.stage == "final" and (doc.validation or {}).get("checked_sentences"):
            return doc.validation["checked_sentences"]
    return {}


def _finalize(session: Session, setup: LLMSetup, job: Job, source: ApplicationDocument, last: RoleTarget,
              personalization_review: bool = True) -> ApplicationDocument:
    """Validate the latest stage into a 'final' row. personalization_review=False skips the LLM specificity review
    (the application agent's recruiter review judges that, with more context)."""
    profile = get_profile()
    validator = setup.retarget(last, Role.VALIDATOR)  # same model as the last stage → no extra model swap
    templates = _group_templates(session, source.group_id)
    w = _writing_for(get_prefs().writing, templates)
    content, header_removed = strip_header(source.content)  # before the checks: a header date is no claim
    content, placeholders_removed = drop_placeholders(content)
    text, report, calls = validate_letter(session, validator, job, content, profile,
                                          rewrite=source.stage != "humanized",
                                          target_words=(w.target_words_min, w.target_words_max),
                                          known=_previous_verdicts(session, source))
    fit = _group_fit(session, source.group_id)
    if fit:  # which strong / transferable skills the final text actually makes visible
        report["coverage"] = skill_coverage(text, fit)
    if templates:  # facts/phrases that came from a template instead of the profile
        known = " ".join(str(x or "") for x in (profile.as_prompt_text(), job.title, job.company, job.location,
                                                job.description, job.requirements, job.responsibilities,
                                                get_prefs().writing.applicant_name))
        leaks = template_leakage(text, templates, known)
        report["warnings"] += leaks
        if any("eight-word" not in x["detail"] for x in leaks) and report["status"] == "clean":
            report["status"] = "needs_review"
    if source.stage == "humanized":  # surface the rewrite's own drift checks next to the validation warnings
        report["warnings"] += [{"type": "humanizer", "detail": w} for w in (source.meta or {}).get("humanizer_warnings", [])]
    # Guarantee: the work-permit + relocation sentences from preferences.yaml → writing, for jobs in mobility_countries.
    # Guarantee: the applicant's outline (degree sentence, interview request, closing in their places).
    job_text = " ".join(x or "" for x in (job.title, job.company, job.description, job.requirements, job.responsibilities))
    text, replaced = ensure_salutation(text, source.language, job_text)
    if replaced:
        report["warnings"].append({"type": "salutation", "detail": f"Replaced „{replaced}“: the posting names no such "
                                                                 "contact person — neutral salutation used"})
    text, outline_added = ensure_outline(text, source.language, source.group_id)
    outline_added = (["removed header"] if header_removed else []) + outline_added
    if outline_added:
        report["outline_added"] = outline_added
    text, added = ensure_mobility_statement(text, source.language, job, get_prefs().writing)
    if added:
        report["mobility_added"] = added
    text, late = drop_placeholders(text)  # a correction round can bring one back
    placeholders_removed += late
    if placeholders_removed:
        report["placeholders_removed"] = placeholders_removed
        report["warnings"] = [x for x in report["warnings"]
                              if x.get("type") != "placeholder" or any(ph in text for ph in _PLACEHOLDER.findall(x.get("detail", "")))]
    notes = _group_notes(session, source.group_id)
    review, review_call = _review(session, validator, job, text, notes) if personalization_review else (None, None)
    if review_call is not None:
        calls.append(review_call)
    report["personalization"] = assess(text, job, profile.as_prompt_text(), notes is not None and not notes.is_empty,
                                       report, review, has_motivation=bool(notes and notes.has_motivation))
    versions = FACTCHECK_VERSION + (f"+{CORRECTION_VERSION}" if report["corrected"] else "")
    return _store(
        session, job=job, group_id=source.group_id, stage="final", language=source.language, content=text,
        subject=source.subject, source=source, target=validator, prompt_version=versions, calls=calls,
        meta={"validated_stage": source.stage}, validation=report,
    )


def generate_letter(job_id: int, language: str | None = None, use_editor: bool | None = None,
                    setup: LLMSetup | None = None, use_humanizer: bool | None = None,
                    template: str | None = None, research: str = "", personalization_review: bool = True) -> str:
    """Draft → (German editor) → (humanizer) → validated final. Returns the new group id. Never runs implicitly.
    template: None/'auto' = best-matching human-written templates, 'none' = no templates, else a template id."""
    setup = setup or get_llm()
    writer = setup.target(Role.WRITER)
    if writer is None:
        raise LetterError("No writer model configured (enable LOCAL_WRITER or configure a cloud provider)")
    prefs, profile = get_prefs(), get_profile()
    editor = setup.target(Role.EDITOR)

    with _gpu_guard(writer, editor, setup.target(Role.HUMANIZER)):
        with session_scope() as s:
            job = s.get(Job, job_id)
            if job is None:
                raise LetterError("Job not found")
            lang = language if language in ("de", "en") else detect_language(job)[0]
            group_id = uuid.uuid4().hex
            templates = (select_templates(job, lang, prefs.writing.templates_per_letter, template)
                         if prefs.writing.use_templates else [])
            w = _writing_for(prefs.writing, templates)
            version = (WRITER_DE_VERSION if lang == "de" else WRITER_EN_VERSION) + ("+templates" if templates else "")
            fit = fit_from_analysis(job) if prefs.writing.use_fit_analysis else None
            # The applicant's own reasons (motivation/emphasis, never evidence) — snapshot them into this version.
            notes = get_notes(s, job.id)
            notes_text = notes_block(notes)
            variant = choose_variant(group_id, notes.has_motivation)
            # Only WHY_INTERESTED / WHY_FIT are reasons for applying; EMPHASIZE / AVOID alone still travel in the
            # notes block but must not make the writer believe it was given a motivation.
            mobility = mobility_applies(job, prefs.writing)
            if lang == "de":
                system = writer_system_de(w, bool(templates), bool(fit), notes.has_motivation, variant, mobility,
                                          job.country)
            else:
                system = writer_system_en(w, bool(templates), bool(fit), notes.has_motivation, variant, mobility,
                                          job.country)
            from app.services import agent_memory

            system += agent_memory.instructions_block("letter")  # what you taught the agent in earlier feedback
            user = writer_user(profile, job, w, lang, templates, fit, notes_text, notes.has_motivation,
                               research) + today_line(lang)
            tasks.progress(f"Writing draft ({_short(writer)})" + (f" from template “{templates[0].title}”" if templates else ""))
            res = llm_call(
                s, writer, kind="letter_draft", job_id=job.id, prompt_version=version,
                profile_hash=profile.fingerprint(), system=system, user=user,
                schema=CoverLetterDraft, use_cache=False,  # "Regenerate" must produce a new draft
            )
            calls = [res]
            words = word_count(res.data["letter"])
            if words < w.target_words_min * SHORT_DRAFT_FACTOR:  # one rewrite only when clearly too thin — never pad
                retry_note = (
                    f"\n\nDein letzter Entwurf hatte nur {words} Wörter. Schreibe ihn etwas ausführlicher, indem du konkreter "
                    "auf die Aufgaben der Stelle und die gewählten Belege eingehst – nicht auffüllen, keine Floskeln, "
                    "keine neuen Fakten, keine erfundene Motivation." if lang == "de" else
                    f"\n\nYour last draft had only {words} words. Make it a little fuller by engaging more concretely with "
                    "the actual tasks and the chosen evidence — no padding, no stock phrases, no new facts, no invented motivation."
                )
                tasks.progress(f"Draft too short ({words} words) — rewriting once ({_short(writer)})")
                retry = llm_call(
                    s, writer, kind="letter_draft", job_id=job.id, prompt_version=version,
                    profile_hash=profile.fingerprint(), system=system,
                    user=user + retry_note, schema=CoverLetterDraft, use_cache=False,
                )
                calls.append(retry)
                if word_count(retry.data["letter"]) > words:
                    res = retry
            draft = _store(
                s, job=job, group_id=group_id, stage="draft", language=lang, content=normalize_letter(res.data["letter"]),
                subject=res.data.get("subject"), source=None, target=writer, prompt_version=version, calls=calls,
                meta={"angle": res.data.get("angle"), "selected_evidence": res.data.get("selected_evidence", []),
                      "language_reason": None if language in ("de", "en") else detect_language(job)[1],
                      "templates": [{"id": t.id, "title": t.title} for t in templates],
                      "fit": fit, "personal_notes": notes.as_dict(), "notes_provided": not notes.is_empty,
                      "style_variant": variant.key, "strategy": strategy_for(notes), "research_used": bool(research)},
            )
            draft_id = draft.id
        # Each stage is committed before the next one, so a failing later stage never loses earlier work.
        last_id, base = draft_id, res.target
        wants_editor = prefs.writing.german_editor_by_default if use_editor is None else use_editor
        if lang == "de" and wants_editor and editor is not None:
            last_id, base = _edit_stage(job_id, draft_id, setup)
        wants_humanizer = prefs.writing.humanizer_by_default if use_humanizer is None else use_humanizer
        if wants_humanizer and setup.target(Role.HUMANIZER) is not None:
            last_id = _humanize_stage(job_id, last_id, setup)
        # Validation reuses the last *instruction* model (editor or writer): the humanizer can't fact-check.
        with session_scope() as s:
            _finalize(s, setup, s.get(Job, job_id), s.get(ApplicationDocument, last_id), base,
                      personalization_review=personalization_review)
            record_generation(s, job_id, notes)
    return group_id


def _edit_stage(job_id: int, source_document_id: int, setup: LLMSetup) -> tuple[int, RoleTarget]:
    """German style editor → committed 'edited' row. Returns (row id, target that answered)."""
    editor = setup.target(Role.EDITOR)
    prefs, profile = get_prefs(), get_profile()
    with _gpu_guard(editor), session_scope() as s:
        job, source = s.get(Job, job_id), s.get(ApplicationDocument, source_document_id)
        if job is None or source is None or source.job_id != job_id:
            raise LetterError("Draft not found")
        if source.language != "de":
            raise LetterError("The German style editor only edits German letters")
        templates = _group_templates(s, source.group_id)
        w = _writing_for(prefs.writing, templates)
        group_notes = _group_notes(s, source.group_id)
        notes_text = notes_block(group_notes)
        tasks.progress(f"German style editing ({_short(editor)})")
        res = llm_call(
            s, editor, kind="letter_edit", job_id=job.id,
            prompt_version=EDITOR_DE_VERSION + ("+templates" if templates else ""),
            profile_hash=profile.fingerprint(),
            system=editor_system_de(w, bool(templates), bool(group_notes and group_notes.has_motivation)),
            user=editor_user(profile, job, source.content, w, templates, notes_text), use_cache=False,
        )
        edited = _store(
            s, job=job, group_id=source.group_id, stage="edited", language=source.language,
            content=normalize_letter(res.data), subject=source.subject, source=source, target=editor,
            prompt_version=EDITOR_DE_VERSION + ("+templates" if templates else ""), calls=[res], meta={},
        )
        return edited.id, res.target


def _humanize_stage(job_id: int, source_document_id: int, setup: LLMSetup) -> int:
    """Humanizer rewrite → committed 'humanized' row (salutation/closing kept, drift reported in meta)."""
    target = setup.target(Role.HUMANIZER)
    with _gpu_guard(target), session_scope() as s:
        job, source = s.get(Job, job_id), s.get(ApplicationDocument, source_document_id)
        if job is None or source is None or source.job_id != job_id:
            raise LetterError("Letter version not found")
        try:
            result = humanize(source.content, keep_letter_frame=True, setup=setup, session=s, job_id=job.id)
        except HumanizerUnavailable as e:
            raise LetterError(str(e)) from e
        content = normalize_letter(result.text)
        doc = ApplicationDocument(
            job_id=job.id, group_id=source.group_id, stage="humanized", language=source.language,
            subject=source.subject, content=content, source_document_id=source.id, provider=target.provider.name,
            model=result.model, prompt_version=HUMANIZER_VERSION,
            meta={"word_count": word_count(content), "humanizer_warnings": result.warnings, "chunks": result.chunks,
                  "language_fallbacks": result.fallbacks, "humanized_stage": source.stage},
            input_tokens=result.input_tokens, output_tokens=result.output_tokens, cost_usd=0.0,
            duration_ms=result.duration_ms,
        )
        s.add(doc)
        s.flush()
        return doc.id


def _validation_base(setup: LLMSetup, source_stage: str) -> RoleTarget:
    """Instruction model that validates a humanized letter: the editor for edited letters, else the writer."""
    base = (setup.target(Role.EDITOR) if source_stage in ("edited", "final") else None) or setup.target(Role.WRITER) \
        or setup.target(Role.EDITOR)
    if base is None:
        raise LetterError("No instruction model available for factual validation")
    return base


def run_humanizer(job_id: int, source_document_id: int, setup: LLMSetup | None = None) -> int:
    """Humanizer pass on an existing letter version → new 'humanized' + 'final' rows in the same group."""
    setup = setup or get_llm()
    target = setup.target(Role.HUMANIZER)
    if target is None:
        raise LetterError("Humanizer is not configured (LOCAL_LLM_ENABLED + LOCAL_HUMANIZER_ENABLED)")
    with session_scope() as s:
        source = s.get(ApplicationDocument, source_document_id)
        if source is None or source.job_id != job_id:
            raise LetterError("Letter version not found")
        source_stage = source.stage
        if source_stage == "final":  # a final's own source tells whether it came from the editor
            parent = s.get(ApplicationDocument, source.source_document_id) if source.source_document_id else None
            source_stage = parent.stage if parent else "draft"
    base = _validation_base(setup, source_stage)
    with _gpu_guard(target, base):
        humanized_id = _humanize_stage(job_id, source_document_id, setup)
        with session_scope() as s:
            final = _finalize(s, setup, s.get(Job, job_id), s.get(ApplicationDocument, humanized_id), base)
            return final.id


def run_editor(job_id: int, source_document_id: int, setup: LLMSetup | None = None) -> int:
    """German style editor on an existing draft → new 'edited' + 'final' rows in the same group."""
    setup = setup or get_llm()
    editor = setup.target(Role.EDITOR)
    if editor is None:
        raise LetterError("German style editor is not configured (LOCAL_GERMAN_EDITOR_ENABLED or a cloud provider)")
    with _gpu_guard(editor):
        edited_id, answered = _edit_stage(job_id, source_document_id, setup)
        with session_scope() as s:
            final = _finalize(s, setup, s.get(Job, job_id), s.get(ApplicationDocument, edited_id), answered)
            return final.id


def revise_letter(job_id: int, group_id: str, changes: list[str], keywords: list[str] | None = None,
                  setup: LLMSetup | None = None, research: str = "") -> int:
    """Reviewer feedback → writer rewrites the group's latest final → committed 'revised' row → validated final
    (same fact check as every letter). Returns the new final document id. Used by the application agent."""
    setup = setup or get_llm()
    writer = setup.target(Role.WRITER)
    if writer is None:
        raise LetterError("No writer model configured (enable LOCAL_WRITER or configure a cloud provider)")
    profile = get_profile()
    with _gpu_guard(writer):
        with session_scope() as s:
            job = s.get(Job, job_id)
            source = s.scalars(select(ApplicationDocument).where(
                ApplicationDocument.job_id == job_id, ApplicationDocument.group_id == group_id,
                ApplicationDocument.stage == "final").order_by(ApplicationDocument.id.desc())).first()
            if job is None or source is None:
                raise LetterError("Finished letter not found")
            tasks.progress(f"Application agent: revising the cover letter with the reviewer's feedback ({_short(writer)})")
            res = llm_call(
                s, writer, kind="letter_revise", job_id=job.id, prompt_version=LETTER_REVISE_VERSION,
                profile_hash=profile.fingerprint(),
                system=letter_revise_system(source.language, get_prefs().writing.target_words_max)
                + __import__("app.services.agent_memory", fromlist=["x"]).instructions_block("letter"),
                user=letter_revise_user(profile.as_prompt_text(), job, source.content, changes, keywords or [],
                                        research) + today_line(source.language),
                use_cache=False,
            )
            revised = _store(
                s, job=job, group_id=group_id, stage="revised", language=source.language,
                content=normalize_letter(res.data), subject=source.subject, source=source,
                target=writer, prompt_version=LETTER_REVISE_VERSION, calls=[res],
                meta={"reviewer_changes": changes[:10], "reviewer_keywords": (keywords or [])[:8]},
            )
            revised_id, base = revised.id, res.target
        with session_scope() as s:
            final = _finalize(s, setup, s.get(Job, job_id), s.get(ApplicationDocument, revised_id), base,
                              personalization_review=False)
            return final.id


# --------------------------------------------------------------------------- triggers & queries


def finish_letter(job_id: int, group_id: str, setup: LLMSetup | None = None) -> int:
    """Completes a version a run left unfinished (app restart, model error after the draft): runs the German editor
    if a German draft never got edited (and the editor is on by default), otherwise validates the latest stage.
    Returns the new final document id."""
    setup = setup or get_llm()
    with session_scope() as s:
        docs = s.scalars(select(ApplicationDocument).where(ApplicationDocument.job_id == job_id,
                                                           ApplicationDocument.group_id == group_id)
                         .order_by(ApplicationDocument.id)).all()
        if not docs:
            raise LetterError("Letter version not found")
        if any(d.stage == "final" for d in docs):
            raise LetterError("This version is already finished")
        last = docs[-1]
        last_id, last_stage, language = last.id, last.stage, last.language
        edited = any(d.stage == "edited" for d in docs)
    editor = setup.target(Role.EDITOR)
    if (language == "de" and last_stage == "draft" and not edited and editor is not None
            and get_prefs().writing.german_editor_by_default):
        return run_editor(job_id, last_id, setup=setup)
    base = (editor if last_stage == "edited" and editor is not None else setup.target(Role.WRITER)) or editor
    if base is None:
        raise LetterError("No model available for factual validation")
    with _gpu_guard(base):
        with session_scope() as s:
            final = _finalize(s, setup, s.get(Job, job_id), s.get(ApplicationDocument, last_id), base)
            return final.id


def has_letters(session: Session, job_id: int) -> bool:
    return session.scalars(select(ApplicationDocument.id).where(ApplicationDocument.job_id == job_id).limit(1)).first() is not None


def letter_groups(session: Session, job_id: int) -> list[dict[str, Any]]:
    docs = session.scalars(
        select(ApplicationDocument).where(ApplicationDocument.job_id == job_id)
        .order_by(ApplicationDocument.created_at, ApplicationDocument.id)
    ).all()
    groups: dict[str, dict[str, Any]] = {}
    for d in docs:
        g = groups.setdefault(d.group_id, {"group_id": d.group_id, "created_at": d.created_at, "language": d.language, "docs": []})
        g["docs"].append(d)
    out = sorted(groups.values(), key=lambda g: g["created_at"], reverse=True)
    for g in out:
        finals = [d for d in g["docs"] if d.stage == "final"]
        drafts = [d for d in g["docs"] if d.stage == "draft"]
        g["final"] = finals[-1] if finals else None
        g["draft"] = drafts[-1] if drafts else None
    return out
