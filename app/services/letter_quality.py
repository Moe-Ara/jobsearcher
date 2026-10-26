"""Personalization quality of a finished letter — a lightweight, explainable check (not an AI detector).

Deterministic signals (generic phrases, vacancy-specific references, concrete candidate evidence, validation
results) are combined with the reviewer model's content-specificity verdict ("could this be sent elsewhere?").
The result is advice for the human reviewer; it never blocks a letter.
"""

from __future__ import annotations

import re
from typing import Any

from app.llm.letter_prompts import AVOID_PHRASES_DE, AVOID_PHRASES_EN, INVENTED_MOTIVATION_DE, INVENTED_MOTIVATION_EN
from app.llm.schemas import SUPPORTED_STATUSES

# Boilerplate patterns beyond the exact phrase lists (inflections, word order).
_GENERIC_PATTERNS = [
    r"mit (großem|grossem) interesse",
    r"mit (großer|grosser) begeisterung",
    r"hat (sofort|direkt) mein interesse geweckt",
    r"(ihr|ihre|ihres) (innovativ\w*|zukunftsweisend\w*|zukunftsorientiert\w*) (unternehmen|lösungen|unternehmenskultur)",
    r"meine leidenschaft (für|gilt)",
    r"passt perfekt zu meinem profil",
    r"ich bin (fest )?(davon )?überzeugt, dass ich",
    r"(ihr|ihre) team (optimal|ideal|perfekt) (ergänzen|verstärken)",
    r"spannend\w* und dynamisch\w*",
    r"einzigartige kombination",
    r"wertvolle synergien",
    r"hiermit bewerbe ich mich",
    r"i am writing to express my (strong )?interest",
    r"(perfect|ideal) fit",
    r"(fast-paced|dynamic) (and )?(fast-paced |dynamic )?environment",
    r"my passion for",
    r"i am (confident|convinced) that i",
    r"unique combination of",
]

# Words too common in vacancies AND letters to count as a vacancy-specific reference.
_COMMON = {
    "erfahrung", "kenntnisse", "aufgaben", "stelle", "team", "unternehmen", "bereich", "entwicklung", "arbeit",
    "projekte", "projekt", "kunden", "mitarbeiter", "bewerbung", "position", "möglichkeit", "gespräch", "rahmen",
    "experience", "skills", "team", "company", "role", "position", "work", "development", "knowledge", "projects",
    "software", "softwareentwicklung", "studium", "deutsch", "englisch", "german", "english", "sehr", "gute",
    "eigenverantwortlich", "zusammenarbeit", "verantwortung", "weiterentwicklung", "mehrjährige", "abgeschlossenes",
    "kommunikationsfähigkeit", "teamfähigkeit", "flexibilität", "motivation", "standort", "vollzeit", "unbefristet",
    # verbs and fillers that every letter and every vacancy share
    "arbeiten", "arbeitest", "arbeite", "einsetzen", "entwickeln", "entwickelst", "unterstützen", "unterstützt",
    "bringen", "bringst", "suchen", "suchst", "bieten", "bietest", "gestalten", "umsetzen", "verbinden", "anforderungen",
    "bereichen", "themen", "lösungen", "systeme", "prozesse", "working", "develop", "support", "solutions", "systems",
}
# Example sentences from the writer prompt: a letter that reuses them is a template, not a person.
PROMPT_EXAMPLE_SENTENCES = [
    "An der ausgeschriebenen Position interessiert mich besonders die Verbindung von Softwareentwicklung und angewandter KI.",
    "Genau an dieser Schnittstelle habe ich mich sowohl in meinem Masterstudium als auch in eigenen technischen Projekten intensiv bewegt.",
    "Besonders interessant finde ich, dass die Stelle klassische Softwareentwicklung mit KI-gestützten Entscheidungsprozessen verbindet.",
    "Mit containerisierten Anwendungen, CI/CD und technischen Infrastrukturen habe ich bereits praktisch gearbeitet und kann mich daher schnell in eine neue Cloud-Umgebung einarbeiten.",
]
# Required letter formalities are not "generic" content.
_FORMALITY = re.compile(r"^\s*(sehr geehrte|mit freundlichen grüßen|dear|kind regards|best regards)\b", re.IGNORECASE)

_TERM = re.compile(r"[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß0-9+#./-]{4,}")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zäöüß0-9]+", text.lower())


def copied_prompt_examples(letter: str, window: int = 7) -> list[str]:
    """Prompt example sentences the letter reuses (any run of `window` consecutive words)."""
    letter_text = " ".join(_words(letter))
    hits = []
    for sentence in PROMPT_EXAMPLE_SENTENCES:
        words = _words(sentence)
        if any(" ".join(words[i:i + window]) in letter_text for i in range(0, max(1, len(words) - window + 1))):
            hits.append(sentence)
    return hits


def generic_phrase_warnings(letter: str) -> list[str]:
    text = (letter or "").lower()
    found: list[str] = [f"copied prompt example: “{s[:60]}…”" for s in copied_prompt_examples(letter)]
    for phrase in AVOID_PHRASES_DE + AVOID_PHRASES_EN + INVENTED_MOTIVATION_DE + INVENTED_MOTIVATION_EN:
        if phrase.lower() in text and phrase not in found:
            found.append(phrase)
    for pattern in _GENERIC_PATTERNS:
        m = re.search(pattern, text)
        if m and not any(m.group(0) in f.lower() or f.lower() in m.group(0) for f in found):
            found.append(m.group(0))
    return found


def _terms(text: str) -> set[str]:
    return {t.lower().strip(".-/") for t in _TERM.findall(text or "")}


def job_reference_terms(letter: str, job, profile_text: str) -> list[str]:
    """Vacancy words the letter picks up that are neither common filler nor from the applicant's own profile."""
    job_text = " ".join(str(x or "") for x in (job.title, job.description, job.requirements, job.responsibilities))
    company = (job.company or "").lower()
    exclude = _COMMON | _terms(profile_text) | {w for w in company.split() if len(w) > 2}
    shared = (_terms(letter) & _terms(job_text)) - exclude
    return sorted(shared)


def candidate_evidence(validation: dict[str, Any]) -> list[str]:
    """Distinct supported claims about the candidate in the final letter."""
    claims = (validation or {}).get("claims") or []
    seen: list[str] = []
    for c in claims:
        if c.get("status") in SUPPORTED_STATUSES:
            key = (c.get("profile_evidence") or c.get("claim") or "").strip().lower()[:80]
            if key and key not in seen:
                seen.append(key)
    return seen


def evidence_level_violations(validation: dict[str, Any]) -> int:
    """Claims that overstated an evidence level (academic/project presented as professional), guard or model."""
    return sum(1 for c in (validation or {}).get("claims") or [] if "Evidence-level" in (c.get("note") or ""))


def factual_warning_count(validation: dict[str, Any]) -> int:
    kinds = {"ambiguous", "unsupported", "removed", "number", "template", "placeholder"}
    return sum(1 for w in (validation or {}).get("warnings") or [] if w.get("type") in kinds)


def assess(letter: str, job, profile_text: str, notes_provided: bool, validation: dict[str, Any],
           review: dict[str, Any] | None, has_motivation: bool | None = None) -> dict[str, Any]:
    has_motivation = notes_provided if has_motivation is None else has_motivation
    generic = generic_phrase_warnings(letter)
    job_refs = job_reference_terms(letter, job, profile_text)
    evidence = candidate_evidence(validation)
    unsupported = sum(1 for c in (validation or {}).get("claims") or [] if c.get("status") == "UNSUPPORTED")
    violations = evidence_level_violations(validation)

    if review:
        genericity = review.get("could_be_sent_elsewhere") or "PARTLY"
        uses_notes = bool(has_motivation and review.get("uses_personal_motivation"))
        specific_count = len(review.get("job_specific_references") or [])
        generic += [s for s in (review.get("generic_sentences") or []) if s not in generic and not _FORMALITY.match(s)][:5]
        reviewed_by = "model"
    else:
        # Reviewer unavailable: judge specificity from vacancy words alone.
        specific_count = len(job_refs)
        genericity = "NO" if specific_count >= 4 else "PARTLY" if specific_count >= 2 else "YES"
        uses_notes = has_motivation  # unknown — assume the writer followed them
        reviewed_by = "heuristic"

    if genericity == "YES" or (not uses_notes and specific_count < 2) or len(generic) >= 4:
        quality = "LOW"
    elif uses_notes and 2 <= len(evidence) and specific_count >= 2 and len(generic) <= 1 and genericity == "NO" and unsupported == 0:
        quality = "HIGH"
    else:
        quality = "MEDIUM"

    messages = []
    if not notes_provided:
        messages.append("No personal application notes were provided. This letter may be less distinctive.")
    elif not has_motivation:
        messages.append("Your notes say what to emphasize/avoid, but not why the job interests you or why you fit — "
                        "the letter's reasons come only from the technical overlap.")
    if genericity == "YES":
        messages.append("This letter may be too generic. Add personal application notes or regenerate.")
    return {
        "uses_personal_notes": uses_notes,
        "notes_provided": notes_provided,
        "specific_job_references_count": specific_count,
        "job_reference_terms": job_refs[:12],
        "concrete_candidate_evidence_count": len(evidence),
        "generic_phrase_warnings": generic,
        "unsupported_claims": unsupported,
        "evidence_level_violations": violations,
        "could_be_sent_elsewhere": genericity,
        "personalization_quality": quality,
        "factual_warnings": factual_warning_count(validation),
        "reviewed_by": reviewed_by,
        "messages": messages,
    }
