"""Posting keywords and a forgiving German/English keyword matcher.

Shared by the CV tailoring (which bullets/skills are relevant) and the ATS check (which posting keywords the CV
covers). ATS keyword matching is mostly literal, so the matcher is literal first and only then falls back to the
key words of a phrase (with a small English→German stem map, because analyses are English and CVs often German).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.services.evidence import TECH_TERMS

# Requirements come from the (English) analysis, letters/CVs are often German.
EN_DE: dict[str, tuple[str, ...]] = {
    "ai": ("ki",), "development": ("entwicklung", "entwickl"), "developer": ("entwickler",), "software": ("software",),
    "infrastructure": ("infrastruktur",), "automation": ("automatisier",), "automated": ("automatisier",),
    "testing": ("test",), "tests": ("test",), "test": ("test",), "machine": ("maschinell",), "learning": ("lernen",),
    "workflows": ("workflow", "abläuf", "prozess"), "workflow": ("workflow", "ablauf", "prozess"),
    "pipelines": ("pipeline",), "integration": ("integration", "integrier"), "data": ("daten",),
    "visualization": ("visualisier",), "security": ("sicherheit",), "requirements": ("anforderung",),
    "engineering": ("entwicklung", "engineering"), "app": ("app", "anwendung"), "applications": ("anwendung",),
    "systems": ("system",), "embedded": ("eingebettet", "embedded"), "analysis": ("analyse",),
    "communication": ("kommunikation",), "teamwork": ("team",), "english": ("englisch",), "german": ("deutsch",),
    "networks": ("netzwerk",), "network": ("netzwerk",), "networking": ("netzwerk",), "classification": ("klassifikation",),
    "computer": ("informatik", "computer"), "science": ("informatik", "science"), "robotics": ("robotik",),
    "administration": ("administration",), "troubleshooting": ("fehleranalyse", "behebung"),
}

STOP_WORDS = {"and", "with", "for", "the", "und", "mit", "der", "die", "das", "von", "experience", "erfahrung",
              "knowledge", "kenntnisse", "skills", "basic", "gute", "sehr", "strong", "solid", "using", "based",
              "in", "of", "or", "oder", "to", "a", "an", "im", "eine", "ein", "years", "year", "jahre", "jahren",
              "plus", "good", "excellent", "proven", "working", "familiarity", "understanding", "ability"}

_WORD = re.compile(r"[a-zäöüß0-9+#./-]{2,}")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())


def literal_in(term: str, text: str) -> bool:
    """`term` as a whole token sequence in `text` (handles C++, C#, .NET, CI/CD, Node.js)."""
    t = _norm(term).strip()
    if not t:
        return False
    return re.search(r"(?<![a-z0-9äöüß])" + re.escape(t) + r"(?![a-z0-9äöüß])", _norm(text)) is not None


def plural_variants(term: str) -> list[str]:
    """'api' → ['apis'], 'databases' → ['database'], 'REST APIs' → ['REST API']. Only for plain words of 3+ letters."""
    t = term.strip()
    if not re.fullmatch(r"[A-Za-zÄÖÜäöüß][\w .-]*[A-Za-zÄÖÜäöüß]", t) or len(t) < 3:
        return []
    if t.lower().endswith("ses") or t.lower().endswith(("xes", "ches", "shes")):
        return [t[:-2], t[:-1]]
    if t.lower().endswith("s") and not t.lower().endswith("ss"):
        return [t[:-1]]
    return [t + "s", t + "es"] if t.lower().endswith(("x", "ch", "sh")) else [t + "s"]


def term_in(term: str, text: str) -> bool:
    """Literal match (singular or plural: "API" / "APIs"), else most key words of the phrase (with EN→DE stems)
    appear in the text."""
    if literal_in(term, text) or any(literal_in(v, text) for v in plural_variants(term)):
        return True
    words = [w for w in _WORD.findall(_norm(term).replace("-", " ")) if w not in STOP_WORDS]
    if not words or len(words) == 1 and len(words[0]) < 4:
        return False
    low = _norm(text)
    text_words = set(_WORD.findall(low.replace("-", " ")))

    def present(word: str) -> bool:
        for option in {word, *EN_DE.get(word, ())}:
            if len(option) < 5:
                if literal_in(option, low):
                    return True
            elif option in low or any(tw.startswith(option[:5]) for tw in text_words):
                return True
        return False

    return sum(present(w) for w in words) / len(words) >= 0.6


Priority = Literal["required", "preferred"]


@dataclass
class Keyword:
    term: str
    priority: Priority
    source: str  # analysis | posting

    @property
    def weight(self) -> int:
        return 2 if self.priority == "required" else 1

    def as_dict(self) -> dict[str, str]:
        return {"term": self.term, "priority": self.priority, "source": self.source}


# Short terms that are real skills; other 1-2 letter terms ("C", "Go") match far too much ("C++", "go-live").
_SHORT_OK = {"ai", "ki", "ml", "qt", "c#", "r&d"}


def usable_term(term: str | None) -> bool:
    t = (term or "").strip().lower()
    return len(t) >= 3 or t in _SHORT_OK


def _core_terms(phrase: str, vocab: list[str]) -> list[str]:
    """'3+ years of Python and Docker experience' → ['Python', 'Docker']; short phrases are kept as they are."""
    found: dict[str, str] = {}
    for v in vocab:  # one entry per term, whatever its case ("java" from cv.yaml keywords, "Java" from the profile)
        if literal_in(v, phrase):
            key = _norm(v).strip()
            if key not in found or found[key].islower() and not v.islower():
                found[key] = v
    if found:
        # drop terms contained in a longer found term ("Spring" inside "Spring Boot")
        return [v for k, v in found.items() if not any(k != g and literal_in(k, g) for g in found)]
    cleaned = re.sub(r"^\W*(\d+\+?\s*(years?|jahre)\s*(of)?\s*)", "", phrase.strip(), flags=re.I)
    cleaned = re.sub(r"\b(experience|erfahrung|knowledge|kenntnisse)\s+(with|in|of|mit)\b\s*", "", cleaned, flags=re.I)
    return [cleaned.strip(" .;:")] if 0 < len(cleaned.split()) <= 4 else []


def job_keywords(job, vocab: list[str], limit: int = 30) -> list[Keyword]:
    """Keywords an ATS would screen this posting for: the analysis' requirement list (reduced to core terms) plus
    every known skill/technology named in the posting. Required = in the analysis' requirements, title or the
    posting's requirements section; preferred = nice-to-have or only mentioned in the description."""
    vocab = sorted({v.strip() for v in [*vocab, *TECH_TERMS] if usable_term(v)}, key=len, reverse=True)
    out: dict[str, Keyword] = {}

    def add(term: str, priority: Priority, source: str) -> None:
        key = _norm(term).strip()
        if not key or len(key) > 60:
            return
        if key in out:
            if priority == "required":
                out[key].priority = "required"
            if out[key].term.islower() and not term.strip().islower():  # prefer "Java" over "java"
                out[key].term = term.strip()
            return
        out[key] = Keyword(term.strip(), priority, source)

    a = job.current_analysis.result if getattr(job, "current_analysis", None) and job.current_analysis.result else {}
    for key, priority in (("strong_matches", "required"), ("partial_matches", "required"),
                          ("critical_gaps", "required"), ("nice_to_have_gaps", "preferred")):
        for item in a.get(key) or []:
            phrase = str(item.get("skill") if isinstance(item, dict) else item)
            for term in _core_terms(phrase, vocab):
                add(term, priority, "analysis")
    title, req = job.title or "", "\n".join(x for x in (job.requirements, job.responsibilities) if x)
    desc = job.description or ""
    for v in vocab:
        if literal_in(v, title) or (req and literal_in(v, req)):
            add(v, "required", "posting")
        elif literal_in(v, desc):
            add(v, "preferred", "posting")
    ranked = sorted(out.values(), key=lambda k: (k.priority != "required", k.source != "analysis"))
    return ranked[:limit]


def relevance(text: str, keywords: list[Keyword]) -> int:
    return sum(k.weight for k in keywords if term_in(k.term, text))
