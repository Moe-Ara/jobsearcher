"""Human-written cover-letter templates: the writer copies their STRUCTURE and voice, never their facts.

Templates live as plain files in config/letter_templates/<id>.md (YAML front matter + letter text), so they can be
added on the Templates page or dropped into the folder. For each letter the best-matching templates (same language,
most word overlap with the vacancy and the template's tags) are passed to the writer and the German editor as
delimited <EXAMPLE_LETTER> blocks, together with a measured outline (paragraphs, words, sentences, how each paragraph
opens). A deterministic leak check flags names, numbers, technologies and long phrases that came from a template
instead of the profile.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from app.settings import get_settings

MAX_CHARS = 12000
_override_dir: Path | None = None  # tests point this at a temp dir

_GREETING = re.compile(r"^(sehr geehrte|liebe[rs]?|hallo|guten tag|moin|dear|hello|hi|to whom)\b", re.IGNORECASE)
_CLOSING = re.compile(
    r"^(mit (den )?(freundlichen|besten) grüßen|freundliche grüße|viele grüße|beste grüße|herzliche grüße|"
    r"kind regards|best regards|warm regards|regards|sincerely|yours sincerely|yours faithfully|best)\b,?", re.IGNORECASE)
_WORD = re.compile(r"[a-zäöüß][a-zäöüß0-9+#.-]{2,}")
_STOP = {
    "und", "der", "die", "das", "ich", "mit", "für", "bei", "ist", "eine", "ein", "von", "auf", "nicht", "auch", "habe",
    "meine", "mein", "sie", "wir", "als", "den", "dem", "des", "dass", "sich", "mehr", "über", "sind", "werden", "oder",
    "ihre", "ihr", "ihnen", "the", "and", "with", "for", "have", "has", "you", "your", "our", "are", "was", "this", "that",
    "from", "will", "would", "been", "into", "which", "their", "they", "them", "about", "sehr", "geehrte", "damen",
    "herren", "grüßen", "freundlichen", "dear", "regards", "kind", "team",
}


@dataclass
class LetterTemplate:
    id: str
    title: str
    language: str  # de | en
    text: str
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    created: str = ""

    @property
    def body(self) -> str:
        return split_frame(self.text)[1]

    @property
    def paragraphs(self) -> list[str]:
        return [p.strip() for p in re.split(r"\n\s*\n", self.body) if p.strip()]

    @property
    def body_words(self) -> int:
        return len(self.body.split())

    def outline(self) -> str:
        return structure_outline(self.text, self.language)


def templates_dir() -> Path:
    return _override_dir or (get_settings().config_dir / "letter_templates")


# --------------------------------------------------------------------------- text helpers


def clean_text(text: str) -> str:
    """Pasted text → plain letter text: no HTML tags, no control characters, normalised blank lines."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"<[^>\n]{1,200}>", "", text)  # stray HTML from copy & paste
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:MAX_CHARS]


_PLACEHOLDER = re.compile(r"\[[^\]\n]{1,40}\]")


def salutation_line(lines: list[str], is_greeting=None) -> int | None:
    """Index of the salutation that opens the body: the LAST greeting line of the header zone. The zone is everything
    above the first line of running text (header lines — sender, recipient, date, subject — are short). A full header
    block easily runs past 12 lines, and a letter can carry a stray greeting above its header."""
    is_greeting = is_greeting or (lambda ln: bool(_GREETING.match(ln)))
    found = None
    for i, ln in enumerate(lines[:45]):
        s = ln.strip()
        if len(s) > 110:  # running text: the header zone is over
            break
        if s and is_greeting(s):
            found = i
    return found


def strip_placeholders(text: str) -> str:
    """'seit [vier] Jahren bei der [Firma]' → 'seit … Jahren bei der …' — template letters are structure examples;
    bracket placeholders in them only teach a model to write placeholders."""
    return _PLACEHOLDER.sub("…", text or "")


def split_frame(text: str) -> tuple[str, str, str]:
    """(salutation, body, closing). Lines before the salutation (address blocks, date, subject) are dropped."""
    lines = (text or "").strip().split("\n")
    start = salutation_line(lines)
    head = lines[start].strip() if start is not None else ""
    begin = start + 1 if start is not None else 0
    end = next((k for k in range(len(lines) - 1, begin - 1, -1) if _CLOSING.match(lines[k].strip())), None)
    body = "\n".join(lines[begin:end if end is not None else len(lines)]).strip()
    tail = "\n".join(lines[end:]).strip() if end is not None else ""
    return head, body, tail


def _sentences(paragraph: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ„\"(])", paragraph.strip()) if s.strip()]


def structure_outline(text: str, language: str = "de") -> str:
    head, body, tail = split_frame(text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    de = language == "de"
    out = [f"{'Anrede' if de else 'Salutation'}: {head or '—'}"]
    for i, p in enumerate(paragraphs, 1):
        sents = _sentences(p)
        lengths = [len(s.split()) for s in sents]
        opening = " ".join(p.split()[:6])
        out.append(
            (f"Absatz {i}: {len(p.split())} Wörter, {len(sents)} Sätze (Satzlängen {lengths}); beginnt mit „{opening} …“")
            if de else
            (f"Paragraph {i}: {len(p.split())} words, {len(sents)} sentences (sentence lengths {lengths}); opens with \"{opening} …\"")
        )
    out.append(f"{'Gruß' if de else 'Closing'}: {tail.splitlines()[0] if tail else '—'}")
    return "\n".join(out)


def detect_language(text: str) -> str:
    words = re.findall(r"[a-zäöüß]+", text.lower())
    de = sum(w in {"und", "ich", "der", "die", "das", "mit", "für", "bei", "sie", "ihre", "habe", "meine"} for w in words)
    en = sum(w in {"the", "and", "with", "for", "you", "your", "have", "my", "i", "this", "that", "our"} for w in words)
    return "de" if de >= en else "en"


# --------------------------------------------------------------------------- storage


def _slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss"))
    return s.strip("-")[:40] or "template"


def _parse(path: Path) -> LetterTemplate | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    meta: dict = {}
    text = raw
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", raw, re.DOTALL)
    if m:
        try:
            meta = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        text = m.group(2)
    text = clean_text(text)
    if not text:
        return None
    lang = str(meta.get("language") or "").lower()
    return LetterTemplate(
        id=path.stem, title=str(meta.get("title") or path.stem), language=lang if lang in ("de", "en") else detect_language(text),
        text=text, tags=[str(t).strip() for t in (meta.get("tags") or []) if str(t).strip()],
        notes=str(meta.get("notes") or ""), created=str(meta.get("created") or ""),
    )


def list_templates(language: str | None = None) -> list[LetterTemplate]:
    d = templates_dir()
    if not d.is_dir():
        return []
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".md", ".txt") and not p.stem.upper().startswith("README"))
    items = [t for p in files if (t := _parse(p))]
    return [t for t in items if language is None or t.language == language]


def get_template(template_id: str) -> LetterTemplate | None:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", template_id or "") or template_id.upper().startswith("README"):
        return None
    path = _path_for(template_id)
    return _parse(path) if path else None


def _path_for(template_id: str) -> Path | None:
    for ext in (".md", ".txt"):
        path = templates_dir() / f"{template_id}{ext}"
        if path.is_file():
            return path
    return None


def save_template(title: str, text: str, *, language: str = "auto", tags: list[str] | str | None = None,
                  notes: str = "", template_id: str | None = None) -> LetterTemplate:
    text = clean_text(text)
    if len(text.split()) < 40:
        raise ValueError("Template is too short — paste a complete letter (at least ~40 words)")
    title = (title or "").strip()[:120] or " ".join(text.split()[:6])
    lang = language if language in ("de", "en") else detect_language(text)
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    tags = [t for t in (tags or []) if t][:20]
    existing = get_template(template_id) if template_id else None
    tid = existing.id if existing else f"{_slug(title)}-{hashlib.sha1(text.encode()).hexdigest()[:6]}"
    meta = {"title": title, "language": lang, "tags": tags, "notes": notes.strip()[:500],
            "created": existing.created if existing else datetime.now(UTC).strftime("%Y-%m-%d %H:%M")}
    d = templates_dir()
    d.mkdir(parents=True, exist_ok=True)
    old = _path_for(tid)
    if old is not None and old.suffix != ".md":  # a dropped-in .txt becomes .md once edited on the page
        old.unlink()
    (d / f"{tid}.md").write_text(f"---\n{yaml.safe_dump(meta, allow_unicode=True, sort_keys=False)}---\n{text}\n", encoding="utf-8")
    return get_template(tid)  # type: ignore[return-value]


def delete_template(template_id: str) -> bool:
    t = get_template(template_id)
    if t is None:
        return False
    path = _path_for(t.id)
    if path:
        path.unlink(missing_ok=True)
    return True


# --------------------------------------------------------------------------- selection


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP}


def select_templates(job, language: str, k: int, choice: str | None = None) -> list[LetterTemplate]:
    """choice: None/'auto' → best k by language + overlap with the vacancy; 'none' → []; else that template id."""
    if choice == "none" or k <= 0:
        return []
    if choice and choice != "auto":
        t = get_template(choice)
        return [t] if t else []
    pool = list_templates(language)
    if not pool:
        return []
    job_tokens = _tokens(" ".join(str(x or "") for x in (job.title, job.title, job.description, job.requirements,
                                                          job.responsibilities)))

    def score(t: LetterTemplate) -> float:
        tag_hits = sum(1 for tag in t.tags if _tokens(tag) & job_tokens)
        overlap = len(_tokens(t.text) & job_tokens) / (len(_tokens(t.text)) or 1)
        return tag_hits * 2 + overlap

    return sorted(pool, key=lambda t: (-score(t), t.id))[:k]


def target_words(templates: list[LetterTemplate], default: tuple[int, int]) -> tuple[int, int]:
    """Length target from the examples (±15 % around their average body length), within sane bounds."""
    if not templates:
        return default
    avg = sum(t.body_words for t in templates) / len(templates)
    return max(120, int(avg * 0.85)), min(500, max(150, int(avg * 1.15)))


# --------------------------------------------------------------------------- leak check


_CAPITALISED = re.compile(r"(?<![\w-])[A-ZÄÖÜ][\w&.+#/-]*(?:\s+[A-ZÄÖÜ][\w&.+#/-]*)*")
_NUM = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?(?![\w])")
_NUMBER_WORDS = {"zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn", "elf", "zwölf", "zwanzig",
                 "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "twenty"}
_FUNCTION_WORDS = _STOP | {
    "als", "in", "im", "bei", "mit", "seit", "während", "durch", "für", "zum", "zur", "nach", "neben", "auch", "dort",
    "dabei", "darüber", "zudem", "außerdem", "besonders", "gerade", "hier", "so", "da", "wie", "wenn", "weil", "um",
    "at", "as", "my", "during", "in", "on", "since", "while", "there", "here", "when", "because", "also", "i", "we",
    "an", "a", "ihr", "ihre", "ihrem", "ihren", "ihrer", "unser", "unsere", "meiner", "meinem", "meinen",
}


def leakage(letter: str, templates: list[LetterTemplate], known_text: str) -> list[dict[str, str]]:
    """Facts or long phrases in the letter that come from a template but not from the profile/vacancy (`known_text`).
    German capitalises every noun, so only names/terms absent from profile+vacancy and present in the template count."""
    if not templates:
        return []
    known = known_text.lower()
    lowered_letter = letter.lower()
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for t in templates:
        _, tbody, _ = split_frame(t.text)
        # numbers (digits and spelled out: "drei Monate" from a sample letter is still the sample's fact)
        spelled = {w for w in re.findall(r"[a-zäöüß]+", tbody.lower()) if w in _NUMBER_WORDS}
        for w in sorted(spelled & set(re.findall(r"[a-zäöüß]+", lowered_letter))):
            if not re.search(rf"\b{w}\b", known) and w not in seen:
                seen.add(w)
                out.append({"type": "template", "detail": f"Number “{w}” appears in template “{t.title}” but not in your profile"})
        for n in set(_NUM.findall(tbody)) & set(_NUM.findall(letter)):
            if n not in known and n not in seen:
                seen.add(n)
                out.append({"type": "template", "detail": f"Number “{n}” appears in template “{t.title}” but not in your profile"})
        # names / proper terms (multi-word capitalised sequences, or single words that are clearly names/brands)
        for raw_term in {m.group(0).strip() for m in _CAPITALISED.finditer(tbody)}:
            words = raw_term.split()
            while words and words[0].lower() in _FUNCTION_WORDS:  # sentence-initial "Als Werkstudent", "In Berlin"
                words = words[1:]
            words = [w.rstrip(".,;:") for w in words]
            # German capitalises nouns, so "Contoso Systems GmbH Testautomatisierung" is one run: try the longest
            # sub-span that occurs in the letter.
            spans = [words[i:j] for n in range(len(words), 0, -1) for i in range(len(words) - n + 1) for j in [i + n]]
            for span in spans:
                term = " ".join(span)
                distinctive = len(span) > 1 or re.search(r"[A-Z].*[A-Z0-9+#/]|\d", term) is not None
                if len(term) < 3 or not distinctive or term in seen:
                    continue
                if re.search(rf"(?<![\w-]){re.escape(term.lower())}(?![\w-])", lowered_letter):
                    if term.lower() not in known:
                        seen.add(term)
                        out.append({"type": "template", "detail": f"“{term}” comes from template “{t.title}”, not from your profile"})
                    break
        # long verbatim phrases (8+ words)
        tw = re.findall(r"\w+", tbody.lower())
        lw = " ".join(re.findall(r"\w+", lowered_letter))
        copied = 0
        for i in range(0, max(0, len(tw) - 7)):
            phrase = " ".join(tw[i:i + 8])
            if phrase in lw:
                copied += 1
        if copied:
            out.append({"type": "template", "detail": f"{copied} eight-word phrase(s) copied verbatim from template “{t.title}” — reword them"})
    return out
