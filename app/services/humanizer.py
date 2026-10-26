"""Humanizer: optional natural-rewrite pass with a dedicated local model (default: jialinyyzz/humanizer, Gemma-4-12B
fine-tune). Used as an extra letter stage and standalone (Humanize page / API).

The model is a *completion* model trained on one exact prompt, so it is called in raw mode with that prompt byte for
byte and the authors' samplers (preferences.yaml → local_llm.roles.humanizer). It was trained on English and Chinese:
German works only as well as the base model allows, so every result is guarded:
  - salutation and closing are kept out of the rewrite and re-attached (the model sometimes drops sign-offs);
  - text is rewritten paragraph-group by paragraph-group (~350 words, as recommended);
  - a paragraph whose language changed (e.g. German → English) is replaced by the original and reported;
  - numbers and technical names that disappear or appear are reported as warnings.
Letters still go through the normal factual validation afterwards.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import time
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.llm.factory import LLMSetup, Role, get_llm
from app.llm.local_provider import GPU_LOCK
from app.models import JobAnalysis
from app.services import tasks

PROMPT_VERSION = "humanizer-v1"

# Exactly as published in the model's USAGE.md (do not reword or translate — the model was trained on this string).
INSTR = (
    "Rewrite the text below so it reads like a person wrote it, not a language model.\n"
    "\n"
    "Reorganize it as you see fit. Vary sentence length on purpose. Cut hedging,\n"
    "throat-clearing, and any sentence that only announces what comes next.\n"
    "Prefer the concrete word over the abstract one. It is fine to sound uneven.\n"
    "\n"
    "Every fact, number, unit, date, name and quotation must survive unchanged."
)
SEP = "\n\n### Rewritten:\n\n"
MAX_CHUNK_WORDS = 350

_GREETING = re.compile(r"^(sehr geehrte|liebe[rs]?|hallo|guten tag|moin|dear|hello|hi)\b", re.IGNORECASE)
_CLOSING = re.compile(
    r"^(mit (den )?(freundlichen|besten) grüßen|freundliche grüße|viele grüße|beste grüße|herzliche grüße|"
    r"kind regards|best regards|warm regards|regards|sincerely|yours sincerely|yours faithfully)\b", re.IGNORECASE)
_DE = {"und", "der", "die", "das", "ich", "mit", "für", "bei", "ist", "eine", "ein", "zu", "von", "auf", "nicht",
       "auch", "habe", "meine", "mein", "sie", "wir", "als", "im", "den", "dem", "des", "dass", "es", "sich", "mehr",
       "über", "sind", "werden", "wurde", "oder", "aber", "noch", "nur", "einen", "einem", "einer", "ihre", "ihr"}
_EN = {"the", "and", "with", "for", "have", "has", "my", "is", "of", "to", "in", "on", "not", "also", "this", "that",
       "we", "you", "as", "at", "an", "i", "a", "more", "than", "it", "your", "our", "are", "was", "be", "will",
       "from", "but", "or", "which", "would", "been"}
_TERM = re.compile(r"(?<![\w-])(?:[A-Z][\w]*[A-Z0-9+#/][\w+#./-]*|[A-Z]{2,}[\w+#]*|[A-Za-z]+\+\+|C#|\.NET)(?![\w])")
_NUMBER = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?(?![\w])")


class HumanizerUnavailable(RuntimeError):
    pass


@dataclass
class HumanizeResult:
    text: str
    model: str
    chunks: int
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    fallbacks: int = 0
    warnings: list[str] = field(default_factory=list)


def build_prompt(draft: str) -> str:
    return INSTR + "\n\n" + draft.strip() + SEP


def text_language(text: str) -> str | None:
    words = re.findall(r"[a-zäöüß]+", (text or "").lower())
    de, en = sum(w in _DE for w in words), sum(w in _EN for w in words)
    if de + en < 4:
        return None
    return "de" if de > en else "en"


def split_letter_frame(text: str) -> tuple[str, str, str]:
    """(salutation, body, closing) — the frame is kept verbatim, only the body is rewritten."""
    lines = (text or "").strip().split("\n")
    head: list[str] = []
    start = 0
    if lines and _GREETING.match(lines[0].strip()):
        head, start = [lines[0]], 1
    end = next((k for k in range(len(lines) - 1, start - 1, -1) if _CLOSING.match(lines[k].strip())), None)
    body = "\n".join(lines[start:end if end is not None else len(lines)]).strip()
    tail = "\n".join(lines[end:]) if end is not None else ""
    return "\n".join(head), body, tail


def chunk_paragraphs(body: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body or "") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for p in paragraphs:
        if current and len(" ".join(current + [p]).split()) > max_words:
            chunks.append("\n\n".join(current))
            current = []
        current.append(p)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def fact_drift(original: str, rewritten: str) -> list[str]:
    """Numbers / technical names that vanished or appeared during the rewrite (deterministic check)."""
    warnings = []
    before, after = set(_NUMBER.findall(original)), set(_NUMBER.findall(rewritten))
    for n in sorted(before - after):
        warnings.append(f"Number “{n}” from the original is missing — check the rewrite")
    for n in sorted(after - before):
        warnings.append(f"Number “{n}” was added — not in the original")
    terms_before = set(_TERM.findall(original))
    lowered = rewritten.lower()
    for t in sorted(terms_before):
        if t.lower() not in lowered:
            warnings.append(f"Term “{t}” from the original is missing")
    return warnings


def humanize(text: str, *, keep_letter_frame: bool = True, setup: LLMSetup | None = None,
             session: Session | None = None, job_id: int | None = None) -> HumanizeResult:
    setup = setup or get_llm()
    target = setup.target(Role.HUMANIZER)
    if target is None or not hasattr(target.provider, "complete_raw"):
        raise HumanizerUnavailable("Humanizer model not configured (LOCAL_LLM_ENABLED + LOCAL_HUMANIZER_ENABLED)")
    head, body, tail = split_letter_frame(text) if keep_letter_frame else ("", (text or "").strip(), "")
    if not body:
        raise ValueError("Nothing to rewrite")

    result = HumanizeResult(text="", model=target.model, chunks=0)
    started = time.monotonic()
    parts = []
    guard = GPU_LOCK if target.is_local else contextlib.nullcontext()
    chunks = chunk_paragraphs(body)
    with guard:  # all chunks in one go: no model swaps in between
        for i, chunk in enumerate(chunks, 1):
            tasks.progress(f"Humanizing part {i}/{len(chunks)} ({target.model.split('/')[-1]})")
            resp = target.provider.complete_raw(model=target.model, prompt=build_prompt(chunk),
                                                max_tokens=target.max_tokens, options=target.options)
            out = resp.text.strip()
            result.input_tokens += resp.input_tokens
            result.output_tokens += resp.output_tokens
            src, dst = text_language(chunk), text_language(out)
            if src and dst and src != dst:
                result.warnings.append(f"A paragraph came back in {'English' if dst == 'en' else 'German'} — kept the original")
                result.fallbacks += 1
                out = chunk
            parts.append(out)
            result.chunks += 1
    new_body = "\n\n".join(parts)
    result.warnings += fact_drift(body, new_body)
    result.text = "\n\n".join(p for p in (head, new_body, tail) if p)
    result.duration_ms = int((time.monotonic() - started) * 1000)

    if session is not None:  # usage tracking (local → $0)
        session.add(JobAnalysis(
            job_id=job_id, kind="humanize", provider=target.provider.name, model=target.model,
            prompt_version=PROMPT_VERSION, input_hash=hashlib.sha256(body.encode()).hexdigest(),
            result={"chunks": result.chunks, "fallbacks": result.fallbacks, "warnings": len(result.warnings)},
            input_tokens=result.input_tokens, output_tokens=result.output_tokens, cost_usd=0.0,
            duration_ms=result.duration_ms,
        ))
        session.flush()
    return result
