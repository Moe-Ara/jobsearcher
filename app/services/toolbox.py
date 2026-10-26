"""Agent toolbox — the deterministic skills the agents call instead of making a language model do the work.

Every tool is a plain, fast, testable function. The agents (scout, research, application, reviewer) call them in a
fixed order; there is no LLM tool-picking loop (that would cost minutes per application on local models). Each call
is recorded (name, seconds, ok, note) in the running agent's tool log, shown on the application package page.

    render_cv          LaTeX CV (your design) -> PDF; built-in renderer as fallback
    render_letter      DIN 5008 letter (KOMA-Script) -> PDF; built-in renderer as fallback
    fit_cv_to_page     relevance-weighted cutting until the compiled CV fits one page
    latex_compile      Tectonic / xelatex / pdflatex in a sandboxed temp dir
    verify_pdf         page count, text layer, garbled glyphs
    ats_extract        the text an applicant tracking system parses from a PDF
    merge_pdfs         complete application PDF (letter + CV + certificates)
    company_research   company website + Wikipedia -> sourced facts (see app.services.research)
    fact_check         claims vs. your profile, incremental per sentence (see app.services.letters)
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import io
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    agent: str
    description: str


REGISTRY: dict[str, ToolSpec] = {}
_state = threading.local()


def tool(name: str, agent: str, description: str) -> Callable:
    """Register a function as an agent tool and record each call in the active tool log."""
    REGISTRY[name] = ToolSpec(name, agent, description)

    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.monotonic()
            entry: dict[str, Any] = {"tool": name, "ok": True}
            try:
                result = fn(*args, **kwargs)
                if isinstance(result, ToolResult):
                    entry["ok"], entry["note"] = result.ok, result.note
                return result
            except Exception as exc:
                entry["ok"], entry["note"] = False, str(exc)[:200]
                raise
            finally:
                entry["seconds"] = round(time.monotonic() - started, 2)
                calls = getattr(_state, "calls", None)
                if calls is not None:
                    calls.append(entry)
        return wrapper
    return decorate


@contextlib.contextmanager
def recording() -> Iterator[list[dict[str, Any]]]:
    """Collect the tool calls made in this thread (one agent run)."""
    previous = getattr(_state, "calls", None)
    _state.calls = []
    try:
        yield _state.calls
    finally:
        _state.calls = previous


def note(entry: dict[str, Any]) -> None:
    """Add a non-tool step (e.g. an LLM stage with its duration) to the active log."""
    calls = getattr(_state, "calls", None)
    if calls is not None:
        calls.append(entry)


@dataclass
class ToolResult:
    ok: bool
    value: Any = None
    note: str = ""


# --------------------------------------------------------------------------- PDF / LaTeX


def _key(obj: Any) -> str:
    clean = {k: v for k, v in obj.items() if k not in ("meta", "fit_level")} if isinstance(obj, dict) else obj
    return hashlib.sha256(json.dumps(clean, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


_pdf_cache: dict[str, tuple[bytes, str]] = {}
_cache_lock = threading.Lock()


def _cached(key: str, make: Callable[[], tuple[bytes, str]]) -> tuple[bytes, str]:
    with _cache_lock:
        hit = _pdf_cache.get(key)
    if hit:
        return hit
    value = make()
    with _cache_lock:
        if len(_pdf_cache) > 64:
            _pdf_cache.clear()
        _pdf_cache[key] = value
    return value


@tool("latex_compile", "application", "Compile a LaTeX document to PDF (Tectonic in Docker, else xelatex/pdflatex).")
def latex_compile(source: str) -> ToolResult:
    from app.services import latex

    res = latex.compile_tex(source)
    return ToolResult(res.ok, res, f"{res.engine} {res.seconds:.1f}s" if res.ok else res.log_tail[-200:])


@tool("render_cv", "application", "Typeset the tailored CV as PDF in your CV design (built-in renderer as fallback).")
def render_cv(cv: dict[str, Any]) -> tuple[bytes, str]:
    """(pdf, engine). Cached by content, so the ATS check and downloads reuse the same compile."""
    from app.services import latex
    from app.services.cv_render import render_pdf

    def make() -> tuple[bytes, str]:
        if latex.available():
            res = latex.compile_tex(latex.cv_tex(cv))
            if res.ok:
                return res.pdf, f"latex/{res.engine}"
            log.warning("LaTeX CV failed, using the built-in renderer: %s", res.log_tail[-300:])
        return render_pdf(cv), "builtin"

    return _cached("cv:" + _key(cv), make)


@tool("render_letter", "application", "Typeset the cover letter as a DIN 5008 business letter PDF (KOMA-Script).")
def render_letter(letter: str, subject: str | None, cv: dict[str, Any], recipient: list[str], date_line: str,
                  attachments: list[str] | None, lang: str) -> tuple[bytes, str]:
    from app.services import latex
    from app.services.cv_render import render_letter_pdf

    def make() -> tuple[bytes, str]:
        if latex.available():
            best = None
            for level in range(len(latex.LETTER_FIT)):  # first layout that fits one page, checked on the real PDF
                res = latex.compile_tex(latex.letter_tex(letter, subject, cv, recipient, date_line, attachments, lang,
                                                         fit=level))
                if not res.ok:
                    break
                best = (res.pdf, f"latex/{res.engine}" + (f" fit={level}" if level else ""))
                if page_count(res.pdf) <= 1:
                    return best
            if best:
                note({"tool": "render_letter", "ok": True, "note": "letter is longer than one page at the tightest layout"})
                return best
            log.warning("LaTeX letter failed, using the built-in renderer: %s", res.log_tail[-300:])
        return render_letter_pdf(letter, subject, cv, recipient, date_line, attachments,
                                 "Anlagen" if lang == "de" else "Enclosures"), "builtin"

    key = _key({"l": letter, "s": subject, "c": cv.get("contact"), "n": cv.get("name"), "r": recipient,
                "d": date_line, "a": attachments, "lang": lang})
    return _cached("letter:" + key, make)


def page_count(pdf: bytes) -> int:
    from pypdf import PdfReader

    return len(PdfReader(io.BytesIO(pdf)).pages)


@tool("fit_cv_to_page", "application", "Cut the least relevant optional CV content until the compiled CV fits one page.")
def fit_cv_to_page(build: Callable[[int], dict[str, Any]], levels: int, max_pages: int = 1) -> dict[str, Any]:
    """`build(level)` -> cv. Estimates the level with the fast built-in renderer, then confirms with the real
    (LaTeX) PDF and adjusts by one level either way — usually 1-2 compiles instead of up to `levels`."""
    from app.services.cv_render import page_count as builtin_pages

    level = next((lv for lv in range(levels) if builtin_pages(build(lv)) <= max_pages), levels - 1)
    cv = build(level)
    while page_count(render_cv(cv)[0]) > max_pages and level < levels - 1:
        level += 1
        cv = build(level)
    if level > 0:  # the real typesetting may be tighter than the estimate: keep more content when it fits
        fuller = build(level - 1)
        if page_count(render_cv(fuller)[0]) <= max_pages:
            cv, level = fuller, level - 1
    cv["fit_level"] = level
    return cv


@tool("verify_pdf", "reviewer", "Check a PDF: page count, extractable text, no garbled glyphs.")
def verify_pdf(pdf: bytes, max_pages: int) -> ToolResult:
    import re

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf))
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    problems = []
    if len(reader.pages) > max_pages:
        problems.append(f"{len(reader.pages)} pages (max {max_pages})")
    if len(text.strip()) < 200:
        problems.append("almost no extractable text")
    if re.search(r"\(cid:\d+\)|�", text):
        problems.append("garbled glyphs in the text layer")
    return ToolResult(not problems, {"pages": len(reader.pages), "chars": len(text)}, "; ".join(problems))


@tool("ats_extract", "reviewer", "Extract the text layer of a PDF the way an applicant tracking system parses it.")
def ats_extract(pdf: bytes) -> tuple[str, int]:
    from app.services.ats import pdf_text

    return pdf_text(pdf)


@tool("merge_pdfs", "application", "Merge letter, CV and certificates into one application PDF.")
def merge_pdfs(parts: list[bytes], documents: list[tuple[Any, Any]]) -> bytes:
    from app.services.application_agent import bundle_pdf

    return bundle_pdf(parts, documents)


REGISTRY.setdefault("company_research", ToolSpec(
    "company_research", "research", "Read the company's website and Wikipedia; extract sourced facts and letter hooks."))
REGISTRY.setdefault("fact_check", ToolSpec(
    "fact_check", "application", "Check every claim about you against profile.yaml — only new or changed sentences."))
REGISTRY.setdefault("keyword_extract", ToolSpec(
    "keyword_extract", "reviewer", "Reduce the posting to the keywords an ATS screens for (required / preferred)."))
