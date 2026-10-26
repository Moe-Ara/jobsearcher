"""Sanitising untrusted text (scraped HTML, emails, pasted descriptions)."""

from __future__ import annotations

import html
import re
import unicodedata

from bs4 import BeautifulSoup

# C0/C1 controls (except \n, \t), zero-width and bidi-override characters (used to hide text).
_CONTROL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​-‏‪-‮⁠-⁤⁦-⁩﻿]"
)
_SPACES_RE = re.compile(r"[ \t  - 　]+")
_MANY_NL_RE = re.compile(r"\n{3,}")

_BLOCK_TAGS = [
    "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
    "tr", "table", "section", "article", "header", "footer", "blockquote", "pre", "dt", "dd",
]  # fmt: skip

MAX_TEXT_CHARS = 60_000


def clean_text(value: str | None, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Normalise unicode, strip control/invisible characters, tidy whitespace, cap length."""
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACES_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MANY_NL_RE.sub("\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " …"
    return text


def clean_line(value: str | None, max_chars: int = 500) -> str:
    """Single-line field (title, company, location)."""
    return clean_text(value, max_chars=max_chars).replace("\n", " ").strip()


def html_to_text(value: str | None, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Convert (possibly entity-escaped) HTML to readable plain text. Scripts/styles are dropped."""
    if not value:
        return ""
    raw = str(value)
    if "&lt;" in raw and "<" not in raw:  # e.g. Greenhouse returns escaped HTML
        raw = html.unescape(raw)
    if "<" not in raw:
        return clean_text(html.unescape(raw), max_chars)
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "form", "head", "template"]):
        tag.decompose()
    for tag in soup.find_all("li"):
        tag.insert_before("\n• ")
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_after("\n")
    return clean_text(soup.get_text(""), max_chars)


def truncate(text: str | None, max_chars: int) -> str:
    if not text:
        return ""
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + " …[truncated]"


def strip_accents(value: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c)
    ).replace("ß", "ss")
