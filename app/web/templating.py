"""Jinja2 setup and template helpers. Autoescape is on: scraped text is always rendered as text."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

from fastapi.templating import Jinja2Templates

from app.enums import ACTIONS, RELEVANCE_REASONS, JobStatus
from app.models import utcnow
from app.services.links import job_links

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
templates = Jinja2Templates(directory=WEB_DIR / "templates")


def ago(dt: datetime | None) -> str:
    if not dt:
        return "—"
    secs = (utcnow() - dt).total_seconds()
    if secs < 90:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} h ago"
    days = int(secs // 86400)
    return f"{days} d ago" if days < 60 else dt.strftime("%Y-%m-%d")


def score_class(score: float | None) -> str:
    if score is None:
        return "s-none"
    if score >= 8:
        return "s-high"
    if score >= 6.5:
        return "s-mid"
    return "s-low"


def fmt_dt(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d %H:%M") if dt else "—"


def safe_url(url: str | None) -> str:
    """Only allow http(s) links in rendered pages (blocks javascript: etc. from scraped data)."""
    if url and url.lower().startswith(("http://", "https://")):
        return url
    return "#"


def qs(params: dict, **overrides) -> str:
    merged = {**{k: v for k, v in params.items() if v not in (None, "")}, **overrides}
    return urlencode({k: v for k, v in merged.items() if v not in (None, "")})


_SOURCE_NAMES = {
    "linkedin": "LinkedIn", "indeed": "Indeed", "stepstone": "StepStone", "xing": "XING", "glassdoor": "Glassdoor",
    "greenhouse": "Greenhouse", "lever": "Lever", "personio": "Personio", "smartrecruiters": "SmartRecruiters",
    "workday": "Workday", "generic": "Company site", "manual": "Added manually", "email": "Email alert",
    "arbeitsagentur": "Arbeitsagentur", "bayt": "Bayt", "gulftalent": "GulfTalent", "naukrigulf": "Naukrigulf",
}


def source_label(source: str | None) -> str:
    """'linkedin' → 'LinkedIn', 'email:indeed' → 'Indeed (email)'."""
    if not source:
        return "—"
    base, _, via = source.partition(":")
    name = _SOURCE_NAMES.get(base.lower(), base.replace("_", " ").replace("-", " ").title())
    if via:
        return f"{_SOURCE_NAMES.get(via.lower(), via.title())} (email)" if base.lower() == "email" else f"{name} ({via})"
    return name


def host(url: str | None) -> str:
    """Bare domain for display ('https://jobs.example.com/x' → 'jobs.example.com')."""
    from urllib.parse import urlparse

    try:
        return (urlparse(url or "").hostname or "").removeprefix("www.") or "link"
    except ValueError:
        return "link"


def short_model(model: str | None) -> str:
    """'hf.co/Qwen/Qwen3-14B-GGUF:Q5_K_M' → 'Qwen3-14B'."""
    if not model:
        return "—"
    name = model.rsplit("/", 1)[-1].split(":")[0]
    return name.removesuffix("-GGUF").removeprefix("mistralai_").replace("-Instruct-2503", "")


def compact_num(value) -> str:
    """1729859 → '1.73M', 99808 → '99.8k', 8244 → '8,244'."""
    if value is None or value == "":
        return ""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if abs(n) >= 10_000:
        return f"{n / 1000:.1f}k"
    return f"{n:,.0f}"


def task_result(value, _depth: int = 0) -> str:
    """Background-task results as readable 'key: value' text instead of a Python repr."""
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            if v in (None, "", [], {}) or k in ("checked_at",):
                continue
            parts.append(f"{str(k).replace('_', ' ')}: {task_result(v, _depth + 1)}")
        return " · ".join(parts) if _depth == 0 else "(" + ", ".join(parts) + ")"
    if isinstance(value, list | tuple | set):
        items = list(value)
        return ", ".join(str(i) for i in items) if len(items) <= 3 else f"{len(items)} items"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, datetime):
        return fmt_dt(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


EVIDENCE_LABEL = {
    "direct_professional": "professional",
    "academic": "academic",
    "project": "project",
    "transferable": "transferable",
}

templates.env.filters.update(ago=ago, score_class=score_class, fmt_dt=fmt_dt, safe_url=safe_url,
                             source_label=source_label, host=host, short_model=short_model, task_result=task_result,
                             compact_num=compact_num)
def asset(path: str) -> str:
    """/static URL with a version stamp (file modification time): cached for a year, refreshed when it changes."""
    try:
        return f"/static/{path}?v={int((STATIC_DIR / path).stat().st_mtime)}"
    except OSError:
        return f"/static/{path}"


def auth_enabled() -> bool:
    from app.web.auth import _password

    return _password() is not None


_progress_cache: dict[str, tuple[float, tuple[int, int]]] = {}


def setup_progress() -> tuple[int, int]:
    """(essential setup steps done, total) for the nav badge — recomputed at most every 30 s."""
    import time

    hit = _progress_cache.get("p")
    if hit and time.monotonic() - hit[0] < 30:
        return hit[1]
    try:
        from app.services.onboarding import setup_progress as compute

        value = compute()
    except Exception:  # noqa: BLE001 - the badge must never break a page
        value = (0, 0)
    _progress_cache["p"] = (time.monotonic(), value)
    return value


def is_new(first_seen) -> bool:
    """First seen in the last 24 hours (the "new" badge on job cards)."""
    from datetime import timedelta

    from app.models import utcnow

    return bool(first_seen) and utcnow() - first_seen < timedelta(hours=24)


templates.env.globals.update(
    asset=asset, is_new=is_new, auth_enabled=auth_enabled, setup_progress=setup_progress, qs=qs, job_links=job_links, ACTIONS=ACTIONS, RELEVANCE_REASONS=RELEVANCE_REASONS, ALL_STATUSES=[s.value for s in JobStatus], EVIDENCE_LABEL=EVIDENCE_LABEL
)
