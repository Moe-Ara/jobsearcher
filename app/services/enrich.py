"""Enrichment tool — finds the full posting for jobs that arrived from e-mail alerts with only a snippet.

Indeed blocks automated access to its job pages and StepStone's alert links are tracking redirects (some are personal
auto-login "magic links" that must never be followed). So instead of the board, the job is searched where it is
also published: the employer's own careers page or ATS (Personio, Greenhouse, softgarden, join ...) or another
readable board. A page is used only when it verifiably is the same vacancy: the page title carries the job title's
key words and the page names the company. Then the full description (and the full title, when the e-mail
truncated it) is taken over, the real posting URL is kept as the careers-page link, and the job is re-analysed with
the full text. Every attempt is recorded so a job is searched at most once a week.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified

from app.core.http import assert_public_http_url, get_http
from app.core.normalize import normalize_company
from app.core.text import clean_text
from app.db import session_scope
from app.enums import AnalysisStatus, FilterStatus
from app.models import Job, utcnow
from app.services import tasks, toolbox
from app.services.links import _clean_title, is_tracking_url
from app.sources.jsonld import extract_jsonld_postings, extract_page_fallback, jsonld_to_fields

log = logging.getLogger(__name__)

THIN_CHARS = 400
EMAIL_SOURCES = ("indeed", "stepstone", "linkedin", "xing", "email", "arbeitsagentur")
# never fetched: they block bots, need a login, or are the alert's own (tracking) links
_SKIP_HOSTS = re.compile(r"(indeed\.|stepstone\.|linkedin\.|xing\.|glassdoor\.|kununu\.|monster\.|facebook\.|"
                         r"instagram\.|youtube\.|duckduckgo\.|google\.|bing\.|jobs\.ch|jooble|talent\.com|jobrapido|"
                         r"careerjet|adzuna|neuvoo|wikipedia|jobgether|simplyhired|jobteaser|joinhyra|whatjobs|jobleads)", re.I)
_LOGINISH = re.compile(r"magiclink|/login|signin|/account|token=|session", re.I)
_STOP = {"m/w/d", "w/m/d", "m/f/d", "f/m/d", "gn", "all", "genders", "und", "and", "der", "die", "das", "for", "für",
         "mit", "with", "in", "im", "of", "the", "senior", "junior", "mitarbeiter", "mitarbeiterin"}


def _title_words(title: str) -> list[str]:
    words = re.findall(r"[a-zäöüß0-9+#.]{3,}", _clean_title(title).lower())
    return [w for w in words if w not in _STOP]


def is_thin(job: Job) -> bool:
    return len(job.description or "") < THIN_CHARS


def clean_page_title(title: str) -> str:
    """'Agentic Software Engineer (m/w/d) | YoungCapital.de' → 'Agentic Software Engineer (m/w/d)'."""
    return re.split(r"\s+[|·]\s+", (title or "").strip())[0].strip()


def same_posting(job: Job, page_title: str, page_text: str, url: str) -> bool:
    """The page is this vacancy: most title key words in the page title (or its first lines) and the company named.
    Very short titles ("Software Engineer") match too many pages: those also need the company in the page title or URL."""
    words = _title_words(job.title)
    if not words:
        return False
    company_tokens = [t for t in normalize_company(job.company).split() if len(t) >= 3][:2]
    if len(words) <= 2 and company_tokens:
        strong = f"{page_title} {url}".lower().replace("-", " ")
        if not any(t in strong for t in company_tokens):
            return False
    head = f"{page_title} {page_text[:400]}".lower()
    # a truncated e-mail title ("Senior Software Engineer – (.NET / Az") must match on the words it has
    hit = sum(1 for w in words if w in head)
    if hit / len(words) < 0.6:
        return False
    hay = (page_text + " " + urlsplit(url).netloc).lower()
    return not company_tokens or any(t in hay for t in company_tokens)


@toolbox.tool("find_full_posting", "scout", "Find the full posting of an e-mail-alert job on the employer's site or ATS.")
def find_full_posting(job: Job, max_pages: int = 4) -> dict[str, Any] | None:
    from app.services.research import web_search

    query = f"{_clean_title(job.title)} {job.company}".strip()
    if len(query) < 8:
        return None
    for phrasing in (f"{query} Stellenangebot", f"\"{_clean_title(job.title)}\" {job.company} Karriere"):
        try:
            urls = web_search(phrasing)
        except Exception as exc:  # noqa: BLE001 - search is best effort
            log.info("Search for %r failed: %s", phrasing, exc)
            continue
        found = _first_match(job, urls, max_pages)
        if found:
            return found
    return None


def _first_match(job: Job, urls: list[str], max_pages: int) -> dict[str, Any] | None:
    http = get_http()
    tried = 0
    for url in urls:
        host = urlsplit(url).netloc.lower()
        if not host or _SKIP_HOSTS.search(host) or is_tracking_url(url) or _LOGINISH.search(url):
            continue
        if tried >= max_pages:
            break
        tried += 1
        try:
            assert_public_http_url(url)
            html = http.get_text(url)
        except Exception as exc:  # noqa: BLE001
            log.info("Skipping %s: %s", url, exc)
            continue
        nodes = extract_jsonld_postings(html)
        page = extract_page_fallback(html)
        if nodes:
            fields = jsonld_to_fields(nodes[0])
            text = clean_text(fields.get("description") or "") or page["text"]
            title = fields.get("title") or page["title"]
        else:
            fields, text, title = {}, page["text"], page["title"]
        title = clean_page_title(title or "")
        if len(text) < 600 or not same_posting(job, title, text, url):
            continue
        return {"url": url, "title": title, "description": text[:20000], "fields": fields}
    return None


def apply_enrichment(job: Job, found: dict[str, Any] | None) -> str:
    """Write the result into the job (and record the attempt). Returns 'found' / 'not found'."""
    from app.services.jobs import rebuild_search_text, run_hard_filters

    raw = dict(job.raw_source_data or {})
    raw["enrich"] = {"at": utcnow().isoformat(timespec="seconds"), "result": "found" if found else "not found",
                     "url": (found or {}).get("url")}
    job.raw_source_data = raw
    flag_modified(job, "raw_source_data")
    if not found:
        return "not found"
    raw["enrich"]["snippet"] = (job.description or "")[:4000]  # the e-mail text, to undo a wrong match
    job.description = found["description"]
    fields = found.get("fields") or {}
    for key in ("requirements", "responsibilities"):
        if fields.get(key) and not getattr(job, key):
            setattr(job, key, fields[key])
    if not job.location and fields.get("location"):
        job.location = fields["location"]
    full_title = clean_page_title(found.get("title") or "")
    truncated = (job.raw_source_data or {}).get("truncated_title") or job.title.endswith(("...", "…"))
    if truncated and full_title and _clean_title(full_title).lower().startswith(_clean_title(job.title).lower()[:15]):
        job.title = full_title[:300]  # only when the e-mail cut the title off
    job.career_page_url = found["url"]
    rebuild_search_text(job)
    run_hard_filters(job)
    if job.analysis_status in (AnalysisStatus.DONE, AnalysisStatus.SCREENED) and job.filter_status != FilterStatus.REJECTED:
        job.analysis_status = AnalysisStatus.PENDING  # analysed on a snippet before → analyse the full text
    return "found"


def thin_job_ids(limit: int) -> list[int]:
    cutoff = utcnow() - timedelta(days=7)
    out = []
    with session_scope() as s:
        jobs = s.scalars(select(Job).where(Job.active.is_(True), Job.filter_status != FilterStatus.REJECTED,
                                           Job.source.in_(EMAIL_SOURCES)).order_by(Job.first_seen.desc())).all()
        for job in jobs:
            if not is_thin(job):
                continue
            last = ((job.raw_source_data or {}).get("enrich") or {}).get("at")
            if last and datetime.fromisoformat(last) > cutoff:
                continue
            out.append(job.id)
            if len(out) >= limit:
                break
    return out


def enrich_thin_jobs(limit: int = 25) -> dict[str, Any]:
    """Background task: look up full postings for the newest snippet-only e-mail jobs."""
    counts = {"found": 0, "not found": 0}
    ids = thin_job_ids(limit)
    for i, job_id in enumerate(ids, 1):
        tasks.progress(f"Finding full postings ({i}/{len(ids)})")
        with session_scope() as s:
            job = s.get(Job, job_id)
            if job is None:
                continue
            counts[apply_enrichment(job, find_full_posting(job))] += 1
    log.info("Enrichment: %s of %d snippet-only jobs found in full", counts["found"], len(ids))
    return {"checked": len(ids), **counts}
