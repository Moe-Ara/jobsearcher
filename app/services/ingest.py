"""Ingestion pipeline for one posting: sanitise → normalise → dedup → (create | merge | update) → hard filter.

LLM analysis is deliberately NOT done here; new jobs are queued (analysis_status=PENDING) and processed
in batches by app.services.analysis so costs stay controlled.
"""

from __future__ import annotations

import hashlib
import re
import logging
import time
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.normalize import (
    canonical_seniority,
    infer_country_from_title,
    canonicalize_url,
    extract_language_requirements,
    extract_years_experience,
    infer_country,
    infer_remote_status,
    normalize_company,
    normalize_title,
)
from app.core.text import clean_line, clean_text
from app.dedup import Candidate, find_duplicate
from app.enums import SOURCE_RANK, AnalysisStatus, FilterStatus, JobStatus
from app.models import Job, JobSource, TargetCompany, utcnow
from app.services.jobs import rebuild_search_text, rescore, run_hard_filters, set_status
from app.services.retention import is_academic_source, is_tombstoned, posted_too_long_ago
from app.sources.base import JobPosting

log = logging.getLogger(__name__)

Outcome = Literal["new", "merged", "updated", "unchanged", "too_old", "purged"]

_CONTENT_FIELDS = ("title", "company", "location", "description", "requirements", "responsibilities")
_COPY_FIELDS = (
    "title", "company", "location", "description", "requirements", "responsibilities", "salary_min", "salary_max",
    "currency", "employment_type", "sector", "career_page_url", "date_posted",
)  # fmt: skip


@dataclass
class IngestResult:
    job: Job | None  # None when skipped by the retention policy
    outcome: Outcome
    source: JobSource | None
    dedup_reason: str = ""


def _hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(p or "") for p in parts).encode()).hexdigest()


def content_hash(job: Job) -> str:
    return _hash(*(getattr(job, f) for f in _CONTENT_FIELDS))


def tidy_location(location: str | None) -> str:
    """'Berlin, , Germany' → 'Berlin, Germany' (empty regions from ATS address fields)."""
    loc = re.sub(r"\s*,(\s*,)+\s*", ", ", location or "")
    return loc.strip(" ,;")


def sanitize_posting(p: JobPosting) -> JobPosting:
    data = p.model_dump()
    for f in ("title", "company", "location", "country", "employment_type", "seniority", "sector", "department", "currency", "remote_status"):
        if data.get(f):
            data[f] = clean_line(data[f], 300)
    for f in ("description", "requirements", "responsibilities"):
        data[f] = clean_text(data.get(f))
    data["location"] = tidy_location(data.get("location"))
    data["title"] = data["title"] or "Untitled position"
    return JobPosting.model_validate(data)


# --------------------------------------------------------------------------- company aliases

_alias_cache: tuple[float, dict[str, int]] = (0.0, {})


def invalidate_alias_cache() -> None:
    global _alias_cache
    _alias_cache = (0.0, {})


def _alias_map(session: Session) -> dict[str, int]:
    """normalised company name/alias → TargetCompany.id (cached briefly)."""
    global _alias_cache
    ts, mapping = _alias_cache
    if time.monotonic() - ts < 60 and mapping:
        return mapping
    mapping = {}
    for tc in session.scalars(select(TargetCompany)):
        for name in [tc.company_name, *(tc.aliases or [])]:
            n = normalize_company(name)
            if n:
                mapping.setdefault(n, tc.id)
    _alias_cache = (time.monotonic(), mapping)
    return mapping


def resolve_target_company(session: Session, company: str) -> TargetCompany | None:
    tc_id = _alias_map(session).get(normalize_company(company))
    return session.get(TargetCompany, tc_id) if tc_id else None


# --------------------------------------------------------------------------- core


def _source_key(p: JobPosting, canonical: str | None) -> str:
    if p.external_job_id:
        return f"{p.source}:{p.external_job_id}"[:600]
    if canonical:
        return f"{p.source}:url:{canonical}"[:600]
    return f"{p.source}:hash:{_hash(normalize_title(p.title), normalize_company(p.company), p.location)}"


def _apply_fields(job: Job, p: JobPosting, overwrite: bool) -> None:
    for f in _COPY_FIELDS:
        new = getattr(p, f)
        if new in (None, ""):
            continue
        if overwrite or getattr(job, f) in (None, ""):
            setattr(job, f, new)
    if overwrite and p.url:
        job.source_url = p.url
        job.canonical_url = canonicalize_url(p.url)
    job.normalized_title = normalize_title(job.title)
    if not job.target_company_id:
        job.normalized_company = normalize_company(job.company)
    _derive(job, p)


def _derive(job: Job, p: JobPosting) -> None:
    """Fill inferred fields only where the source didn't provide them."""
    job.country = infer_country(job.location, p.country or job.country) or job.country or infer_country_from_title(job.title)
    job.remote_status = p.remote_status or infer_remote_status(job.title, job.location, job.description) or job.remote_status
    job.seniority = canonical_seniority(job.title, p.seniority) or job.seniority
    years = extract_years_experience(job.requirements, job.description)
    if years is not None:
        job.years_experience_required = years
    langs = extract_language_requirements(job.requirements, job.description)
    if langs:
        job.language_requirements = langs


def _after_content_change(job: Job, *, is_new: bool) -> None:
    new_hash = content_hash(job)
    changed = new_hash != job.content_hash
    job.content_hash = new_hash
    if is_new or changed:
        if not is_new and job.analysis_status in (AnalysisStatus.DONE, AnalysisStatus.ERROR):
            job.analysis_status = AnalysisStatus.PENDING  # cache key changes → re-analysed once
            job.analysis_attempts = 0
        run_hard_filters(job)
    rescore(job)
    rebuild_search_text(job)


def ingest_posting(
    session: Session,
    posting: JobPosting,
    *,
    target_company: TargetCompany | None = None,
    email_message_id: int | None = None,
    enforce_retention: bool = True,
    employer_from_posting: bool = False,
) -> IngestResult:
    """`employer_from_posting`: the feed (`target_company`) is an aggregator such as an academic job board — it keeps
    the provenance (feed checks, deactivation) but the posting's own employer becomes the job's company."""
    p = sanitize_posting(posting)
    feed = target_company
    if employer_from_posting:
        target_company = None
    canonical = canonicalize_url(p.url)
    key = _source_key(p, canonical)
    now = utcnow()
    rank = SOURCE_RANK.get(p.source_type, 9)
    has_content = bool(p.description or p.requirements or p.responsibilities)

    if target_company is None and p.company:
        target_company = resolve_target_company(session, p.company)

    # 1) known source record → refresh
    js = session.scalars(select(JobSource).where(JobSource.source_key == key)).first()
    if js:
        js.last_seen, js.active = now, True
        job = js.job
        job.last_seen = now
        if not job.active:
            job.active = True
            if job.status == JobStatus.EXPIRED:
                set_status(session, job, JobStatus.NEW, note="posting reappeared")
        if has_content and rank <= SOURCE_RANK.get(job.best_source_type, 9):
            before = job.content_hash
            _apply_fields(job, p, overwrite=True)
            _after_content_change(job, is_new=False)
            return IngestResult(job, "updated" if job.content_hash != before else "unchanged", js)
        return IngestResult(job, "unchanged", js)

    # 2) retention: don't save postings older than the window, nor ones deleted by retention before
    if enforce_retention:
        if is_tombstoned(session, key):
            return IngestResult(None, "purged", None, "deleted earlier by the retention policy")
        if posted_too_long_ago(p.date_posted, academic=is_academic_source(p.source)):
            return IngestResult(None, "too_old", None, f"posted {p.date_posted:%Y-%m-%d}")

    normalized_company = target_company.normalized_name if target_company else normalize_company(p.company)
    candidate = Candidate(normalized_company, normalize_title(p.title), p.title, p.location, p.description)
    dup, reason = find_duplicate(
        session, canonical_url=canonical, source=p.source, external_job_id=p.external_job_id, candidate=candidate
    )

    new_source = JobSource(
        source=p.source, source_type=p.source_type, source_key=key, external_job_id=p.external_job_id,
        url=p.url, canonical_url=canonical, title=p.title, company=p.company, location=p.location,
        raw_data=_jsonable(p.raw), target_company_id=feed.id if feed else None,
        email_message_id=email_message_id, first_seen=now, last_seen=now, dedup_reason=reason or None,
    )

    # 2) duplicate of an existing vacancy → attach provenance, upgrade canonical data if this source is better
    if dup:
        new_source.job = dup
        session.add(new_source)
        dup.last_seen, dup.active = now, True
        better = rank < SOURCE_RANK.get(dup.best_source_type, 9)
        if better:
            dup.best_source_type, dup.source = p.source_type, p.source
            dup.external_job_id = p.external_job_id or dup.external_job_id
        _apply_fields(dup, p, overwrite=better and has_content)
        if target_company and not dup.target_company_id:
            dup.target_company = target_company  # relationship (not just the id) so rescore sees the bonus
            dup.normalized_company = target_company.normalized_name
        _after_content_change(dup, is_new=False)
        log.info("Merged %s posting into job %s (%s)", p.source, dup.id, reason)
        return IngestResult(dup, "merged", new_source, reason)

    # 3) new vacancy
    job = Job(
        external_job_id=p.external_job_id, title=p.title, company=p.company or ((target_company or feed).company_name if (target_company or feed) else ""),
        source=p.source, best_source_type=p.source_type, source_url=p.url, canonical_url=canonical,
        first_seen=now, last_seen=now, active=True, raw_source_data=_jsonable(p.raw),
        target_company=target_company,
        status=JobStatus.NEW, filter_status=FilterStatus.PENDING, analysis_status=AnalysisStatus.PENDING,
        language_requirements=[], filter_reasons=[], score_breakdown={}, track=p.track or "main",
    )
    _apply_fields(job, p, overwrite=True)
    job.normalized_company = normalized_company
    if not job.country and not job.location and target_company and target_company.country:
        job.country = target_company.country
    if target_company and not job.sector:
        job.sector = target_company.sector
    session.add(job)
    new_source.job = job
    session.add(new_source)
    session.flush()
    _after_content_change(job, is_new=True)
    from app.services.submit import inherit_applied

    inherit_applied(session, job)  # the same vacancy again after you applied: never a "new" job
    return IngestResult(job, "new", new_source)


def _jsonable(value):
    """Make adapter raw payloads JSON-safe (datetimes → ISO strings) and bounded."""
    import json

    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):
        return {}
    if len(text) > 200_000:
        return {"truncated": True}
    return json.loads(text)
