"""Deduplication: is this incoming posting a vacancy we already know (from another source)?

Order of evidence:
  1. same source + same external id          → same JobSource (handled by the ingest service)
  2. same canonical URL                      → same job
  3. fuzzy: company ≈, title ≈ (synonym-normalised), location compatible, description ≈ when available
Guards: different seniority levels, or two different ids from the same source, are never merged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from rapidfuzz import fuzz
from sqlalchemy import or_, select
from sqlalchemy.orm import Session, selectinload

from app.core.normalize import infer_seniority
from app.models import Job, JobSource, utcnow

DEDUP_WINDOW_DAYS = 90


@dataclass
class Candidate:
    normalized_company: str
    normalized_title: str
    title: str
    location: str
    description: str


@dataclass
class MatchResult:
    same: bool
    score: float
    reason: str


def _loc_tokens(location: str) -> set[str]:
    words = re.sub(r"[^a-zÀ-ɏ]+", " ", location.lower()).split()
    ignore = {"germany", "deutschland", "remote", "hybrid", "onsite", "de", "and", "or", "uae", "ksa", "saudi", "arabia", "united", "arab", "emirates"}
    return {w for w in words if len(w) > 2 and w not in ignore}


def locations_compatible(a: str, b: str) -> bool:
    ta, tb = _loc_tokens(a or ""), _loc_tokens(b or "")
    if not ta or not tb:
        return True  # unknown on one side → don't block
    if ta & tb:
        return True
    return fuzz.token_set_ratio(" ".join(ta), " ".join(tb)) >= 85


def same_vacancy(a: Candidate, b: Candidate) -> MatchResult:
    company_sim = fuzz.token_set_ratio(a.normalized_company, b.normalized_company)
    if company_sim < 90:
        return MatchResult(False, 0, "company differs")
    sa, sb = infer_seniority(a.title), infer_seniority(b.title)
    if sa and sb and sa != sb:
        return MatchResult(False, 0, f"seniority differs ({sa}/{sb})")
    if not locations_compatible(a.location, b.location):
        return MatchResult(False, 0, "location differs")

    title_sim = max(
        fuzz.token_sort_ratio(a.normalized_title, b.normalized_title),
        0.95 * fuzz.token_set_ratio(a.normalized_title, b.normalized_title),
    )
    da, db = (a.description or "")[:5000], (b.description or "")[:5000]
    if len(da) >= 300 and len(db) >= 300:
        desc_sim = fuzz.token_set_ratio(da, db)
        if (title_sim >= 80 and desc_sim >= 75) or (title_sim >= 92 and desc_sim >= 55):
            return MatchResult(True, (title_sim + desc_sim) / 2, f"title {title_sim:.0f}%, description {desc_sim:.0f}%")
        return MatchResult(False, 0, f"title {title_sim:.0f}%, description {desc_sim:.0f}%")
    if title_sim >= 90:
        return MatchResult(True, title_sim, f"company+title {title_sim:.0f}% + location")
    return MatchResult(False, 0, f"title {title_sim:.0f}%")


def _company_candidates_query(normalized_company: str, since: datetime):
    tokens = sorted((t for t in normalized_company.split() if len(t) >= 3), key=len, reverse=True)
    conds = [Job.normalized_company == normalized_company]
    if tokens:
        conds.append(Job.normalized_company.contains(tokens[0]))
    return (
        select(Job)
        .where(or_(*conds), Job.last_seen >= since)
        .options(selectinload(Job.sources))
        .limit(300)
    )


def find_duplicate(
    session: Session,
    *,
    canonical_url: str | None,
    source: str,
    external_job_id: str | None,
    candidate: Candidate,
) -> tuple[Job | None, str]:
    if canonical_url:
        js = session.scalars(select(JobSource).where(JobSource.canonical_url == canonical_url).limit(1)).first()
        if js:
            return js.job, "same canonical URL"
        job = session.scalars(select(Job).where(Job.canonical_url == canonical_url).limit(1)).first()
        if job:
            return job, "same canonical URL"

    if not candidate.normalized_company or not candidate.normalized_title:
        return None, ""
    since = utcnow() - timedelta(days=DEDUP_WINDOW_DAYS)
    best: tuple[float, Job | None, str] = (0.0, None, "")
    for job in session.scalars(_company_candidates_query(candidate.normalized_company, since)):
        if external_job_id and any(
            s.source == source and s.external_job_id and s.external_job_id != external_job_id for s in job.sources
        ):
            continue  # two distinct postings from the same source are distinct vacancies
        other = Candidate(job.normalized_company, job.normalized_title, job.title, job.location or "", job.description or "")
        res = same_vacancy(candidate, other)
        if res.same and res.score > best[0]:
            best = (res.score, job, res.reason)
    return best[1], best[2]
