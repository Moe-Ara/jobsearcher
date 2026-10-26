"""Retention: keep only recent vacancies (default: posted within the last 14 days).

Age = date posted when the source provides a plausible one, otherwise the date the job was first seen.
Jobs you have worked with are never deleted (saved/priority, in the application pipeline, labeled, with notes or
letters), so application history and calibration labels survive. Deleted feed postings get a tombstone so an
undated posting that is still listed doesn't come back as "new" on the next feed check.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import delete, exists, or_, select
from sqlalchemy.orm import Session

from app.config_files import get_prefs
from app.db import session_scope
from app.enums import JobStatus, RelevanceLabel
from app.models import ApplicationDocument, ApplicationNotes, ApplicationPackage, Job, PurgedPosting, StatusEvent, utcnow

log = logging.getLogger(__name__)

# Statuses that mean "I'm working with this job" — never deleted.
KEEP_STATUSES = {
    JobStatus.REVIEW, JobStatus.PRIORITY, JobStatus.APPLYING, JobStatus.APPLIED, JobStatus.HR_SCREEN,
    JobStatus.INTERVIEW, JobStatus.TECHNICAL, JobStatus.FINAL, JobStatus.OFFER,
}


def cutoff(academic: bool = False) -> datetime:
    r = get_prefs().retention
    days = max(r.max_job_age_days, r.academic_max_age_days) if academic else r.max_job_age_days
    return utcnow() - timedelta(days=days)


def is_academic_source(source: str | None) -> bool:
    """Doctoral / research postings stay open for weeks (application deadlines), so they get a longer window."""
    return (source or "") == "academic"


def _job_is_academic(job: Job) -> bool:
    return is_academic_source(job.source) or any(is_academic_source(src.source) for src in job.sources)


def posted_too_long_ago(date_posted: datetime | None, academic: bool = False) -> bool:
    """For new postings: True if the source says it was posted before the retention window."""
    if date_posted is None or date_posted > utcnow() + timedelta(days=1):  # missing or implausible (future) date
        return False
    return date_posted < cutoff(academic)


def job_reference_date(job: Job) -> datetime:
    if job.date_posted and job.date_posted <= utcnow() + timedelta(days=1):
        return job.date_posted
    return job.first_seen


def is_tombstoned(session: Session, source_key: str) -> bool:
    row = session.scalars(select(PurgedPosting).where(PurgedPosting.source_key == source_key)).first()
    if row is None:
        return False
    row.last_seen_at = utcnow()  # still listed somewhere → keep remembering it
    return True


def _interacted(job: Job, session: Session) -> bool:
    if job.status in KEEP_STATUSES or job.relevance_label != RelevanceLabel.UNRATED or (job.notes or "").strip():
        return True
    has_letters = session.scalar(select(exists().where(ApplicationDocument.job_id == job.id)))
    has_notes = session.scalar(select(exists().where(ApplicationNotes.job_id == job.id)))
    has_package = session.scalar(select(exists().where(ApplicationPackage.job_id == job.id)))
    ever_applied = session.scalar(select(exists().where(StatusEvent.job_id == job.id, StatusEvent.to_status == JobStatus.APPLIED)))
    return bool(has_letters or has_notes or has_package or ever_applied)


def purge_old_jobs() -> dict[str, Any]:
    prefs = get_prefs().retention
    limit = cutoff()
    deleted = kept = 0
    with session_scope() as s:
        # candidates: either date is before the cutoff (refined per job below)
        candidate_ids = list(s.scalars(select(Job.id).where(or_(Job.first_seen < limit, Job.date_posted < limit))))
    # One short transaction per job: a long purge must not hold SQLite's single write lock for minutes while
    # other tasks (mail poll, application agent, letters) wait for it.
    for job_id in candidate_ids:
        with session_scope() as s:
            job = s.get(Job, job_id)
            if job is None or job_reference_date(job) >= (cutoff(academic=True) if _job_is_academic(job) else limit):
                continue
            if prefs.keep_interacted and _interacted(job, s):
                kept += 1
                continue
            for src in job.sources:
                if not s.scalars(select(PurgedPosting.id).where(PurgedPosting.source_key == src.source_key)).first():
                    s.add(PurgedPosting(source_key=src.source_key))
            s.delete(job)  # cascades: sources, analyses, status events (letters cascade at DB level)
            deleted += 1
    with session_scope() as s:
        pruned = s.execute(
            delete(PurgedPosting).where(PurgedPosting.last_seen_at < utcnow() - timedelta(days=prefs.tombstone_days))
        ).rowcount
    log.info("Retention: deleted %d jobs older than %d days (kept %d you worked with), pruned %d tombstones",
             deleted, prefs.max_job_age_days, kept, pruned)
    return {"deleted": deleted, "kept_interacted": kept, "tombstones_pruned": pruned, "max_age_days": prefs.max_job_age_days}
