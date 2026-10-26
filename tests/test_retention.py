"""14-day retention: old postings are not saved, old jobs are deleted, jobs you worked with are kept."""

from datetime import timedelta

from app.db import session_scope
from app.enums import JobStatus, SourceType
from app.models import ApplicationDocument, Job, JobSource, PurgedPosting, utcnow
from app.services.calibration import set_relevance
from app.services.ingest import ingest_posting
from app.services.jobs import set_status
from app.services.retention import purge_old_jobs
from app.sources.base import JobPosting


def posting(ext, title="Software Engineer", days_ago=None, source="greenhouse", source_type=SourceType.EMPLOYER):
    return JobPosting(source=source, source_type=source_type, external_job_id=ext, url=f"https://jobs.example/{ext}",
                      title=title, company=f"Company {ext}", location="Berlin",
                      description=f"{title}: Python and Docker. " * 6,
                      date_posted=(utcnow() - timedelta(days=days_ago)) if days_ago is not None else None)


def test_old_postings_are_not_saved_but_recent_and_undated_are():
    with session_scope() as s:
        assert ingest_posting(s, posting("old", days_ago=20)).outcome == "too_old"
        assert ingest_posting(s, posting("recent", days_ago=5)).outcome == "new"
        assert ingest_posting(s, posting("undated")).outcome == "new"
        # adding a job by hand is an explicit choice → allowed even if old
        manual = posting("manual-old", days_ago=40, source="manual", source_type=SourceType.MANUAL)
        assert ingest_posting(s, manual, enforce_retention=False).outcome == "new"
    with session_scope() as s:
        assert sorted(j.external_job_id for j in s.query(Job)) == ["manual-old", "recent", "undated"]


def _old_job(s, ext, *, posted_days=None, first_seen_days=20):
    job = ingest_posting(s, posting(ext, title=f"Engineer {ext}", days_ago=None), enforce_retention=False).job
    job.first_seen = utcnow() - timedelta(days=first_seen_days)
    if posted_days is not None:
        job.date_posted = utcnow() - timedelta(days=posted_days)
    return job


def test_purge_deletes_old_jobs_but_keeps_the_ones_you_worked_with():
    with session_scope() as s:
        _old_job(s, "old-untouched")
        _old_job(s, "old-by-date", posted_days=30, first_seen_days=3)  # seen recently, but posted long ago
        _old_job(s, "recent", first_seen_days=2)
        set_status(s, _old_job(s, "old-priority"), JobStatus.PRIORITY)
        set_relevance(_old_job(s, "old-labeled"), "BAD")
        _old_job(s, "old-notes").notes = "call recruiter"
        applied = _old_job(s, "old-applied-rejected")
        set_status(s, applied, JobStatus.APPLIED)
        set_status(s, applied, JobStatus.REJECTED)  # employer rejection after applying: analytics history
        letter = _old_job(s, "old-letter")
        s.flush()
        s.add(ApplicationDocument(job_id=letter.id, group_id="g", stage="draft", language="de", content="x"))
        skipped = _old_job(s, "old-skipped")
        set_status(s, skipped, JobStatus.SKIPPED)  # you weren't interested → deleted
    result = purge_old_jobs()
    assert result["deleted"] == 3 and result["kept_interacted"] == 5
    with session_scope() as s:
        remaining = sorted(j.external_job_id for j in s.query(Job))
        assert remaining == ["old-applied-rejected", "old-labeled", "old-letter", "old-notes", "old-priority", "recent"]
        assert s.query(JobSource).filter(JobSource.external_job_id == "old-untouched").count() == 0
        assert {p.source_key for p in s.query(PurgedPosting)} == {"greenhouse:old-untouched", "greenhouse:old-by-date", "greenhouse:old-skipped"}


def test_deleted_postings_are_not_re_added_and_tombstones_expire():
    with session_scope() as s:
        _old_job(s, "gone")
    purge_old_jobs()
    with session_scope() as s:
        # still listed in the feed, no date → would look "new" again without the tombstone
        assert ingest_posting(s, posting("gone")).outcome == "purged"
        assert s.query(Job).count() == 0
    with session_scope() as s:
        s.query(PurgedPosting).update({PurgedPosting.last_seen_at: utcnow() - timedelta(days=40)})
    assert purge_old_jobs()["tombstones_pruned"] == 1
