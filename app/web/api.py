"""JSON API (for scripts/integrations). Same service layer as the dashboard."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.config_files import get_prefs
from app.db import session_scope
from app.enums import JobStatus
from app.models import Job, TargetCompany
from app.services import tasks
from app.services.analysis import usage_summary
from app.services.application_notes import get_notes, get_strategy
from app.services.humanizer import HumanizerUnavailable, humanize
from app.services.jobs import set_status
from app.services.links import job_links
from app.services.manual import ManualIngestError, ingest_manual
from app.services.queries import JobFilters, dashboard_stats, search_jobs

router = APIRouter(prefix="/api", tags=["api"])


def job_dict(job: Job, full: bool = False) -> dict[str, Any]:
    a = job.current_analysis.result if job.current_analysis and job.current_analysis.result else {}
    out = {
        "id": job.id, "title": job.title, "company": job.company, "location": job.location, "country": job.country,
        "remote_status": job.remote_status, "seniority": job.seniority, "sector": job.sector,
        "role_family": job.role_family, "source": job.source, "canonical_url": job.canonical_url, "open_url": job_links(job)["primary"].url,
        "first_seen": job.first_seen.isoformat(), "active": job.active, "status": job.status,
        "filter_status": job.filter_status, "fit_score": job.fit_score, "recommendation": job.recommendation,
        "summary": a.get("summary"),
        "strong_matches": [m.get("skill") for m in a.get("strong_matches", [])],
        "critical_gaps": a.get("critical_gaps", []),
    }
    if full:
        out.update({
            "description": job.description, "requirements": job.requirements, "responsibilities": job.responsibilities,
            "analysis": a, "score_breakdown": job.score_breakdown, "filter_reasons": job.filter_reasons,
            "sources": [{"source": s.source, "type": s.source_type, "url": s.url, "first_seen": s.first_seen.isoformat(), "active": s.active} for s in job.sources],
            "notes": job.notes, "cv_version": job.cv_version,
        })
    return out


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/stats")
def stats(since_days: int | None = None) -> dict[str, Any]:
    with session_scope() as s:
        return {**dashboard_stats(s, since_days or get_prefs().dashboard.default_since_days), "llm_usage": usage_summary(s)}


@router.get("/jobs")
def list_jobs(request: Request) -> dict[str, Any]:
    f = JobFilters.from_params(dict(request.query_params), None)
    with session_scope() as s:
        jobs, total = search_jobs(s, f, 100)
        return {"total": total, "page": f.page, "jobs": [job_dict(j) for j in jobs]}


@router.get("/jobs/{job_id}")
def get_job(job_id: int) -> dict[str, Any]:
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        out = job_dict(job, full=True)
        notes = get_notes(s, job_id)
        # For later comparison of applications with/without personal input vs. responses — no conclusions drawn here.
        out["cover_letter_strategy"] = get_strategy(s, job_id).value
        out["personal_notes"] = None if notes.is_empty else notes.as_dict()
        return out


class StatusIn(BaseModel):
    status: JobStatus
    note: str | None = None


@router.post("/jobs/{job_id}/status")
def update_status(job_id: int, body: StatusIn) -> dict[str, Any]:
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        set_status(s, job, body.status, note=body.note)  # workflow state only — never triggers generation
        return {"id": job.id, "status": job.status}


class ManualIn(BaseModel):
    url: str = ""
    title: str = ""
    company: str = ""
    location: str = ""
    description: str = ""


@router.post("/ingest/manual")
def manual(body: ManualIn) -> dict[str, Any]:
    try:
        res = ingest_manual(**body.model_dump())
    except ManualIngestError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"job_id": res.job_id, "outcome": res.outcome, "message": res.message}


@router.get("/companies")
def companies() -> list[dict[str, Any]]:
    with session_scope() as s:
        return [
            {"id": c.id, "company_name": c.company_name, "country": c.country, "priority": c.priority,
             "ats_type": c.ats_type, "ats_identifier": c.ats_identifier, "active": c.active,
             "last_checked": c.last_checked.isoformat() if c.last_checked else None, "last_status": c.last_status}
            for c in s.scalars(select(TargetCompany).order_by(TargetCompany.company_name))
        ]


_TASKS = {
    "companies": tasks.task_check_due_companies,
    "companies_all": tasks.task_check_all_companies,
    "email": tasks.task_poll_email,
    "analysis": tasks.task_analyze,
}


@router.post("/tasks/{name}")
def run_task(name: str) -> dict[str, Any]:
    if name not in _TASKS:
        raise HTTPException(404, f"Unknown task; choose from {sorted(_TASKS)}")
    lock_name = "companies" if name.startswith("companies") else name
    return {"started": tasks.run_in_background(lock_name, _TASKS[name])}


@router.get("/tasks")
def task_state() -> dict[str, Any]:
    return {k: {kk: str(vv) for kk, vv in v.items()} for k, v in tasks.task_status().items()}


class HumanizeIn(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    keep_letter_frame: bool = True


@router.post("/humanize")
def humanize_text(body: HumanizeIn) -> dict[str, Any]:
    """Synchronous standalone rewrite (local humanizer model). Waits for the GPU if a batch is running."""
    try:
        with session_scope() as s:
            r = humanize(body.text, keep_letter_frame=body.keep_letter_frame, session=s)
    except HumanizerUnavailable as e:
        raise HTTPException(503, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"text": r.text, "model": r.model, "chunks": r.chunks, "warnings": r.warnings,
            "language_fallbacks": r.fallbacks, "duration_ms": r.duration_ms}
