"""Job-level operations shared by ingestion, analysis and the UI."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.config_files import get_prefs, get_profile
from app.enums import AnalysisStatus, FilterStatus, JobStatus, Recommendation
from app.filters import apply_hard_filters
from app.models import Job, StatusEvent, utcnow
from app.scoring import compute_score


def set_status(session: Session, job: Job, status: JobStatus | str, note: str | None = None) -> bool:
    status = JobStatus(status)
    if job.status == status:
        return False
    session.add(StatusEvent(job_id=job.id, from_status=job.status, to_status=status, note=note))
    job.status = status
    job.status_changed_at = utcnow()
    if status == JobStatus.APPLIED and not job.applied_at:
        job.applied_at = utcnow()
    return True


def screening_as_components(r: dict[str, Any]) -> dict[str, Any]:
    """Map a screening result onto the scoring components (provisional, screening-only score)."""
    rel = r.get("relevance_score", 0)
    level = next((p.strip().capitalize() for p in str(r.get("seniority") or "").replace("-", "/").split("/")
                  if p.strip().capitalize() in {"Intern", "Junior", "Mid", "Senior", "Lead", "Principal", "Executive"}), None)
    return {
        "technical_match": rel, "experience_match": rel, "career_match": rel, "location_match": None,
        "role_family": r.get("role_family"), "sector": r.get("sector"), "seniority": level,
        "critical_gaps": r.get("possible_critical_gaps") or [], "overall_score": rel, "screen_only": True,
    }


def current_analysis_result(job: Job) -> dict[str, Any] | None:
    a = job.current_analysis
    if not a or not a.result:
        return None
    if a.kind == "screen":
        return screening_as_components(a.result)
    return a.result


def rescore(job: Job) -> None:
    prefs = get_prefs()
    if getattr(job, "track", "main") == "side":
        from app.services.side_jobs import apply_side

        apply_side(job, prefs)
        return
    if getattr(job, "track", "main") == "intern":
        from app.services.internships import intern_rescore

        intern_rescore(job, prefs, get_profile().years_professional_experience)
        return
    res = compute_score(
        filter_status=job.filter_status,
        analysis=current_analysis_result(job),
        country=job.country,
        location=job.location,
        remote_status=job.remote_status,
        seniority=job.seniority,
        best_source_type=job.best_source_type,
        target_priority=job.target_company.priority if job.target_company else None,
        prefs=prefs,
        years_required=job.years_experience_required,
        candidate_years=get_profile().years_professional_experience,
    )
    if job.analysis_status == AnalysisStatus.SCREENED and res.score is not None:
        # screened out by the cheap model: always low priority, ranked below every reviewed candidate
        res.score = min(res.score, prefs.scoring.thresholds.get("review", 6.5) - 0.01)
        res.recommendation = Recommendation.LOW_PRIORITY
        res.breakdown["screened_out"] = True
    job.fit_score = res.score
    job.recommendation = res.recommendation
    job.score_breakdown = res.breakdown


def run_hard_filters(job: Job) -> None:
    """(Re)apply rule-based filters. LLM-originated rejections (rule startswith 'llm_') are preserved."""
    from app.services.internships import apply_intern, is_internship

    if getattr(job, "track", "main") in ("main", "side") and is_internship(job.title, get_prefs()):
        job.track = "intern"  # internships / trainee programmes: own page, stricter rules
    if job.track == "intern":
        apply_intern(job, get_prefs())
        return
    if getattr(job, "track", "main") == "side":  # part-time / temporary work: own rules, no LLM analysis
        from app.services.side_jobs import apply_side

        apply_side(job, get_prefs())
        return
    outcome = apply_hard_filters(job, get_prefs())
    llm_reasons = [r for r in (job.filter_reasons or []) if str(r.get("rule", "")).startswith("llm_")]
    reasons = outcome.reasons + llm_reasons
    status = outcome.status
    if any(r["severity"] == "reject" for r in llm_reasons):
        status = FilterStatus.REJECTED
    elif status == FilterStatus.PASSED and llm_reasons:
        status = FilterStatus.FLAGGED
    job.filter_status = status
    job.filter_reasons = reasons
    if status == FilterStatus.REJECTED:
        if job.analysis_status == AnalysisStatus.PENDING:
            job.analysis_status = AnalysisStatus.SKIPPED
    elif job.analysis_status == AnalysisStatus.SKIPPED:
        job.analysis_status = AnalysisStatus.DONE if job.current_analysis_id else AnalysisStatus.PENDING


def llm_flag(job: Job, rule: str, detail: str) -> None:
    """Uncertain LLM finding (e.g. possibly defence): keep the job, mark it for human review."""
    reasons = [r for r in (job.filter_reasons or []) if r.get("rule") != rule]
    reasons.append({"rule": rule, "severity": "flag", "detail": detail})
    job.filter_reasons = reasons
    if job.filter_status != FilterStatus.REJECTED:
        job.filter_status = FilterStatus.FLAGGED


def llm_reject(job: Job, rule: str, detail: str) -> None:
    reasons = [r for r in (job.filter_reasons or []) if r.get("rule") != rule]
    reasons.append({"rule": rule, "severity": "reject", "detail": detail})
    job.filter_reasons = reasons
    job.filter_status = FilterStatus.REJECTED


def rebuild_search_text(job: Job) -> None:
    analysis = current_analysis_result(job) or {}
    skills = [m.get("skill", "") for m in (analysis.get("strong_matches") or []) + (analysis.get("partial_matches") or [])]
    parts = [
        job.title, job.company, job.location, job.country, job.sector, job.role_family, job.seniority,
        job.remote_status, job.employment_type, " ".join(skills), " ".join(analysis.get("critical_gaps") or []),
        job.description, job.requirements, job.responsibilities, job.notes,
    ]
    job.search_text = "\n".join(p for p in parts if p).lower()[:100_000]
