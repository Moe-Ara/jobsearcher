"""Job analysis pipeline: hard filters → cheap screening → (promising only) deep analysis → deterministic scoring.

Batch runs are two-phase so each model is loaded once per run (important with one local GPU):
  phase 1  screen every pending job with the screen model   (local: Qwen3-8B)
  phase 2  deep-analyse the promising ones with the analysis model (local: Qwen3-14B)
The German editor / writer models are never used here.

Cost controls:
- only jobs that passed the hard filters are analysed;
- every call is keyed by sha256(kind, prompt version, provider, model, profile, exact prompt) and the result is
  reused whenever the same input is seen again (unchanged jobs are never re-sent);
- only relevant, truncated text fields are sent;
- deep analysis only runs above a minimum screening relevance;
- a daily USD budget stops paid calls (local calls are free).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from rapidfuzz import fuzz
from sqlalchemy import case, false, func, or_, select
from sqlalchemy.orm import Session

from app.config_files import CandidateProfile, get_prefs, get_profile
from app.db import session_scope
from app.enums import AnalysisStatus, FilterStatus
from app.llm import heuristic
from app.llm.base import LLMError, LLMUnavailable
from app.llm.factory import LLMSetup, Role, get_llm
from app.llm.prompts import (
    EXTRACT_PROMPT_VERSION,
    EXTRACT_SYSTEM,
    FIT_PROMPT_VERSION,
    SCREEN_PROMPT_VERSION,
    extract_user_prompt,
    fit_system_prompt,
    fit_user_prompt,
    screen_system_prompt,
    screen_user_prompt,
)
from app.llm.schemas import ExtractedPosting, JobFitAnalysis, JobScreening
from app.models import Job, JobAnalysis, TargetCompany
from app.services.jobs import llm_flag, llm_reject, rebuild_search_text, rescore
from app.services.llm_calls import BudgetExceeded, input_hash, llm_call, spent_today

log = logging.getLogger(__name__)

__all__ = ["BudgetExceeded", "analyze_job", "analyze_pending", "spent_today", "verify_and_clean", "usage_summary"]

_SENIORITIES = {"Intern", "Junior", "Mid", "Senior", "Lead", "Principal", "Executive"}
# Below this model confidence a "defence-related" verdict only flags the job for human review.
DEFENCE_REJECT_CONFIDENCE = 0.7


# --------------------------------------------------------------------------- verification


def _clamp(v: Any, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return lo


def _supported_by_profile(skill: str, evidence: str, profile: CandidateProfile) -> bool:
    corpus = profile.evidence_corpus()
    evidence, skill = (evidence or "").lower().strip(), (skill or "").lower().strip()
    if evidence and (evidence in corpus or fuzz.partial_ratio(evidence, corpus) >= 88):
        return True
    return any(fuzz.token_set_ratio(skill, s.lower()) >= 88 for s, _ in profile.skills.all())


def _actual_level(match: dict[str, Any], profile: CandidateProfile) -> str | None:
    """The profile's evidence level for a claimed-professional match, or None if it is genuinely professional/unknown."""
    professional = profile.professional_corpus()
    for text in (match.get("profile_evidence", ""), match.get("skill", "")):
        t = (text or "").strip().lower()
        if t and (t in professional or fuzz.partial_ratio(t, professional) >= 92):
            return None
    best, best_score = None, 0
    for skill, levels in profile.skills.levels().items():
        score = max(fuzz.ratio(skill, (match.get("profile_evidence") or "").lower()), fuzz.ratio(skill, (match.get("skill") or "").lower()))
        if score > best_score:
            best, best_score = levels, score
    if best and best_score >= 88 and "direct_professional" not in best:
        return "academic" if "academic" in best else ("project" if "project" in best else sorted(best)[0])
    return None


_DEGREE_REQ = re.compile(r"studium|degree|bachelor|master|abschluss|hochschul|universit|diplom|b\.?\s?sc|m\.?\s?sc", re.I)
_DOCTORATE = re.compile(r"promotion|phd|ph\.d|doktor|doctorate", re.I)
_FIELD_SYNONYMS = {
    "informatik": "computer science", "computer science": "computer science", "wirtschaftsinformatik": "computer science",
    "software": "computer science", "intelligent": "intelligent systems", "ki": "intelligent systems",
    "künstliche intelligenz": "intelligent systems", "artificial intelligence": "intelligent systems",
}
_COMPARABLE = re.compile(r"vergleichbar|ähnlich|comparable|related|similar|equivalent|or a? ?similar|mint|stem|technisch|technical", re.I)


def degree_gap_satisfied(gap: str, profile: CandidateProfile) -> bool:
    """A 'missing degree' gap the education entries already satisfy (e.g. 'Studium der Informatik' vs B.Sc. CS)."""
    if not _DEGREE_REQ.search(gap) or _DOCTORATE.search(gap):
        return False
    degrees = " ".join(f"{e.degree} {e.focus} {(e.model_extra or {}).get('field', '')}" for e in profile.education).lower()
    if not degrees.strip():
        return False
    held = {canon for word, canon in _FIELD_SYNONYMS.items() if word in degrees}
    wanted = {canon for word, canon in _FIELD_SYNONYMS.items() if re.search(rf"\b{re.escape(word)}", gap, re.I)}
    if wanted & held:
        return True
    return bool(_COMPARABLE.search(gap)) and bool(held)


def promote_direct_matches(result: dict[str, Any], profile: CandidateProfile) -> list[str]:
    """A "partial" match that names one of your skills exactly, with professional evidence, is a strong match
    (the model sometimes files "Java" under partial while citing your Java experience). Returns the promoted skills."""
    own = {name.strip().lower() for name, _ in profile.skills.all()}
    promoted, partial = [], []
    for m in result.get("partial_matches") or []:
        skill = (m.get("skill") or "").strip()
        own_skill = m.get("evidence") == "direct_professional" and skill.lower() in own
        if own_skill or degree_gap_satisfied(skill, profile):  # "Studium der Informatik oder vergleichbar" + your M.Sc.
            result.setdefault("strong_matches", []).append(m)
            promoted.append(skill)
        else:
            partial.append(m)
    result["partial_matches"] = partial
    if promoted:
        result["promoted_matches"] = promoted
    return promoted


def _norm_req(text: str) -> str:
    return re.sub(r"[^a-z0-9äöüß]+", " ", (text or "").lower()).strip()


def _same_requirement(skill: str, gap: str) -> bool:
    a, b = _norm_req(skill), _norm_req(gap)
    return len(a) >= 12 and len(b) >= 12 and (b.startswith(a) or a.startswith(b))


def reconcile_matches_and_gaps(result: dict[str, Any]) -> list[str]:
    """The model sometimes lists one requirement as a strong match AND as a gap ("… while the candidate has quantized
    models, not all techniques"). Its own caveat wins: the match is partial, the gap stays (gaps are never dropped —
    that would inflate the score). Returns the demoted requirements."""
    gaps = [g for key in ("critical_gaps", "nice_to_have_gaps") for g in result.get(key) or []]
    if not gaps:
        return []
    demoted, strong = [], []
    for m in result.get("strong_matches") or []:
        if any(_same_requirement(m.get("skill", ""), g) for g in gaps):
            result.setdefault("partial_matches", []).append(m)
            demoted.append(m.get("skill", ""))
        else:
            strong.append(m)
    result["strong_matches"] = strong
    if demoted:
        result["demoted_matches"] = demoted
    return demoted


def verify_and_clean(result: dict[str, Any], profile: CandidateProfile) -> dict[str, Any]:
    """Drop skill claims that cannot be traced to the profile; clamp numbers."""
    unverified: list[str] = []
    corrected_levels: list[str] = []
    for key in ("strong_matches", "partial_matches"):
        kept = []
        for m in result.get(key) or []:
            if not _supported_by_profile(m.get("skill", ""), m.get("profile_evidence", ""), profile):
                unverified.append(m.get("skill", ""))
                continue
            if m.get("evidence") == "direct_professional":
                actual = _actual_level(m, profile)
                if actual and actual != "direct_professional":
                    m = {**m, "evidence": actual}
                    corrected_levels.append(f"{m.get('skill')} → {actual}")
            kept.append(m)
        result[key] = kept
    result["unverified_claims"] = [u for u in unverified if u]
    result["evidence_level_corrections"] = corrected_levels
    promote_direct_matches(result, profile)
    resolved = []
    for key in ("critical_gaps", "nice_to_have_gaps"):
        kept_gaps = []
        for gap in result.get(key) or []:
            (resolved if degree_gap_satisfied(gap, profile) else kept_gaps).append(gap)
        result[key] = kept_gaps
    result["resolved_gaps"] = resolved  # e.g. degree "gaps" the education entries already satisfy
    reconcile_matches_and_gaps(result)
    for key in ("technical_match", "experience_match", "career_match", "location_match"):
        result[key] = int(round(_clamp(result.get(key), 0, 10)))
    result["overall_score"] = round(_clamp(result.get("overall_score"), 0, 10), 1)
    result["confidence"] = round(_clamp(result.get("confidence"), 0, 1), 2)
    if result["unverified_claims"]:
        result["confidence"] = round(result["confidence"] * 0.85, 2)
    return result


def clean_screening(result: dict[str, Any], profile: CandidateProfile) -> dict[str, Any]:
    """The screener may only claim overlap that exists in profile.yaml."""
    kept, dropped = [], []
    for item in result.get("candidate_overlap") or []:
        (kept if _supported_by_profile(item, item, profile) else dropped).append(item)
    result["candidate_overlap"] = kept
    result["unverified_overlap"] = dropped
    result["relevance_score"] = int(round(_clamp(result.get("relevance_score"), 0, 10)))
    result["confidence"] = round(_clamp(result.get("confidence"), 0, 1), 2)
    return result


def canonical_level(value: str | None) -> str | None:
    """'Junior/Mid' → 'Junior' (first recognised level)."""
    for part in (value or "").replace("-", "/").split("/"):
        if part.strip().capitalize() in _SENIORITIES:
            return part.strip().capitalize()
    return None


# --------------------------------------------------------------------------- stages


def _apply_fit(job: Job, result: dict[str, Any], rec: JobAnalysis) -> None:
    job.current_analysis = rec
    job.current_analysis_id = rec.id
    job.analysis_status = AnalysisStatus.DONE
    job.role_family = result.get("role_family") or job.role_family
    job.sector = job.sector or result.get("sector")
    if result.get("seniority") in _SENIORITIES and job.seniority not in _SENIORITIES:
        job.seniority = result["seniority"]
    if job.years_experience_required is None and result.get("years_experience_required"):
        job.years_experience_required = float(result["years_experience_required"])
    if not job.language_requirements and result.get("language_requirements"):
        job.language_requirements = result["language_requirements"]
    if (not job.remote_status or job.remote_status == "unknown") and result.get("remote_status") not in (None, "unknown"):
        job.remote_status = result["remote_status"]
    if result.get("defence_related"):
        reason = result.get("defence_reason") or "see analysis"
        if float(result.get("confidence") or 0) >= DEFENCE_REJECT_CONFIDENCE:
            llm_reject(job, "llm_defence", f"Defence-related (analysis): {reason}")
        else:  # uncertain → human decides, no invented certainty
            llm_flag(job, "llm_defence_check", f"Possibly defence-related (analysis, low confidence): {reason}")


def _fail(job: Job, exc: LLMError) -> str:
    if isinstance(exc, LLMUnavailable):
        raise exc  # infrastructure problem: don't count it against the job, stop the batch
    job.analysis_attempts = (job.analysis_attempts or 0) + 1
    job.analysis_status = AnalysisStatus.ERROR
    log.warning("Analysis failed for job %s: %s", job.id, exc)
    return f"error: {exc}"


def screen_job(session: Session, job: Job, setup: LLMSetup) -> tuple[str, dict[str, Any] | None]:
    """Phase 1. Returns (outcome, screening). outcome 'promising' means: continue to deep analysis."""
    prefs, profile = get_prefs(), get_profile()
    target = setup.target(Role.SCREEN)
    if target is None or not prefs.llm.triage_enabled:
        return "promising", None
    try:
        res = llm_call(
            session, target, kind="screen", job_id=job.id, prompt_version=SCREEN_PROMPT_VERSION,
            profile_hash=profile.fingerprint(), system=screen_system_prompt(profile, prefs),
            user=screen_user_prompt(job, prefs.llm.max_description_chars_triage), schema=JobScreening,
        )
    except LLMError as exc:
        return _fail(job, exc), None
    screening = res.data if res.cached else clean_screening(dict(res.data), profile)
    res.record.result = screening

    job.role_family = job.role_family or screening.get("role_family")
    job.sector = job.sector or screening.get("sector")
    if job.seniority not in _SENIORITIES and canonical_level(screening.get("seniority")):
        job.seniority = canonical_level(screening.get("seniority"))

    if screening.get("defence_related"):
        reason = screening.get("defence_reason") or screening.get("reason")
        if float(screening.get("confidence") or 0) >= DEFENCE_REJECT_CONFIDENCE:
            llm_reject(job, "llm_defence", f"Defence-related (screening): {reason}")
            job.analysis_status = AnalysisStatus.SKIPPED
            rescore(job)
            return "rejected (defence)", screening
        # uncertain: flag for human review and let the deep analysis look at the actual business unit
        llm_flag(job, "llm_defence_check", f"Possibly defence-related (screening, low confidence): {reason}")
    if (
        prefs.hard_filters.triage_reject_unrelated
        and screening.get("profession_unrelated")
        and not screening.get("likely_relevant")
        and float(screening.get("confidence") or 0) >= 0.8
    ):
        llm_reject(job, "llm_profession", f"Unrelated profession (screening): {screening.get('reason')}")
        job.analysis_status = AnalysisStatus.SKIPPED
        rescore(job)
        return "rejected (profession)", screening
    if not screening.get("likely_relevant") or screening["relevance_score"] < prefs.llm.min_screen_relevance:
        job.current_analysis = res.record
        job.current_analysis_id = res.record.id
        job.analysis_status = AnalysisStatus.SCREENED
        rescore(job)
        rebuild_search_text(job)
        return "screened out", screening
    return "promising", screening


def latest_screening(session: Session, job_id: int) -> tuple[JobAnalysis | None, dict[str, Any] | None]:
    rec = session.scalars(
        select(JobAnalysis)
        .where(JobAnalysis.job_id == job_id, JobAnalysis.kind == "screen", JobAnalysis.error.is_(None))
        .order_by(JobAnalysis.created_at.desc())
        .limit(1)
    ).first()
    return rec, (rec.result if rec else None)


def deep_analyze(session: Session, job: Job, setup: LLMSetup, screening: dict[str, Any] | None) -> str:
    """Phase 2: detailed fit analysis. The LLM supplies components/evidence; scoring stays deterministic."""
    prefs, profile = get_prefs(), get_profile()
    target = setup.target(Role.ANALYSIS)
    if target is None:  # screening-only setup: keep a provisional, screening-based score
        rec, _ = latest_screening(session, job.id)
        if rec is not None:
            job.current_analysis, job.current_analysis_id = rec, rec.id
            job.analysis_status = AnalysisStatus.DONE
            rescore(job)
            rebuild_search_text(job)
        return "screened (no analysis model)"
    try:
        res = llm_call(
            session, target, kind="fit", job_id=job.id, prompt_version=FIT_PROMPT_VERSION,
            profile_hash=profile.fingerprint(), system=fit_system_prompt(profile, prefs),
            user=fit_user_prompt(job, prefs.llm.max_description_chars_fit, screening), schema=JobFitAnalysis,
        )
    except LLMError as exc:
        return _fail(job, exc)
    if not res.cached:
        res.record.result = verify_and_clean(dict(res.data), profile)
    _apply_fit(job, res.record.result, res.record)
    rescore(job)
    rebuild_search_text(job)
    return "analysed" + (" (cached)" if res.cached else "")


def _heuristic(session: Session, job: Job) -> str:
    prefs, profile = get_prefs(), get_profile()
    parsed = heuristic.analyze(job, profile, prefs)
    rec = JobAnalysis(
        job_id=job.id, kind="fit", provider="heuristic", model=heuristic.HEURISTIC_MODEL, prompt_version="heuristic-v1",
        profile_hash=profile.fingerprint(), input_hash=input_hash("heuristic", job.content_hash or ""),
        result=parsed.model_dump(),
    )
    session.add(rec)
    session.flush()
    _apply_fit(job, rec.result, rec)
    rescore(job)
    rebuild_search_text(job)
    return "heuristic"


def analyze_job(session: Session, job: Job, setup: LLMSetup | None = None) -> str:
    """Analyse one job end to end (used for explicit 'Analyse now'). Returns a short outcome label."""
    setup = setup or get_llm()
    if job.filter_status == FilterStatus.REJECTED:
        job.analysis_status = AnalysisStatus.SKIPPED
        rescore(job)
        return "skipped (hard filter)"
    if not setup.has_llm:
        return _heuristic(session, job)
    outcome, screening = screen_job(session, job, setup)
    if outcome != "promising":
        return outcome
    return deep_analyze(session, job, setup, screening)


# --------------------------------------------------------------------------- batch


def local_preflight(setup: LLMSetup, roles: list[Role]) -> str | None:
    """Problem description if a local model needed (without fallback) is unreachable/not installed, else None."""
    local = setup.local
    if local is None or not hasattr(local, "cached_health"):
        return None
    needed = {t.model for r in roles if (t := setup.target(r)) and t.is_local and t.model and not t.fallback}
    if not needed:
        return None
    health = local.cached_health()
    if not health["reachable"]:
        return f"Local inference server unreachable ({health['error']})"
    missing = sorted(m for m in needed if not local.is_installed(m, health["installed"]))
    if missing:
        return f"Local model(s) not installed: {', '.join(missing)} — run scripts/setup-local-models"
    return None


def _pending_ids(setup: LLMSetup, limit: int) -> list[int]:
    prefs = get_prefs()
    with session_scope() as s:
        heuristic_ids = select(JobAnalysis.id).where(JobAnalysis.provider == "heuristic")
        priority_order = case((TargetCompany.priority == "A", 0), (TargetCompany.priority == "B", 1), else_=2)
        return list(
            s.scalars(
                select(Job.id)
                .outerjoin(TargetCompany, Job.target_company_id == TargetCompany.id)
                .where(
                    Job.active.is_(True),
                    Job.filter_status != FilterStatus.REJECTED,
                    Job.track.in_(("main", "intern")),  # side jobs are ranked by rules, never by the models
                    or_(
                        Job.analysis_status == AnalysisStatus.PENDING,
                        (Job.analysis_status == AnalysisStatus.ERROR) & (Job.analysis_attempts < prefs.llm.max_attempts),
                        # once an LLM is available, upgrade jobs that only have a free heuristic analysis
                        (Job.analysis_status == AnalysisStatus.DONE) & Job.current_analysis_id.in_(heuristic_ids)
                        if setup.has_llm and prefs.llm.upgrade_heuristic_backlog else false(),
                    ),
                )
                .order_by(priority_order, Job.first_seen.desc())
                .limit(limit)
            )
        )


def analyze_pending(limit: int | None = None) -> dict[str, Any]:
    prefs = get_prefs()
    setup = get_llm()
    counts: dict[str, int] = {}

    def count(outcome: str) -> None:
        key = outcome.split(":")[0]
        counts[key] = counts.get(key, 0) + 1

    if not setup.has_llm:  # free heuristic: instant, no batching needed
        ids = _pending_ids(setup, limit or 100_000)
        for job_id in ids:
            with session_scope() as s:
                job = s.get(Job, job_id)
                if job:
                    count(analyze_job(s, job, setup))
        return {"queued": len(ids), "results": counts, "stopped": None}

    problem = local_preflight(setup, [Role.SCREEN, Role.ANALYSIS])
    if problem:
        log.warning("Analysis run skipped: %s", problem)
        return {"queued": 0, "results": {}, "stopped": problem}

    ids = _pending_ids(setup, limit or prefs.scheduler.screen_batch_size)
    deep_limit = prefs.scheduler.analysis_batch_size if limit is None else limit
    return {"queued": len(ids), **_two_phase(setup, ids, deep_limit, counts)}


def _two_phase(setup: LLMSetup, ids: list[int], deep_limit: int, counts: dict[str, int]) -> dict[str, Any]:
    """Phase 1: screen all `ids` (screen model only). Phase 2: deep-analyse the promising ones (analysis model only)."""

    def count(outcome: str) -> None:
        key = outcome.split(":")[0]
        counts[key] = counts.get(key, 0) + 1

    promising: list[int] = []
    stopped = None
    for job_id in ids:
        try:
            with session_scope() as s:
                job = s.get(Job, job_id)
                if job is None:
                    continue
                if job.filter_status == FilterStatus.REJECTED:
                    count(analyze_job(s, job, setup))  # hard-filtered: never sent to any model
                    continue
                outcome, _ = screen_job(s, job, setup)
        except (BudgetExceeded, LLMUnavailable) as exc:
            stopped = str(exc)
            log.warning("%s — stopping screening phase", exc)
            break
        if outcome == "promising":
            promising.append(job_id)
        else:
            count(outcome)

    # Phase 2 — after ALL screening is done, so the runtime swaps models once per run, not once per job.
    for job_id in promising[:deep_limit] if not stopped else []:
        try:
            with session_scope() as s:
                job = s.get(Job, job_id)
                if job is None:
                    continue
                _, screening = latest_screening(s, job_id)
                count(deep_analyze(s, job, setup, screening))
        except (BudgetExceeded, LLMUnavailable) as exc:
            stopped = str(exc)
            log.warning("%s — stopping analysis phase", exc)
            break
    deferred = max(0, len(promising) - deep_limit) if not stopped else 0
    if deferred:
        counts["deferred to next run"] = deferred
    return {"screened_promising": len(promising), "results": counts, "stopped": stopped}


def analyze_selected(job_ids: list[int], max_jobs: int = 150) -> dict[str, Any]:
    """Calibration run: screening + deep analysis for exactly these jobs (e.g. the human-labeled ones), nothing else.

    Unchanged jobs that were analysed before are served from the cache (no model call)."""
    setup = get_llm()
    ids = list(dict.fromkeys(job_ids))[:max_jobs]
    counts: dict[str, int] = {}
    if not setup.has_llm:
        return {"queued": len(ids), "results": {}, "stopped": "No LLM configured (local or cloud) — nothing to compare"}
    problem = local_preflight(setup, [Role.SCREEN, Role.ANALYSIS])
    if problem:
        return {"queued": 0, "results": {}, "stopped": problem}
    with session_scope() as s:
        for job in s.scalars(select(Job).where(Job.id.in_(ids))):
            job.analysis_attempts = 0  # explicit request: give previously failed jobs a fresh try
    return {"queued": len(ids), **_two_phase(setup, ids, len(ids), counts)}


def mark_all_for_reanalysis() -> int:
    """Queue every non-rejected active job again. Unchanged inputs are served from the cache (free)."""
    with session_scope() as s:
        jobs = s.scalars(select(Job).where(Job.active.is_(True), Job.filter_status != FilterStatus.REJECTED)).all()
        for job in jobs:
            job.analysis_status = AnalysisStatus.PENDING
            job.analysis_attempts = 0
        return len(jobs)


def rescore_all() -> int:
    from app.services.jobs import run_hard_filters

    with session_scope() as s:
        jobs = s.scalars(select(Job)).all()
        for job in jobs:
            run_hard_filters(job)
            rescore(job)
        return len(jobs)


def extract_fields(session: Session, text: str) -> dict[str, str]:
    """Cheap-model extraction of title/company/location from pasted text (manual ingestion)."""
    target = get_llm().target(Role.EXTRACT)
    if target is None:
        return {}
    try:
        res = llm_call(
            session, target, kind="extract", job_id=None, prompt_version=EXTRACT_PROMPT_VERSION, profile_hash="-",
            system=EXTRACT_SYSTEM, user=extract_user_prompt(text), schema=ExtractedPosting,
        )
        data = dict(res.data)
        data["title"] = data.pop("job_title", "") or data.get("title", "")
        return {k: v for k, v in data.items() if v}
    except (LLMError, BudgetExceeded) as exc:
        log.warning("Field extraction failed: %s", exc)
        return {}


def usage_summary(session: Session) -> dict[str, Any]:
    rows = session.execute(
        select(
            JobAnalysis.provider,
            JobAnalysis.model,
            JobAnalysis.kind,
            func.count(),
            func.sum(case((JobAnalysis.cached.is_(True), 1), else_=0)),
            func.sum(case((JobAnalysis.error.is_not(None), 1), else_=0)),
            func.sum(JobAnalysis.input_tokens),
            func.sum(JobAnalysis.output_tokens),
            func.sum(JobAnalysis.cost_usd),
            func.avg(case((JobAnalysis.cached.is_(False) & JobAnalysis.error.is_(None), JobAnalysis.duration_ms))),
        ).group_by(JobAnalysis.provider, JobAnalysis.model, JobAnalysis.kind)
    ).all()
    total = sum(r[8] or 0 for r in rows)
    return {
        "rows": [
            {"provider": r[0], "model": r[1], "kind": r[2], "calls": r[3], "cached": r[4] or 0, "errors": r[5] or 0,
             "input_tokens": r[6] or 0, "output_tokens": r[7] or 0, "cost": round(r[8] or 0, 4),
             "avg_ms": int(r[9]) if r[9] is not None else None}
            for r in rows
        ],
        "total_cost": round(total, 4),
        "today_cost": round(spent_today(session), 4),
        "budget": get_prefs().llm.daily_budget_usd,
    }
