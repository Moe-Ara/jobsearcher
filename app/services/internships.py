"""Internships and trainee programmes (job.track == "intern"), on their own page — stricter than the career search.

A posting joins this track when its title says Praktikum / Intern / Trainee (from any source), or when it comes from
a source configured with `track: intern` (Bundesagentur internship search). Shown only when all of these hold:

  - your field: the title names software, AI / ML, data, robotics, embedded, IT … and no unrelated function
    (marketing, HR, finance, sales …)
  - open to graduates: no student-only formats (Werkstudent, Pflichtpraktikum, Praxissemester, Schülerpraktikum,
    Ausbildung, theses) and no enrollment requirement in the text ("immatrikuliert", "während deines Studiums"),
    unless the posting also welcomes graduates ("Absolventen", "nach dem Studium")
  - near home: at most internships.max_drive_minutes by car (real road routing), or fully remote; Germany only
  - the defence, untrusted-source and nationals-only rules of the career search

Ranked by the same AI analysis as your career jobs, with two differences: being an internship is not held against
it (scored like a junior position), and the location score comes from the drive time.

Drive times: postings with coordinates (Bundesagentur) are routed at fetch time; for the others the town is geocoded
(OpenStreetMap, cached) by resolve_distances() — outside any database transaction, before each analysis run.
Until then a posting waits (flag "intern_distance_pending") and is not analysed.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import select

from app.config_files import Preferences
from app.enums import AnalysisStatus, FilterStatus, Recommendation
from app.filters import FilterOutcome, defence_check, nationals_only_check, untrusted_source_check

log = logging.getLogger(__name__)

PENDING_RULE = "intern_distance_pending"


def is_internship(title: str | None, prefs: Preferences) -> bool:
    ip = prefs.internships
    return ip.enabled and bool(re.search(ip.posting_pattern, title or "", re.IGNORECASE))


def _first(patterns: list[str], text: str) -> str | None:
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return None


def _home(prefs: Preferences) -> tuple[float, float] | None:
    from app.services.drive_time import home_point

    _, lat, lon = home_point(prefs)
    return (lat, lon) if lat is not None and lon is not None else None


def _drive_minutes(job, prefs: Preferences) -> float | None:
    """Stored drive time, or one the caches already know (geocoded town + routed before)."""
    from app.services.drive_time import cached_minutes, cached_place, place_query

    raw = job.raw_source_data or {}
    if isinstance(raw.get("drive_minutes"), (int, float)):
        return float(raw["drive_minutes"])
    place = cached_place(place_query(job.location))
    if place and _home(prefs):
        minutes = cached_minutes(_home(prefs), place)
        if minutes is not None:
            job.raw_source_data = {**raw, "drive_minutes": minutes, "geo": list(place)}
            return minutes
    return None


def intern_filters(job, prefs: Preferences) -> FilterOutcome:
    from app.config_files import get_profile
    from app.services.drive_time import cached_place, place_query

    ip = prefs.internships
    enrolled = get_profile().enrolled_student
    out = FilterOutcome()
    title = job.title or ""
    hit = _first(ip.excluded_title_patterns, title) or (None if enrolled else _first(ip.student_title_patterns, title))
    if hit:
        why = "you are not enrolled" if not enrolled or not _first(ip.excluded_title_patterns, title) else "not an internship"
        out.reject("intern_format", f"School / vocational / student-only format (“{hit}”) — {why}")
        return out
    if not re.search(ip.field_title_pattern, title, re.IGNORECASE):
        out.reject("intern_field", "Not in your field (software, AI / ML, data, robotics, embedded, IT)")
        return out
    m = re.search(ip.unrelated_title_pattern, title, re.IGNORECASE)
    if m:
        out.reject("intern_field", f"Not your field (“{m.group(0).strip()}”)")
        return out
    text = "\n".join(x for x in (title, job.description, job.requirements) if x)
    student_only = _first(ip.enrollment_patterns, text)
    if student_only and not enrolled and not _first(ip.graduate_patterns, text):
        if ip.student_only != "flag":
            out.reject("intern_enrollment", f"Requires being enrolled at a university (“{student_only}”) — you are not enrolled")
            return out
        out.flag("intern_enrollment", f"Asks for enrolled students (“{student_only}”) — only if you are enrolled")
    untrusted_source_check(job, prefs, out)
    defence_check(job, prefs, out)
    nationals_only_check(job, prefs, out)
    if out.rejected:
        return out
    remote = (job.remote_status or "") == "remote"
    if job.country and job.country != "Germany":
        out.reject("intern_location", f"Internships are kept near home (this one is in {job.country})")
        return out
    if remote:
        return out
    if _home(prefs) is None:
        out.flag("intern_no_home", "Set your home town (Settings) to keep internships within driving distance")
        return out
    minutes = _drive_minutes(job, prefs)
    if minutes is not None:
        if minutes > ip.max_drive_minutes:
            out.reject("intern_distance", f"{minutes:.0f} min by car — more than {ip.max_drive_minutes} min from home")
        return out
    query = place_query(job.location)
    if not query:
        out.reject("intern_location", "No town given (and not fully remote) — can't tell whether it is near you")
    elif cached_place(query) is None:
        out.reject("intern_location", f"Could not locate “{job.location}” on the map")
    else:
        out.flag(PENDING_RULE, "Drive time from home is being looked up")
    return out


def apply_intern(job, prefs: Preferences) -> None:
    """Filters for an internship (LLM-originated rejections are kept, like in the career search)."""
    out = intern_filters(job, prefs)
    llm_reasons = [r for r in (job.filter_reasons or []) if str(r.get("rule", "")).startswith("llm_")]
    status = out.status
    if any(r["severity"] == "reject" for r in llm_reasons):
        status = FilterStatus.REJECTED
    elif status == FilterStatus.PASSED and llm_reasons:
        status = FilterStatus.FLAGGED
    job.filter_status = status
    job.filter_reasons = out.reasons + llm_reasons
    waiting = any(r["rule"] == PENDING_RULE for r in out.reasons)
    if status == FilterStatus.REJECTED or waiting:  # no model time for a posting that is too far or not for you
        if job.analysis_status in (AnalysisStatus.PENDING, AnalysisStatus.ERROR):
            job.analysis_status = AnalysisStatus.SKIPPED
    elif job.analysis_status == AnalysisStatus.SKIPPED:
        job.analysis_status = AnalysisStatus.DONE if job.current_analysis_id else AnalysisStatus.PENDING


def location_points(job, prefs: Preferences) -> float:
    if (job.remote_status or "") == "remote":
        return 9.0
    minutes = (job.raw_source_data or {}).get("drive_minutes")
    if not isinstance(minutes, (int, float)):
        return 6.0
    return 10.0 if minutes <= 15 else 9.0 if minutes <= 30 else 7.5 if minutes <= prefs.internships.max_drive_minutes else 5.0


def intern_rescore(job, prefs: Preferences, candidate_years: float | None) -> None:
    from app.scoring import compute_score
    from app.services.jobs import current_analysis_result

    res = compute_score(
        filter_status=job.filter_status, analysis=current_analysis_result(job), country=job.country,
        location=job.location, remote_status=job.remote_status,
        seniority="Junior",  # an internship is what this page is for: not held against it
        best_source_type=job.best_source_type,
        target_priority=job.target_company.priority if job.target_company else None, prefs=prefs,
        years_required=job.years_experience_required, candidate_years=candidate_years,
        location_override=location_points(job, prefs),
    )
    if job.analysis_status == AnalysisStatus.SCREENED and res.score is not None:
        res.score = min(res.score, prefs.scoring.thresholds.get("review", 6.5) - 0.01)
        res.recommendation = Recommendation.LOW_PRIORITY
        res.breakdown["screened_out"] = True
    res.breakdown["intern"] = True
    res.breakdown["drive_minutes"] = (job.raw_source_data or {}).get("drive_minutes")
    job.fit_score = res.score
    job.recommendation = res.recommendation
    job.score_breakdown = res.breakdown


def resolve_distances(limit: int = 60) -> dict[str, Any]:
    """Geocode + route the internships still waiting for a drive time, then re-apply their filters. The lookups
    happen between two short transactions — never while holding the database."""
    from app.config_files import get_prefs
    from app.db import session_scope
    from app.models import Job
    from app.services.drive_time import cached_place, drive_minutes, geocode, key, place_query
    from app.services.jobs import rescore, run_hard_filters

    prefs = get_prefs()
    with session_scope() as s:
        waiting = [(j.id, place_query(j.location)) for j in s.scalars(
            select(Job).where(Job.track == "intern", Job.active.is_(True), Job.filter_status == FilterStatus.FLAGGED))
            if any(r.get("rule") == PENDING_RULE for r in j.filter_reasons or [])]
    home = _home(prefs)
    if not waiting or home is None:
        return {"waiting": len(waiting)}
    found = geocode([q for _, q in waiting if q], limit=limit)
    places = {q: c for q, c in found.items() if c}
    minutes = drive_minutes(home, places.values()) if places else {}
    done = 0
    with session_scope() as s:
        for job_id, query in waiting:
            if cached_place(query) is False:
                continue  # not looked up yet (limit / network) — next run
            job = s.get(Job, job_id)
            if job is None:
                continue
            place = places.get(query)
            if place:
                job.raw_source_data = {**(job.raw_source_data or {}), "drive_minutes": minutes.get(key(*place)),
                                       "geo": list(place)}
            run_hard_filters(job)
            rescore(job)
            done += 1
    return {"waiting": len(waiting), "resolved": done, "looked_up": len(found)}
