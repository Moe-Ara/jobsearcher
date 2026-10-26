"""Simple application analytics derived from status history (no separate analytics store)."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.enums import FUNNEL, JobStatus
from app.models import Job, StatusEvent, utcnow

_STAGE_INDEX = {s: i for i, s in enumerate(FUNNEL)}
_RESPONSE = _STAGE_INDEX[JobStatus.HR_SCREEN]
_INTERVIEW = _STAGE_INDEX[JobStatus.INTERVIEW]
_OFFER = _STAGE_INDEX[JobStatus.OFFER]


def _furthest_stage(job: Job) -> int:
    """Highest funnel index ever reached (-1 = never applied). Employer rejection after applying counts as a response."""
    best = -1
    statuses = [e.to_status for e in job.status_events] + [job.status]
    for st in statuses:
        best = max(best, _STAGE_INDEX.get(st, -1))
    return best


def _responded(job: Job, stage: int) -> bool:
    if stage >= _RESPONSE:
        return True
    return stage >= 0 and job.status == JobStatus.REJECTED


def _bucket(score: float | None) -> str:
    if score is None:
        return "unscored"
    if score >= 8:
        return "8+"
    if score >= 6.5:
        return "6.5–8"
    return "<6.5"


def summary(s: Session) -> dict[str, Any]:
    jobs = list(s.scalars(select(Job).options(selectinload(Job.status_events), selectinload(Job.sources))))
    applied = [(j, _furthest_stage(j)) for j in jobs]
    applied = [(j, st) for j, st in applied if st >= 0]

    def rates(items: list[tuple[Job, int]]) -> dict[str, Any]:
        n = len(items)
        resp = sum(1 for j, st in items if _responded(j, st))
        inter = sum(1 for _, st in items if st >= _INTERVIEW)
        offers = sum(1 for _, st in items if st >= _OFFER)
        pct = lambda x: round(100 * x / n, 1) if n else 0.0  # noqa: E731
        return {"applied": n, "responses": resp, "interviews": inter, "offers": offers,
                "response_rate": pct(resp), "interview_rate": pct(inter), "offer_rate": pct(offers)}

    def breakdown(key) -> list[dict[str, Any]]:
        groups: dict[str, list[tuple[Job, int]]] = defaultdict(list)
        for j, st in applied:
            groups[key(j) or "—"].append((j, st))
        return sorted(({"key": k, **rates(v)} for k, v in groups.items()), key=lambda r: -r["applied"])

    # applications per ISO week (last 12 weeks), from the APPLIED status event
    weeks: dict[str, int] = defaultdict(int)
    since = utcnow() - timedelta(weeks=12)
    for ev in s.scalars(select(StatusEvent).where(StatusEvent.to_status == JobStatus.APPLIED, StatusEvent.changed_at >= since)):
        y, w, _ = ev.changed_at.isocalendar()
        weeks[f"{y}-W{w:02d}"] += 1

    status_counts: dict[str, int] = defaultdict(int)
    for j in jobs:
        status_counts[j.status] += 1

    return {
        "overall": rates(applied),
        "per_week": sorted(weeks.items()),
        "status_counts": dict(sorted(status_counts.items())),
        "by": {
            "Source": breakdown(lambda j: j.source),
            "Country": breakdown(lambda j: j.country),
            "Sector": breakdown(lambda j: j.sector),
            "Role family": breakdown(lambda j: j.role_family),
            "Fit score": breakdown(lambda j: _bucket(j.fit_score)),
            "CV version": breakdown(lambda j: j.cv_version),
        },
    }
