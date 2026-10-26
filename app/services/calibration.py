"""Human relevance labels and ranking evaluation (evaluation only — labels never change scores or weights)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.enums import RELEVANCE_REASONS, FilterStatus, RelevanceLabel
from app.models import Job, utcnow

K_VALUES = (5, 10, 20)


def set_relevance(job: Job, label: str, reasons: list[str] | None = None) -> None:
    label = RelevanceLabel(label)
    job.relevance_label = label
    job.relevance_reasons = [r for r in (reasons or []) if r in RELEVANCE_REASONS] or None
    job.relevance_labeled_at = None if label == RelevanceLabel.UNRATED else utcnow()


def precision_at_k(ranked_labels: list[str], k: int, relevant: set[str]) -> dict[str, Any]:
    """Precision@k over the top-k of a ranking (fewer items than k → computed over what exists, n reported)."""
    top = ranked_labels[:k]
    if not top:
        return {"k": k, "n": 0, "value": None}
    return {"k": k, "n": len(top), "value": round(sum(1 for lbl in top if lbl in relevant) / len(top), 3)}


def ranking_metrics(session: Session) -> dict[str, Any]:
    counts = dict.fromkeys((lbl.value for lbl in RelevanceLabel), 0)
    for label, n in session.execute(select(Job.relevance_label, func.count()).group_by(Job.relevance_label)):
        counts[label] = n

    labeled = list(session.scalars(
        select(Job).where(Job.relevance_label != RelevanceLabel.UNRATED).options(selectinload(Job.current_analysis))
    ))
    scored = sorted((j for j in labeled if j.fit_score is not None), key=lambda j: (-j.fit_score, j.id))
    ranked = [j.relevance_label for j in scored]

    avg: dict[str, float | None] = {}
    for lbl in (RelevanceLabel.GREAT, RelevanceLabel.MAYBE, RelevanceLabel.BAD):
        scores = [j.fit_score for j in scored if j.relevance_label == lbl]
        avg[lbl.value] = round(sum(scores) / len(scores), 2) if scores else None

    unscored = {lbl.value: sum(1 for j in labeled if j.fit_score is None and j.relevance_label == lbl)
                for lbl in (RelevanceLabel.GREAT, RelevanceLabel.MAYBE, RelevanceLabel.BAD)}
    analysis_sources: dict[str, int] = {}
    for j in scored:
        src = j.current_analysis.provider if j.current_analysis else "none"
        analysis_sources[src] = analysis_sources.get(src, 0) + 1

    great, good = {RelevanceLabel.GREAT.value}, {RelevanceLabel.GREAT.value, RelevanceLabel.MAYBE.value}
    return {
        "counts": counts,
        "labeled": len(labeled),
        "ranked": len(ranked),
        "primary": [precision_at_k(ranked, k, great) for k in K_VALUES],
        "secondary": [precision_at_k(ranked, k, good) for k in K_VALUES],
        # what a random ordering of the same labeled jobs would score on average
        "base_rate": {
            "primary": round(sum(1 for lbl in ranked if lbl in great) / len(ranked), 3) if ranked else None,
            "secondary": round(sum(1 for lbl in ranked if lbl in good) / len(ranked), 3) if ranked else None,
        },
        "avg_score": avg,
        "unscored_labeled": unscored,
        "analysis_sources": analysis_sources,
        "rejected_but_great": sum(1 for j in labeled if j.relevance_label == RelevanceLabel.GREAT
                                  and j.filter_status == FilterStatus.REJECTED),
    }


def labeled_job_ids(session: Session) -> list[int]:
    return list(session.scalars(
        select(Job.id).where(Job.relevance_label != RelevanceLabel.UNRATED).order_by(Job.relevance_labeled_at)
    ))


@dataclass
class CalibrationQuery:
    unrated_only: bool = True
    include_rejected: bool = False
    order: str = "shuffled"  # shuffled | score | newest
    offset: int = 0


def calibration_candidates(session: Session, q: CalibrationQuery):
    stmt = select(Job).where(Job.active.is_(True))
    if q.unrated_only:
        stmt = stmt.where(Job.relevance_label == RelevanceLabel.UNRATED)
    if not q.include_rejected:
        stmt = stmt.where(Job.filter_status != FilterStatus.REJECTED)
    if q.order == "score":
        stmt = stmt.order_by(Job.fit_score.desc().nulls_last(), Job.id)
    elif q.order == "newest":
        stmt = stmt.order_by(Job.first_seen.desc(), Job.id)
    else:  # stable pseudo-random order → a representative, repeatable sample that works with offsets
        stmt = stmt.order_by((Job.id * 7919) % 10007, Job.id)
    total = session.scalar(select(func.count()).select_from(stmt.order_by(None).subquery())) or 0
    job = session.scalars(
        stmt.options(selectinload(Job.sources), selectinload(Job.current_analysis), selectinload(Job.target_company))
        .offset(max(0, q.offset)).limit(1)
    ).first()
    return job, total
