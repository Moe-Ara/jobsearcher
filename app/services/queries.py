"""Dashboard/API job queries: filters, local search, stats."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from sqlalchemy import Select, and_, func, nulls_last, select
from sqlalchemy.orm import Session, selectinload

from app.enums import FilterStatus, JobStatus, Recommendation
from app.models import Job, JobSource, TargetCompany, utcnow


@dataclass
class JobFilters:
    q: str = ""
    country: str = ""
    location: str = ""
    min_score: float | None = None
    sector: str = ""
    role_family: str = ""
    company: str = ""
    source: str = ""
    seniority: str = ""
    remote: str = ""
    status: str = ""
    since_days: int | None = None
    company_priority: str = ""
    recommendation: str = ""
    relevance: str = ""
    unreviewed: bool = True
    show_rejected: bool = False
    show_inactive: bool = False
    sort: str = "score"
    page: int = 1
    track: str = "main"  # main = career search; side = part-time / temporary (own page)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_params(cls, params: dict[str, str], default_since: int | None) -> JobFilters:
        def num(key: str, typ=float):
            try:
                return typ(params[key]) if params.get(key) not in (None, "") else None
            except ValueError:
                return None

        # Checkbox semantics: an explicitly submitted form carries "applied=1".
        submitted = params.get("applied") == "1"
        return cls(
            q=params.get("q", "").strip(),
            country=params.get("country", ""),
            location=params.get("location", "").strip(),
            min_score=num("min_score"),
            sector=params.get("sector", ""),
            role_family=params.get("role_family", ""),
            company=params.get("company", "").strip(),
            source=params.get("source", ""),
            seniority=params.get("seniority", ""),
            remote=params.get("remote", ""),
            status=params.get("status", ""),
            since_days=num("since_days", int) if "since_days" in params else default_since,
            company_priority=params.get("company_priority", ""),
            recommendation=params.get("recommendation", ""),
            relevance=params.get("relevance", ""),
            unreviewed=(params.get("unreviewed") == "1") if submitted else True,
            show_rejected=params.get("show_rejected") == "1",
            show_inactive=params.get("show_inactive") == "1",
            sort=params.get("sort", "score"),
            page=max(1, num("page", int) or 1),
        )


def _apply(stmt: Select, f: JobFilters) -> Select:
    conds = [Job.track == f.track]
    if f.q:
        for term in f.q.lower().split():
            conds.append(Job.search_text.contains(term, autoescape=True))
    if f.country:
        conds.append(Job.country == f.country)
    if f.location:
        conds.append(func.lower(Job.location).contains(f.location.lower()))
    if f.min_score is not None:
        conds.append(Job.fit_score >= f.min_score)
    if f.sector:
        conds.append(Job.sector == f.sector)
    if f.role_family:
        conds.append(Job.role_family == f.role_family)
    if f.company:
        conds.append(func.lower(Job.company).contains(f.company.lower()))
    if f.source:
        conds.append(Job.sources.any(JobSource.source == f.source))
    if f.seniority:
        conds.append(Job.seniority == f.seniority)
    if f.remote:
        conds.append(Job.remote_status == f.remote)
    if f.status:
        conds.append(Job.status == f.status)
    elif f.unreviewed:
        conds.append(Job.status == JobStatus.NEW)
    if f.since_days:
        conds.append(Job.first_seen >= utcnow() - timedelta(days=f.since_days))
    if f.company_priority:
        conds.append(Job.target_company.has(TargetCompany.priority == f.company_priority))
    if f.recommendation:
        conds.append(Job.recommendation == f.recommendation)
    if f.relevance:
        conds.append(Job.relevance_label == f.relevance)
    if not f.show_rejected and f.recommendation != Recommendation.REJECTED:
        conds.append(Job.filter_status != FilterStatus.REJECTED)
    if not f.show_inactive and f.status != JobStatus.EXPIRED:
        conds.append(Job.active.is_(True))
    return stmt.where(and_(*conds)) if conds else stmt


def search_jobs(s: Session, f: JobFilters, page_size: int = 50) -> tuple[list[Job], int]:
    base = _apply(select(Job), f)
    total = s.scalar(select(func.count()).select_from(base.subquery())) or 0
    order = {
        "score": [nulls_last(Job.fit_score.desc()), Job.first_seen.desc()],
        "newest": [Job.first_seen.desc()],
        "posted": [nulls_last(Job.date_posted.desc()), Job.first_seen.desc()],
        "company": [Job.company.asc(), nulls_last(Job.fit_score.desc())],
    }.get(f.sort, [nulls_last(Job.fit_score.desc())])
    stmt = (
        base.options(selectinload(Job.sources), selectinload(Job.current_analysis), selectinload(Job.target_company))
        .order_by(*order)
        .offset((f.page - 1) * page_size)
        .limit(page_size)
    )
    return list(s.scalars(stmt)), total


def dashboard_stats(s: Session, since_days: int | None) -> dict[str, int]:
    since = utcnow() - timedelta(days=since_days or 3650)
    window = [Job.first_seen >= since, Job.track == "main"]
    count = lambda *c: s.scalar(select(func.count(Job.id)).where(*window, *c)) or 0  # noqa: E731
    return {
        "new_jobs": count(),
        "passed": count(Job.filter_status != FilterStatus.REJECTED),
        "rejected": count(Job.filter_status == FilterStatus.REJECTED),
        "high_match": count(Job.recommendation.in_([Recommendation.REVIEW, Recommendation.PRIORITY])),
        "priority": count(Job.recommendation == Recommendation.PRIORITY),
        "pending": count(Job.recommendation == Recommendation.PENDING, Job.filter_status != FilterStatus.REJECTED),
        "unreviewed": s.scalar(
            select(func.count(Job.id)).where(Job.status == JobStatus.NEW, Job.active.is_(True), Job.filter_status != FilterStatus.REJECTED,
                                             Job.track == "main")
        ) or 0,
    }


def filter_options(s: Session) -> dict[str, list[str]]:
    def distinct_values(col) -> list[str]:
        return [v for v in s.scalars(select(col).distinct().where(col.is_not(None), col != "").order_by(col)) if v]

    return {
        "country": distinct_values(Job.country),
        "sector": distinct_values(Job.sector),
        "role_family": distinct_values(Job.role_family),
        "seniority": distinct_values(Job.seniority),
        "source": [v for v in s.scalars(select(JobSource.source).distinct().order_by(JobSource.source)) if v],
        "remote": ["remote", "hybrid", "onsite"],
        "status": [st.value for st in JobStatus],
        "recommendation": [r.value for r in Recommendation],
        "company_priority": ["A", "B", "C"],
        "relevance": ["UNRATED", "GREAT", "MAYBE", "BAD"],
    }
