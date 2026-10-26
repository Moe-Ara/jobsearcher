"""SQLAlchemy models.

Portable types only (String/Text/JSON/DateTime/Float/Integer/Boolean) so SQLite can be swapped for
PostgreSQL by changing DATABASE_URL. Enums are stored as strings (see app.enums).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.enums import AnalysisStatus, FilterStatus, JobStatus, Recommendation, SourceType


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)  # naive UTC everywhere (SQLite-friendly)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


# --------------------------------------------------------------------------- target companies


class TargetCompany(TimestampMixin, Base):
    __tablename__ = "target_companies"

    id: Mapped[int] = mapped_column(primary_key=True)
    company_name: Mapped[str] = mapped_column(String(200))
    normalized_name: Mapped[str] = mapped_column(String(200), index=True)
    aliases: Mapped[list[Any]] = mapped_column(JSON, default=list)
    country: Mapped[str | None] = mapped_column(String(80))
    city: Mapped[str | None] = mapped_column(String(120))
    sector: Mapped[str | None] = mapped_column(String(120))
    priority: Mapped[str] = mapped_column(String(1), default="B")  # A / B / C
    careers_url: Mapped[str | None] = mapped_column(String(1000))
    ats_type: Mapped[str | None] = mapped_column(String(40))  # greenhouse|lever|personio|smartrecruiters|generic|none
    ats_identifier: Mapped[str | None] = mapped_column(String(200))
    ats_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_checked: Mapped[datetime | None] = mapped_column(DateTime)
    last_status: Mapped[str | None] = mapped_column(String(20))  # ok | error
    last_error: Mapped[str | None] = mapped_column(Text)
    last_jobs_found: Mapped[int | None] = mapped_column(Integer)
    notes: Mapped[str | None] = mapped_column(Text)

    jobs: Mapped[list[Job]] = relationship(back_populates="target_company")


# --------------------------------------------------------------------------- jobs


class Job(TimestampMixin, Base):
    """One canonical vacancy. Multiple JobSource rows may point at it (dedup keeps provenance)."""

    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_dashboard", "filter_status", "status", "fit_score"),
        Index("ix_jobs_company_title", "normalized_company", "normalized_title"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    external_job_id: Mapped[str | None] = mapped_column(String(200))
    title: Mapped[str] = mapped_column(String(500))
    normalized_title: Mapped[str] = mapped_column(String(500), default="")
    company: Mapped[str] = mapped_column(String(300), default="")
    normalized_company: Mapped[str] = mapped_column(String(300), default="", index=True)
    location: Mapped[str | None] = mapped_column(String(500))
    country: Mapped[str | None] = mapped_column(String(80), index=True)
    remote_status: Mapped[str | None] = mapped_column(String(20))
    description: Mapped[str | None] = mapped_column(Text)
    requirements: Mapped[str | None] = mapped_column(Text)
    responsibilities: Mapped[str | None] = mapped_column(Text)
    salary_min: Mapped[float | None] = mapped_column(Float)
    salary_max: Mapped[float | None] = mapped_column(Float)
    currency: Mapped[str | None] = mapped_column(String(10))
    employment_type: Mapped[str | None] = mapped_column(String(80))
    seniority: Mapped[str | None] = mapped_column(String(40))
    sector: Mapped[str | None] = mapped_column(String(120))
    role_family: Mapped[str | None] = mapped_column(String(120))
    source: Mapped[str] = mapped_column(String(60))  # best/canonical source name
    best_source_type: Mapped[str] = mapped_column(String(20), default=SourceType.MANUAL)
    source_url: Mapped[str | None] = mapped_column(String(2000))
    canonical_url: Mapped[str | None] = mapped_column(String(2000), index=True)
    career_page_url: Mapped[str | None] = mapped_column(String(2000), index=True)
    date_posted: Mapped[datetime | None] = mapped_column(DateTime)
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    language_requirements: Mapped[list[Any]] = mapped_column(JSON, default=list)
    years_experience_required: Mapped[float | None] = mapped_column(Float)
    raw_source_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # pipeline state
    content_hash: Mapped[str | None] = mapped_column(String(64))
    filter_status: Mapped[str] = mapped_column(String(20), default=FilterStatus.PENDING)
    filter_reasons: Mapped[list[Any]] = mapped_column(JSON, default=list)
    analysis_status: Mapped[str] = mapped_column(String(20), default=AnalysisStatus.PENDING)
    analysis_attempts: Mapped[int] = mapped_column(Integer, default=0)
    current_analysis_id: Mapped[int | None] = mapped_column(
        ForeignKey("job_analyses.id", use_alter=True, ondelete="SET NULL", name="fk_jobs_current_analysis")
    )
    fit_score: Mapped[float | None] = mapped_column(Float)
    score_breakdown: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    recommendation: Mapped[str] = mapped_column(String(20), default=Recommendation.PENDING)
    search_text: Mapped[str] = mapped_column(Text, default="")

    # human recommendation-quality label (evaluation only — never feeds back into scoring)
    relevance_label: Mapped[str] = mapped_column(String(10), default="UNRATED", server_default="UNRATED", index=True)
    relevance_reasons: Mapped[list[Any] | None] = mapped_column(JSON(none_as_null=True))
    relevance_labeled_at: Mapped[datetime | None] = mapped_column(DateTime)

    # human review / application tracking
    status: Mapped[str] = mapped_column(String(20), default=JobStatus.NEW, index=True)
    # main = your career search · side = part-time / mini-job / temporary work of any kind (own filters and ranking)
    track: Mapped[str] = mapped_column(String(12), default="main", server_default="main", index=True)
    status_changed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    notes: Mapped[str | None] = mapped_column(Text)
    cv_version: Mapped[str | None] = mapped_column(String(120))
    applied_at: Mapped[datetime | None] = mapped_column(DateTime)

    target_company_id: Mapped[int | None] = mapped_column(
        ForeignKey("target_companies.id", ondelete="SET NULL"), index=True
    )
    target_company: Mapped[TargetCompany | None] = relationship(back_populates="jobs")
    sources: Mapped[list[JobSource]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="JobSource.first_seen"
    )
    analyses: Mapped[list[JobAnalysis]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        foreign_keys="JobAnalysis.job_id",
        order_by="JobAnalysis.created_at",
    )
    current_analysis: Mapped[JobAnalysis | None] = relationship(
        foreign_keys=[current_analysis_id], post_update=True
    )
    status_events: Mapped[list[StatusEvent]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="StatusEvent.changed_at"
    )


class JobSource(Base):
    """Provenance: one row per (source, posting). Never deleted when jobs are merged."""

    __tablename__ = "job_sources"
    __table_args__ = (UniqueConstraint("source_key", name="uq_job_sources_source_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    source: Mapped[str] = mapped_column(String(60))  # greenhouse, lever, linkedin, manual, ...
    source_type: Mapped[str] = mapped_column(String(20))
    source_key: Mapped[str] = mapped_column(String(600))  # "<source>:<external id or canonical url>"
    external_job_id: Mapped[str | None] = mapped_column(String(200))
    url: Mapped[str | None] = mapped_column(String(2000))
    canonical_url: Mapped[str | None] = mapped_column(String(2000), index=True)
    title: Mapped[str | None] = mapped_column(String(500))
    company: Mapped[str | None] = mapped_column(String(300))
    location: Mapped[str | None] = mapped_column(String(500))
    raw_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    target_company_id: Mapped[int | None] = mapped_column(
        ForeignKey("target_companies.id", ondelete="SET NULL"), index=True
    )
    email_message_id: Mapped[int | None] = mapped_column(
        ForeignKey("email_messages.id", ondelete="SET NULL")
    )
    dedup_reason: Mapped[str | None] = mapped_column(String(300))
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    job: Mapped[Job] = relationship(back_populates="sources")


class JobAnalysis(Base):
    """Every LLM (or heuristic) call: triage, fit analysis, extraction. Also the result cache."""

    __tablename__ = "job_analyses"
    __table_args__ = (Index("ix_job_analyses_cache", "kind", "input_hash"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int | None] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20))  # triage | fit | extract
    provider: Mapped[str] = mapped_column(String(40))
    model: Mapped[str] = mapped_column(String(120))
    prompt_version: Mapped[str] = mapped_column(String(40))
    profile_hash: Mapped[str | None] = mapped_column(String(32))
    input_hash: Mapped[str] = mapped_column(String(64))
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    error: Mapped[str | None] = mapped_column(Text)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    job: Mapped[Job | None] = relationship(back_populates="analyses", foreign_keys=[job_id])


class StatusEvent(Base):
    __tablename__ = "status_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    from_status: Mapped[str | None] = mapped_column(String(20))
    to_status: Mapped[str] = mapped_column(String(20))
    changed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    note: Mapped[str | None] = mapped_column(Text)

    job: Mapped[Job] = relationship(back_populates="status_events")


# --------------------------------------------------------------------------- ingestion bookkeeping


class SourceRun(Base):
    """One run of one source (a company feed, the mailbox, a manual import)."""

    __tablename__ = "source_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_name: Mapped[str] = mapped_column(String(200), index=True)
    adapter: Mapped[str] = mapped_column(String(40))
    target_company_id: Mapped[int | None] = mapped_column(
        ForeignKey("target_companies.id", ondelete="SET NULL"), index=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    success: Mapped[bool | None] = mapped_column(Boolean)
    jobs_found: Mapped[int] = mapped_column(Integer, default=0)
    jobs_new: Mapped[int] = mapped_column(Integer, default=0)
    jobs_merged: Mapped[int] = mapped_column(Integer, default=0)
    jobs_updated: Mapped[int] = mapped_column(Integer, default=0)
    jobs_deactivated: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    errors: Mapped[list[Any]] = mapped_column(JSON, default=list)


class EmailMessage(Base):
    __tablename__ = "email_messages"
    __table_args__ = (UniqueConstraint("message_id", name="uq_email_messages_message_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[str] = mapped_column(String(500))
    from_addr: Mapped[str | None] = mapped_column(String(300))
    subject: Mapped[str | None] = mapped_column(String(1000))
    received_at: Mapped[datetime | None] = mapped_column(DateTime)
    parser: Mapped[str | None] = mapped_column(String(40))
    jobs_found: Mapped[int] = mapped_column(Integer, default=0)
    processed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    error: Mapped[str | None] = mapped_column(Text)


# --------------------------------------------------------------------------- application material


class ApplicationDocument(Base):
    """One stage of one generated cover letter. Never updated in place: every run adds rows.

    group_id ties the stages of one generation run together:
      draft  → writer output        edited → German style editor output        final → validated version
    """

    __tablename__ = "application_documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    group_id: Mapped[str] = mapped_column(String(32), index=True)
    kind: Mapped[str] = mapped_column(String(30), default="cover_letter")
    stage: Mapped[str] = mapped_column(String(20))  # draft | edited | final
    language: Mapped[str] = mapped_column(String(5))  # de | en
    subject: Mapped[str | None] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text)
    source_document_id: Mapped[int | None] = mapped_column(ForeignKey("application_documents.id", ondelete="SET NULL"))
    provider: Mapped[str | None] = mapped_column(String(40))
    model: Mapped[str | None] = mapped_column(String(200))
    prompt_version: Mapped[str | None] = mapped_column(String(40))
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # evidence, word count, validator model...
    validation: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class PurgedPosting(Base):
    """Tombstone for postings removed by the retention policy, so a still-listed feed posting isn't re-added
    as "new" the next day. Pruned once the posting hasn't been seen in any feed for a while."""

    __tablename__ = "purged_postings"
    __table_args__ = (UniqueConstraint("source_key", name="uq_purged_postings_source_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    source_key: Mapped[str] = mapped_column(String(600))
    purged_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class ApplicationNotes(Base):
    """The applicant's own reasons for one application, written informally in any language.

    Motivation, priorities and caveats for the cover letter — never evidence for skills or employment history
    (that comes only from profile.yaml). One row per job; regenerating a letter never touches it.
    """

    __tablename__ = "application_notes"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), unique=True, index=True)
    why_interested: Mapped[str | None] = mapped_column(Text)
    why_fit: Mapped[str | None] = mapped_column(Text)
    emphasize: Mapped[str | None] = mapped_column(Text)
    avoid: Mapped[str | None] = mapped_column(Text)
    # NONE | AI_WITHOUT_PERSONAL_INPUT | PERSONAL_INPUT_AI_EDITED | MANUAL (see app.enums.CoverLetterStrategy)
    cover_letter_strategy: Mapped[str] = mapped_column(String(40), default="NONE")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class ApplicationPackage(Base):
    """One run of the application agent for one job: tailored CV + cover letter (+ e-mail), reviewed by the
    reviewer agent (ATS + recruiter) and revised until SEND or the round limit. Never submitted automatically.

    The letter itself lives in application_documents (letter_group_id); every review round is kept in `reviews`.
    """

    __tablename__ = "application_packages"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="running")  # running | done | failed
    language: Mapped[str] = mapped_column(String(5))
    verdict: Mapped[str | None] = mapped_column(String(10))  # SEND | REVISE
    ats_score: Mapped[int | None] = mapped_column(Integer)
    recruiter_score: Mapped[float | None] = mapped_column(Float)
    rounds: Mapped[int] = mapped_column(Integer, default=0)
    letter_group_id: Mapped[str | None] = mapped_column(String(32))
    cv: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    email: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    reviews: Mapped[list[Any]] = mapped_column(JSON, default=list)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))  # tool log, timings, research
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
