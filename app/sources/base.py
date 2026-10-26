"""Adapter interface for job sources.

Every source (ATS feed, career page, email alert, manual paste) produces `JobPosting` objects.
The ingestion pipeline (app.services.ingest) takes it from there: normalise → dedup → filter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from app.core.http import HttpClient
from app.enums import SourceType


class JobPosting(BaseModel):
    """Source-agnostic posting, as produced by an adapter/parser. Text fields are plain text."""

    source: str
    source_type: SourceType
    external_job_id: str | None = None
    url: str | None = None
    title: str
    company: str = ""
    location: str = ""
    country: str | None = None
    remote_status: str | None = None
    description: str = ""
    requirements: str = ""
    responsibilities: str = ""
    salary_min: float | None = None
    salary_max: float | None = None
    currency: str | None = None
    employment_type: str | None = None
    seniority: str | None = None
    sector: str | None = None
    department: str | None = None
    career_page_url: str | None = None
    date_posted: datetime | None = None
    track: str = "main"  # "side": part-time / mini-job / temporary work (see app/services/side_jobs.py)
    raw: dict[str, Any] = Field(default_factory=dict)


@dataclass
class CompanyFeedConfig:
    """What an adapter needs to know about a target company (decoupled from the ORM)."""

    company_name: str
    identifier: str | None = None  # board token / company slug
    careers_url: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    country: str | None = None
    city: str | None = None


class JobSourceAdapter(ABC):
    """Base class for company career feeds (ATS APIs, career pages)."""

    name: ClassVar[str]
    source_type: ClassVar[SourceType] = SourceType.EMPLOYER
    # A search over a job board returns a sample, not the complete list: a posting missing from one run is not
    # closed, so it is not deactivated (retention removes old ones).
    partial_listing: ClassVar[bool] = False

    def __init__(self, company: CompanyFeedConfig, http: HttpClient) -> None:
        self.company = company
        self.http = http

    def source_name(self) -> str:
        return self.name

    @abstractmethod
    def fetch_jobs(self) -> list[dict[str, Any]]:
        """Return raw postings (cheap listing call). Raise on failure."""

    @abstractmethod
    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        """Map one raw posting to a JobPosting."""

    def external_id(self, raw: dict[str, Any]) -> str | None:
        """Stable id within this source; lets the pipeline skip detail fetches for known jobs."""
        value = raw.get("id")
        return str(value) if value is not None else None

    def location_hint(self, raw: dict[str, Any]) -> str:
        """Location text available in the cheap listing data ("" = unknown until normalised). Lets the pipeline skip
        postings outside your target countries before any detail request."""
        return ""

    def fetch_details(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Optional second call for full details — only invoked for postings not seen before."""
        return raw

    def _fallback_location(self) -> str:
        return ", ".join(x for x in (self.company.city, self.company.country) if x)
