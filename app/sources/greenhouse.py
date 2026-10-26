"""Greenhouse public Job Board API: https://developers.greenhouse.io/job-board.html"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.core.text import clean_line, html_to_text
from app.sources.base import JobPosting, JobSourceAdapter


def _api_base(options: dict[str, Any]) -> str:
    if options.get("region") == "eu":
        return "https://boards-api.eu.greenhouse.io/v1/boards"
    return "https://boards-api.greenhouse.io/v1/boards"


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


class GreenhouseAdapter(JobSourceAdapter):
    name = "greenhouse"

    def fetch_jobs(self) -> list[dict[str, Any]]:
        token = self.company.identifier
        if not token:
            raise ValueError("Greenhouse adapter needs ats_identifier (board token)")
        data = self.http.get_json(f"{_api_base(self.company.options)}/{token}/jobs", params={"content": "true"})
        return list(data.get("jobs") or [])

    def fetch_single(self, job_id: str) -> dict[str, Any]:
        return self.http.get_json(
            f"{_api_base(self.company.options)}/{self.company.identifier}/jobs/{job_id}"
        )

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        loc = (raw.get("location") or {}).get("name") or ""
        offices = ", ".join(o.get("name", "") for o in raw.get("offices") or [] if o.get("name"))
        departments = ", ".join(d.get("name", "") for d in raw.get("departments") or [] if d.get("name"))
        return JobPosting(
            source=self.name,
            source_type=self.source_type,
            external_job_id=str(raw.get("id")),
            url=raw.get("absolute_url"),
            title=clean_line(raw.get("title")),
            company=self.company.company_name or clean_line(raw.get("company_name")),
            location=clean_line(loc or offices) or self._fallback_location(),
            description=html_to_text(raw.get("content")),
            department=departments or None,
            career_page_url=self.company.careers_url,
            date_posted=_parse_dt(raw.get("first_published") or raw.get("updated_at")),
            raw={k: v for k, v in raw.items() if k != "content"},
        )
