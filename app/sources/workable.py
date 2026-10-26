"""Workable public widget API (the data behind apply.workable.com/<account>)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.core.text import clean_line, html_to_text
from app.sources.base import JobPosting, JobSourceAdapter


def _parse_date(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value[:10]) if value else None
    except ValueError:
        return None


class WorkableAdapter(JobSourceAdapter):
    name = "workable"

    def fetch_jobs(self) -> list[dict[str, Any]]:
        account = self.company.identifier
        if not account:
            raise ValueError("Workable adapter needs ats_identifier (account name)")
        data = self.http.get_json(f"https://apply.workable.com/api/v1/widget/accounts/{account}", params={"details": "true"})
        return list(data.get("jobs") or [])

    def external_id(self, raw: dict[str, Any]) -> str | None:
        return raw.get("shortcode") or raw.get("code")

    def location_hint(self, raw: dict[str, Any]) -> str:
        locs = raw.get("locations") or [{"city": raw.get("city"), "country": raw.get("country")}]
        return "; ".join(", ".join(x for x in (loc.get("city"), loc.get("country")) if x) for loc in locs)

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        return JobPosting(
            source=self.name, source_type=self.source_type, external_job_id=self.external_id(raw),
            url=raw.get("url") or raw.get("shortlink"), title=clean_line(raw.get("title")),
            company=self.company.company_name, location=clean_line(self.location_hint(raw)) or self._fallback_location(),
            remote_status="remote" if raw.get("telecommuting") else None,
            description=html_to_text(raw.get("description")), employment_type=raw.get("employment_type"),
            department=raw.get("department"), career_page_url=self.company.careers_url,
            date_posted=_parse_date(raw.get("published_on") or raw.get("created_at")),
            raw={k: v for k, v in raw.items() if k != "description"},
        )
