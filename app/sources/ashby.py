"""Ashby public job board API: https://developers.ashbyhq.com/docs/public-job-posting-api"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.core.text import clean_line, html_to_text
from app.sources.base import JobPosting, JobSourceAdapter


def _parse_dt(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None) if value else None
    except ValueError:
        return None


class AshbyAdapter(JobSourceAdapter):
    name = "ashby"

    def fetch_jobs(self) -> list[dict[str, Any]]:
        board = self.company.identifier
        if not board:
            raise ValueError("Ashby adapter needs ats_identifier (job board name)")
        data = self.http.get_json(f"https://api.ashbyhq.com/posting-api/job-board/{board}")
        return [j for j in data.get("jobs") or [] if j.get("isListed", True)]

    def location_hint(self, raw: dict[str, Any]) -> str:
        addr = ((raw.get("address") or {}).get("postalAddress") or {})
        return ", ".join(x for x in (raw.get("location"), addr.get("addressLocality"), addr.get("addressCountry")) if x)

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        addr = ((raw.get("address") or {}).get("postalAddress") or {})
        locations = [raw.get("location") or ""] + [s.get("location", "") for s in raw.get("secondaryLocations") or []]
        location = ", ".join(dict.fromkeys(x for x in [*locations, addr.get("addressCountry")] if x))
        return JobPosting(
            source=self.name, source_type=self.source_type, external_job_id=str(raw.get("id")),
            url=raw.get("jobUrl"), title=clean_line(raw.get("title")), company=self.company.company_name,
            location=clean_line(location) or self._fallback_location(),
            remote_status="remote" if raw.get("isRemote") or raw.get("workplaceType") == "Remote" else None,
            description=raw.get("descriptionPlain") or html_to_text(raw.get("descriptionHtml")),
            employment_type=raw.get("employmentType"), department=raw.get("department") or raw.get("team"),
            career_page_url=self.company.careers_url, date_posted=_parse_dt(raw.get("publishedAt")),
            raw={k: v for k, v in raw.items() if k not in ("descriptionHtml", "descriptionPlain")},
        )
