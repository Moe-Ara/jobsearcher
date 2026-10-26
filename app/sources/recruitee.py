"""Recruitee public careers API: https://<company>.recruitee.com/api/offers/ (also behind custom career domains)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.core.text import clean_line, html_to_text
from app.sources.base import JobPosting, JobSourceAdapter


def _parse_dt(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace(" UTC", "").replace("Z", "")[:19]) if value else None
    except ValueError:
        return None


class RecruiteeAdapter(JobSourceAdapter):
    name = "recruitee"

    def _base(self) -> str:
        host = self.company.options.get("host")  # custom domain, e.g. careers.instagrid.co
        if host:
            return f"https://{host}"
        if not self.company.identifier:
            raise ValueError("Recruitee adapter needs ats_identifier (company subdomain) or ats_config.host")
        return f"https://{self.company.identifier}.recruitee.com"

    def fetch_jobs(self) -> list[dict[str, Any]]:
        return list(self.http.get_json(f"{self._base()}/api/offers/").get("offers") or [])

    def location_hint(self, raw: dict[str, Any]) -> str:
        locs = raw.get("locations") or []
        if locs:
            return "; ".join(", ".join(x for x in (loc.get("city"), loc.get("country")) if x) for loc in locs)
        return ", ".join(x for x in (raw.get("city"), raw.get("country")) if x)

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        return JobPosting(
            source=self.name, source_type=self.source_type, external_job_id=str(raw.get("id")),
            url=raw.get("careers_url"), title=clean_line(raw.get("title")), company=self.company.company_name,
            location=clean_line(self.location_hint(raw)) or self._fallback_location(),
            remote_status="remote" if raw.get("remote") else None,
            description=html_to_text(raw.get("description")), requirements=html_to_text(raw.get("requirements")),
            employment_type=raw.get("employment_type_code"), department=raw.get("department"),
            career_page_url=self.company.careers_url, date_posted=_parse_dt(raw.get("published_at")),
            raw={k: v for k, v in raw.items() if k not in ("description", "requirements")},
        )
