"""Arbeitnow (arbeitnow.com) — a free public job API focused on Germany and Europe (300 newest jobs per page, updated
hourly). No key; its terms ask not to abuse it and to link back — postings link to arbeitnow.com.

ats_config:
  max_pages: newest pages to read (default 3 = 900 postings; jobs outside your target countries are dropped)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.core.text import clean_line, html_to_text
from app.enums import SourceType
from app.sources.base import JobPosting, JobSourceAdapter

API = "https://www.arbeitnow.com/api/job-board-api"
# never looked up as a town: work modes and regions
_NOT_A_TOWN = re.compile(r"remote|home ?office|hybrid|emea|europe|europa|worldwide|anywhere|dach|international|various", re.I)


class ArbeitnowAdapter(JobSourceAdapter):
    name = "arbeitnow"
    source_type = SourceType.JOB_BOARD
    partial_listing = True  # a window of the newest postings, not the complete board

    def fetch_jobs(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for page in range(1, int(self.company.options.get("max_pages", 3)) + 1):
            data = self.http.get_json(API, params={"page": page})
            items = data.get("data") or []
            out += items
            if not items or not (data.get("links") or {}).get("next"):
                break
        self._resolve_countries(out)
        return out

    def _resolve_countries(self, items: list[dict[str, Any]]) -> None:
        """Arbeitnow gives a bare town ("Breda", "Zug", "Emstek"): look up the country of towns the built-in list
        doesn't know (cached), so postings outside your countries are dropped before they are stored."""
        from app.core.normalize import infer_country
        from app.services.drive_time import countries_of_towns

        unknown = [i["location"] for i in items if i.get("location") and not i.get("remote") and not _NOT_A_TOWN.search(i["location"])
                   and not infer_country(i["location"])]
        found = countries_of_towns(unknown, self.http) if unknown else {}
        for i in items:
            i["_country"] = found.get((i.get("location") or "").strip())

    def external_id(self, raw: dict[str, Any]) -> str | None:
        return raw.get("slug")

    def location_hint(self, raw: dict[str, Any]) -> str:
        return ", ".join(x for x in (raw.get("location"), raw.get("_country")) if x)

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        created = raw.get("created_at")
        posted = datetime.fromtimestamp(created, tz=timezone.utc).replace(tzinfo=None) if isinstance(created, (int, float)) else None
        return JobPosting(
            source=self.name, source_type=self.source_type, external_job_id=raw.get("slug"), url=raw.get("url"),
            title=clean_line(raw.get("title")), company=clean_line(raw.get("company_name")),
            location=clean_line(raw.get("location")), country=raw.get("_country"),
            remote_status="remote" if raw.get("remote") else None,
            description=html_to_text(raw.get("description")), employment_type=", ".join(raw.get("job_types") or []) or None,
            date_posted=posted, raw={"tags": raw.get("tags") or []},
        )
