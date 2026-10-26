"""Reed.co.uk (official Jobseeker API) — one of the UK's largest job boards.

Needs a free key: register at https://www.reed.co.uk/developers/jobseeker and put it in .env as REED_API_KEY
(HTTP basic auth: the key is the user name, the password is empty). Without a key the source returns nothing.
Search: GET /api/1.0/search (100 per page); full description: GET /api/1.0/jobs/{id} — only for new postings.

ats_config:
  queries:      search terms (default: preferences.yaml → search.queries)
  locationName: e.g. "London" (default: the whole UK)
  max_pages:    pages of 100 per query (default 1 — the newest)
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime
from typing import Any

from app.core.text import clean_line, html_to_text
from app.enums import SourceType
from app.sources.base import JobPosting, JobSourceAdapter

log = logging.getLogger(__name__)

API = "https://www.reed.co.uk/api/1.0"


def _key() -> str | None:
    from app.settings import get_settings

    k = get_settings().reed_api_key
    return k.get_secret_value() if k else None


class ReedAdapter(JobSourceAdapter):
    name = "reed"
    source_type = SourceType.JOB_BOARD
    partial_listing = True

    def _auth(self) -> dict[str, str]:
        return {"Authorization": "Basic " + base64.b64encode(f"{_key()}:".encode()).decode()}

    def fetch_jobs(self) -> list[dict[str, Any]]:
        from app.sources.workable_board import _queries

        if not _key():
            log.info("Reed: no REED_API_KEY in .env — source skipped")
            return []
        opts = self.company.options
        out: dict[str, dict[str, Any]] = {}
        for query in opts.get("queries") or _queries():
            for page in range(int(opts.get("max_pages", 1))):
                params = {"keywords": query, "resultsToTake": 100, "resultsToSkip": page * 100}
                if opts.get("locationName"):
                    params["locationName"] = opts["locationName"]
                data = self.http.get_json_with_headers(f"{API}/search", params=params, headers=self._auth())
                items = data.get("results") or []
                for item in items:
                    out.setdefault(str(item.get("jobId")), item)
                if len(items) < 100:
                    break
        return list(out.values())

    def external_id(self, raw: dict[str, Any]) -> str | None:
        return str(raw.get("jobId") or "") or None

    def location_hint(self, raw: dict[str, Any]) -> str:
        return f"{raw.get('locationName') or ''}, United Kingdom"

    def fetch_details(self, raw: dict[str, Any]) -> dict[str, Any]:
        try:
            details = self.http.get_json_with_headers(f"{API}/jobs/{raw['jobId']}", headers=self._auth())
        except Exception:  # noqa: BLE001 - the search result (with its snippet) is still a posting
            return raw
        return {**raw, **{k: v for k, v in (details or {}).items() if v not in (None, "")}}

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        posted = None
        for key in ("datePosted", "date"):
            try:
                posted = datetime.strptime((raw.get(key) or "")[:10], "%d/%m/%Y")
                break
            except ValueError:
                continue
        return JobPosting(
            source=self.name, source_type=self.source_type, external_job_id=self.external_id(raw),
            url=raw.get("jobUrl") or f"https://www.reed.co.uk/jobs/{raw.get('jobId')}",
            title=clean_line(raw.get("jobTitle")), company=clean_line(raw.get("employerName")),
            location=clean_line(self.location_hint(raw)), country="United Kingdom",
            description=html_to_text(raw.get("jobDescription")),
            salary_min=raw.get("minimumSalary") or None, salary_max=raw.get("maximumSalary") or None,
            currency=raw.get("currency") or ("GBP" if raw.get("minimumSalary") else None),
            employment_type=raw.get("contractType") or None, date_posted=posted,
            raw={"applications": raw.get("applications")},
        )
