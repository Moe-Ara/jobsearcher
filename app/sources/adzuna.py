"""Adzuna (official job search API) — a large aggregator for the UK (gb) and Germany (de), among others.

Needs a free app id + key: register at https://developer.adzuna.com and put them in .env as ADZUNA_APP_ID and
ADZUNA_APP_KEY. Without them the source returns nothing. The free tier is rate-limited (a few hundred calls a day):
keep queries × countries × pages small. Adzuna returns a description excerpt; the enrichment step finds the full
posting later where possible.

ats_config:
  countries:    Adzuna country codes (default: gb, de)
  queries:      search terms (default: preferences.yaml → search.queries)
  max_days_old: only postings this recent (default 7)
  max_pages:    pages of 50 per query and country (default 1)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from app.core.text import clean_line, html_to_text
from app.enums import SourceType
from app.sources.base import JobPosting, JobSourceAdapter

log = logging.getLogger(__name__)

API = "https://api.adzuna.com/v1/api/jobs"
COUNTRY = {"gb": "United Kingdom", "de": "Germany", "at": "Austria", "ch": "Switzerland", "nl": "Netherlands",
           "fr": "France", "us": "United States"}


class AdzunaAdapter(JobSourceAdapter):
    name = "adzuna"
    source_type = SourceType.AGGREGATOR
    partial_listing = True

    def fetch_jobs(self) -> list[dict[str, Any]]:
        from app.settings import get_settings
        from app.sources.workable_board import _queries

        s = get_settings()
        if not (s.adzuna_app_id and s.adzuna_app_key):
            log.info("Adzuna: no ADZUNA_APP_ID / ADZUNA_APP_KEY in .env — source skipped")
            return []
        opts = self.company.options
        key = s.adzuna_app_key.get_secret_value()
        out: dict[str, dict[str, Any]] = {}
        for cc in opts.get("countries") or ["gb", "de"]:
            for query in opts.get("queries") or _queries():
                for page in range(1, int(opts.get("max_pages", 1)) + 1):
                    try:
                        data = self.http.get_json(f"{API}/{cc}/search/{page}", params={
                            "app_id": s.adzuna_app_id, "app_key": key, "what": query, "results_per_page": 50,
                            "max_days_old": int(opts.get("max_days_old", 7)), "sort_by": "date",
                            "content-type": "application/json"})
                    except Exception as exc:  # the key is in the URL: never let it reach logs / the UI
                        raise RuntimeError(str(exc).replace(key, "***")) from None
                    items = data.get("results") or []
                    for item in items:
                        out.setdefault(str(item.get("id")), {**item, "_cc": cc})
                    if len(items) < 50:
                        break
        return list(out.values())

    def external_id(self, raw: dict[str, Any]) -> str | None:
        return str(raw.get("id") or "") or None

    def location_hint(self, raw: dict[str, Any]) -> str:
        loc = raw.get("location") or {}
        return loc.get("display_name") or COUNTRY.get(raw.get("_cc", ""), "")

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        posted = None
        try:
            posted = datetime.fromisoformat((raw.get("created") or "").replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass
        return JobPosting(
            source=self.name, source_type=self.source_type, external_job_id=self.external_id(raw),
            url=raw.get("redirect_url"), title=clean_line(html_to_text(raw.get("title"))),
            company=clean_line((raw.get("company") or {}).get("display_name")), location=clean_line(self.location_hint(raw)),
            country=COUNTRY.get(raw.get("_cc", "")), description=html_to_text(raw.get("description")),
            salary_min=raw.get("salary_min") or None, salary_max=raw.get("salary_max") or None,
            employment_type=raw.get("contract_time") or raw.get("contract_type") or None, date_posted=posted,
            raw={"category": (raw.get("category") or {}).get("label"), "thin": True},
        )
