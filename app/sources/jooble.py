"""Jooble (job search API) — an aggregator with UAE (ae.jooble.org) and Saudi (sa.jooble.org) coverage, indexing
the large Gulf boards and company sites.

Needs a free API key: request it at https://jooble.org/api/about and put it in .env as JOOBLE_API_KEY. Without a
key the source returns nothing (no error). Request format: POST https://jooble.org/api/<key> with
{"keywords", "location", "page"} → {"totalCount", "jobs": [{title, location, snippet, salary, source, type, link,
company, updated, id}]}.

ats_config:
  queries:   search terms (default: your target roles)
  locations: location strings (default: United Arab Emirates, Saudi Arabia)
  max_pages: pages per query (default 2)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from app.core.text import clean_line, html_to_text
from app.enums import SourceType
from app.sources.base import JobPosting, JobSourceAdapter
from app.sources.workable_board import DEFAULT_LOCATIONS, DEFAULT_QUERIES

log = logging.getLogger(__name__)


class JoobleAdapter(JobSourceAdapter):
    name = "jooble"
    source_type = SourceType.AGGREGATOR
    partial_listing = True

    def fetch_jobs(self) -> list[dict[str, Any]]:
        from app.settings import get_settings

        key = get_settings().jooble_api_key
        if not key:
            log.info("Jooble: no JOOBLE_API_KEY in .env — source skipped")
            return []
        opts = self.company.options
        url = f"https://jooble.org/api/{key.get_secret_value()}"
        out: dict[str, dict[str, Any]] = {}
        for location in opts.get("locations") or DEFAULT_LOCATIONS:
            for query in opts.get("queries") or _queries():
                for page in range(1, int(opts.get("max_pages", 2)) + 1):
                    try:
                        data = self.http.post_json(url, {"keywords": query, "location": location, "page": str(page)}) or {}
                    except Exception as exc:  # the key is part of the URL: never let it reach logs / the UI
                        raise RuntimeError(str(exc).replace(key.get_secret_value(), "***")) from None
                    jobs = data.get("jobs") or []
                    for job in jobs:
                        ident = str(job.get("id") or job.get("link") or "")
                        if ident:
                            out.setdefault(ident, {**job, "_location": location})
                    if not jobs:
                        break
        return list(out.values())

    def external_id(self, raw: dict[str, Any]) -> str | None:
        return str(raw.get("id") or raw.get("link") or "") or None

    def location_hint(self, raw: dict[str, Any]) -> str:
        return raw.get("location") or raw.get("_location") or ""

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        posted = None
        try:
            posted = datetime.fromisoformat((raw.get("updated") or "")[:19])
        except ValueError:
            pass
        where = raw.get("_location") or ""
        country = {"emirates": "UAE", "saudi": "Saudi Arabia", "germany": "Germany", "deutschland": "Germany",
                   "united kingdom": "United Kingdom"}.get(next((k for k in ("emirates", "saudi", "germany", "deutschland",
                                                                            "united kingdom") if k in where.lower()), ""))
        return JobPosting(
            source=self.name, source_type=self.source_type, external_job_id=self.external_id(raw), url=raw.get("link"),
            title=clean_line(raw.get("title")), company=clean_line(raw.get("company")),
            location=clean_line(raw.get("location") or where), country=country,
            description=html_to_text(raw.get("snippet")), employment_type=raw.get("type") or None, date_posted=posted,
            raw={"board": raw.get("source"), "salary": raw.get("salary"), "thin": True},
        )


def _queries() -> list[str]:
    """Search terms: the source's own `queries`, else preferences.yaml → search.queries."""
    from app.config_files import get_prefs

    return list(get_prefs().search.queries) or DEFAULT_QUERIES
