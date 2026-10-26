"""Careerjet (official job search API v4) — an aggregator that indexes Bayt, GulfTalent, NaukriGulf, company sites
and more, in the UAE (careerjet.ae) and Saudi Arabia (careerjet.com.sa). The legitimate way to see those boards'
postings without scraping them.

Needs a free publisher API key: register at https://www.careerjet.com/partners/register/as-publisher and put it
in .env as CAREERJET_API_KEY. Without a key the source returns nothing (no error).

ats_config:
  queries:  search terms (default: your target roles)
  locales:  Careerjet locale codes (default: en_AE, en_SA)
  max_pages: pages of 100 per query (default 2)
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime
from typing import Any

from app.core.text import clean_line, html_to_text
from app.enums import SourceType
from app.sources.base import JobPosting, JobSourceAdapter
from app.sources.workable_board import DEFAULT_QUERIES

log = logging.getLogger(__name__)

API = "https://search.api.careerjet.net/v4/query"
COUNTRY = {"AE": "UAE", "SA": "Saudi Arabia", "DE": "Germany", "LU": "Luxembourg", "GB": "United Kingdom"}


class CareerjetAdapter(JobSourceAdapter):
    name = "careerjet"
    source_type = SourceType.AGGREGATOR
    partial_listing = True

    def _key(self) -> str | None:
        from app.settings import get_settings

        key = get_settings().careerjet_api_key
        return key.get_secret_value() if key else None

    def fetch_jobs(self) -> list[dict[str, Any]]:
        from app.settings import get_settings

        key = self._key()
        if not key:
            log.info("Careerjet: no CAREERJET_API_KEY in .env — source skipped")
            return []
        auth = {"Authorization": "Basic " + base64.b64encode(f"{key}:".encode()).decode()}
        opts = self.company.options
        out: dict[str, dict[str, Any]] = {}
        for locale in opts.get("locales") or ["en_AE", "en_SA", "de_DE", "en_GB"]:
            for query in opts.get("queries") or _queries():
                for page in range(1, int(opts.get("max_pages", 2)) + 1):
                    data = self.http.get_json_with_headers(API, headers=auth, params={
                        "locale_code": locale, "keywords": query, "sort": "date", "page": page, "page_size": 100,
                        "fragment_size": 1000, "user_ip": get_settings().careerjet_user_ip,
                        "user_agent": get_settings().http_user_agent})
                    jobs = data.get("jobs") or []
                    for job in jobs:
                        if job.get("url"):
                            out.setdefault(job["url"], {**job, "_locale": locale})
                    if len(jobs) < 100 or page >= int(data.get("pages") or 0):
                        break
        return list(out.values())

    def external_id(self, raw: dict[str, Any]) -> str | None:
        return raw.get("url")

    def location_hint(self, raw: dict[str, Any]) -> str:
        return raw.get("locations") or COUNTRY.get((raw.get("_locale") or "")[-2:], "")

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        posted = None
        try:
            posted = datetime.strptime(raw.get("date") or "", "%a, %d %b %Y %H:%M:%S %Z")
        except ValueError:
            pass
        country = COUNTRY.get((raw.get("_locale") or "")[-2:])
        return JobPosting(
            source=self.name, source_type=self.source_type, url=raw.get("url"), external_job_id=raw.get("url"),
            title=clean_line(raw.get("title")), company=clean_line(raw.get("company")),
            location=clean_line(raw.get("locations") or country or ""), country=country,
            description=html_to_text(raw.get("description")),
            salary_min=raw.get("salary_min") or None, salary_max=raw.get("salary_max") or None,
            currency=raw.get("salary_currency_code") or None, date_posted=posted,
            raw={"locale": raw.get("_locale"), "thin": True},
        )


def _queries() -> list[str]:
    """Search terms: the source's own `queries`, else preferences.yaml → search.queries."""
    from app.config_files import get_prefs

    return list(get_prefs().search.queries) or DEFAULT_QUERIES
