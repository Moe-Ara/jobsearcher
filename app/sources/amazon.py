"""Amazon jobs (amazon.jobs search JSON — the data behind the public job search).

ats_config:
  countries:  ISO-3 country codes (default: your target countries → DEU, ARE, SAU, LUX)
  categories: amazon.jobs category slugs (default: engineering, science and solutions roles — warehouse,
              facilities and sales roles never enter the queue)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.core.text import clean_line, html_to_text
from app.sources.base import JobPosting, JobSourceAdapter

ISO3 = {"Germany": "DEU", "UAE": "ARE", "Saudi Arabia": "SAU", "Luxembourg": "LUX", "Austria": "AUT",
        "Switzerland": "CHE", "Netherlands": "NLD", "Qatar": "QAT"}
DEFAULT_CATEGORIES = ["software-development", "machine-learning-science", "data-science", "research-science",
                      "solutions-architect", "systems-quality-security-engineering", "hardware-development",
                      "operations-it-support-engineering"]
PAGE = 100


class AmazonJobsAdapter(JobSourceAdapter):
    name = "amazon"

    def _countries(self) -> list[str]:
        codes = self.company.options.get("countries")
        if codes:
            return list(codes)
        from app.config_files import get_prefs

        return [ISO3[c] for c in get_prefs().locations.target_countries if c in ISO3]

    def fetch_jobs(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        categories = self.company.options.get("categories") or DEFAULT_CATEGORIES
        for code in self._countries():
            for offset in range(0, int(self.company.options.get("max_jobs", 600)), PAGE):
                params: list[tuple[str, Any]] = [("offset", offset), ("result_limit", PAGE),
                                                 ("normalized_country_code[]", code), ("sort", "recent")]
                params += [("category[]", c) for c in categories]
                data = self.http.get_json("https://www.amazon.jobs/en/search.json", params=params)
                jobs = data.get("jobs") or []
                out += jobs
                if len(jobs) < PAGE:
                    break
        return out

    def external_id(self, raw: dict[str, Any]) -> str | None:
        return str(raw.get("id_icims") or raw.get("id") or "") or None

    def location_hint(self, raw: dict[str, Any]) -> str:
        return raw.get("normalized_location") or raw.get("location") or ""

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        posted = None
        try:
            posted = datetime.strptime(raw.get("posted_date") or "", "%B %d, %Y")
        except ValueError:
            pass
        quals = "\n".join(x for x in (html_to_text(raw.get("basic_qualifications")),
                                      html_to_text(raw.get("preferred_qualifications"))) if x)
        return JobPosting(
            source=self.name, source_type=self.source_type, external_job_id=self.external_id(raw),
            url=f"https://www.amazon.jobs{raw.get('job_path', '')}", title=clean_line(raw.get("title")),
            company=self.company.company_name or "Amazon", location=clean_line(self.location_hint(raw)),
            description=html_to_text(raw.get("description")), requirements=quals,
            department=raw.get("job_category"), career_page_url="https://www.amazon.jobs", date_posted=posted,
            raw={k: raw.get(k) for k in ("id_icims", "job_category", "team", "business_category", "is_intern")},
        )
