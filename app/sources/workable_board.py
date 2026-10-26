"""Workable job board (jobs.workable.com) — the public search across every company that hires through Workable.

Many Gulf employers (Salla, Foodics, Mozn, Moyasar, Bayut | dubizzle, Callsign, …) post there; one search per
role and country reaches all of them. Uses the JSON API behind the board's own search page (full descriptions,
structured locations, page tokens) — no scraping of HTML, nothing that robots.txt disallows.

ats_config:
  queries:      search terms (default: your target roles)
  locations:    country names as the board writes them (default: United Arab Emirates, Saudi Arabia)
  max_pages:    pages of 20 per query and country (default 3)
  max_age_days: skip postings created longer ago (default 60 — the board keeps old postings online)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.text import clean_line, html_to_text
from app.enums import SourceType
from app.sources.base import JobPosting, JobSourceAdapter

API = "https://jobs.workable.com/api/v1/jobs"
DEFAULT_QUERIES = ["software engineer", "machine learning", "AI engineer", "data scientist", "data engineer",
                   "backend developer", "full stack developer", "python developer", "computer vision", "robotics",
                   "devops engineer", "MLOps"]
DEFAULT_LOCATIONS = ["United Arab Emirates", "Saudi Arabia", "Germany", "United Kingdom"]


def _created(raw: dict[str, Any]) -> datetime | None:
    try:
        return datetime.fromisoformat((raw.get("created") or "").replace("Z", "+00:00"))
    except ValueError:
        return None


class WorkableBoardAdapter(JobSourceAdapter):
    name = "workable_board"
    source_type = SourceType.JOB_BOARD
    partial_listing = True

    def fetch_jobs(self) -> list[dict[str, Any]]:
        opts = self.company.options
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(opts.get("max_age_days", 60)))
        out: dict[str, dict[str, Any]] = {}
        for location in opts.get("locations") or DEFAULT_LOCATIONS:
            for query in opts.get("queries") or _queries():
                token = None
                for _ in range(int(opts.get("max_pages", 3))):
                    params = {"query": query, "location": location, **({"pageToken": token} if token else {})}
                    data = self.http.get_json(API, params=params)
                    for job in data.get("jobs") or []:
                        created = _created(job)
                        if job.get("id") and (created is None or created >= cutoff):
                            out.setdefault(job["id"], job)
                    token = data.get("nextPageToken")
                    if not token or not data.get("jobs"):
                        break
        return list(out.values())

    def external_id(self, raw: dict[str, Any]) -> str | None:
        return raw.get("id")

    def location_hint(self, raw: dict[str, Any]) -> str:
        loc = raw.get("location") or {}
        return ", ".join(x for x in (loc.get("city"), loc.get("countryName")) if x)

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        loc = raw.get("location") or {}
        places = raw.get("locations") or []
        workplace = (raw.get("workplace") or "").lower()
        remote = "remote" if workplace == "remote" or "TELECOMMUTE" in places else "hybrid" if workplace == "hybrid" else (
            "onsite" if workplace == "on_site" else None)
        company = raw.get("company") or {}
        created = _created(raw)
        return JobPosting(
            source=self.name, source_type=self.source_type, external_job_id=raw.get("id"), url=raw.get("url"),
            title=clean_line(raw.get("title")), company=clean_line(company.get("title")),
            location=clean_line(self.location_hint(raw)), country=loc.get("countryName") or None, remote_status=remote,
            description=html_to_text(raw.get("description")), requirements=html_to_text(raw.get("requirementsSection")),
            employment_type=raw.get("employmentType"), department=raw.get("department"),
            career_page_url=company.get("url"), date_posted=created.replace(tzinfo=None) if created else None,
            raw={"company_website": company.get("website"), "workplace": raw.get("workplace"), "locations": places},
        )


def _queries() -> list[str]:
    """Search terms: the source's own `queries`, else preferences.yaml → search.queries."""
    from app.config_files import get_prefs

    return list(get_prefs().search.queries) or DEFAULT_QUERIES
