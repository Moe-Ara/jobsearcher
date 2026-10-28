"""Getro job boards — the portfolio / ecosystem boards of VCs and tech hubs (Hub71, BECO Capital, MEVP, …).

ats_identifier: the board's Getro collection id (the "network" id in the board page's data)
ats_config:
  employer_from_posting: true   (each job belongs to a portfolio company, not to the board)
  query:    optional search text(s) (default: everything; the country gate drops jobs outside your countries)
  max_jobs: cap (default 1500)
Listings have no description — the title, company, skills and seniority are enough for the screening; the job page
link goes to the company's own career site.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.core.text import clean_line
from app.enums import SourceType
from app.sources.base import JobPosting, JobSourceAdapter

PAGE = 20


class GetroAdapter(JobSourceAdapter):
    name = "getro"
    source_type = SourceType.AGGREGATOR

    def fetch_jobs(self) -> list[dict[str, Any]]:
        collection = self.company.identifier
        if not collection:
            raise ValueError("Getro adapter needs ats_identifier (collection id)")
        queries = self.company.options.get("query") or [""]
        queries = [queries] if isinstance(queries, str) else list(queries)
        cap = int(self.company.options.get("max_jobs", 1500))
        out, seen = [], set()
        for query in queries:
            for page in range(0, cap // PAGE + 1):
                data = self.http.post_json(f"https://api.getro.com/api/v2/collections/{collection}/search/jobs",
                                           {"hitsPerPage": PAGE, "page": page, "filters": {}, "query": query})
                jobs = (data.get("results") or {}).get("jobs") or []
                for j in jobs:
                    if j.get("id") not in seen:
                        seen.add(j.get("id"))
                        out.append(j)
                if not jobs:  # the API serves at most 20 per page whatever is asked for: read until a page is empty
                    break
            else:
                self.truncated = True  # stopped at max_jobs
        return out

    def location_hint(self, raw: dict[str, Any]) -> str:
        return "; ".join(raw.get("locations") or [])

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        org = raw.get("organization") or {}
        skills = ", ".join(raw.get("skills") or [])
        created = raw.get("created_at")
        details = [f"Skills: {skills}" if skills else "", f"Seniority: {raw.get('seniority')}" if raw.get("seniority") else "",
                   f"Work mode: {raw.get('work_mode')}" if raw.get("work_mode") else "",
                   f"Listed on {self.company.company_name}" if self.company.company_name else ""]
        return JobPosting(
            source=self.name, source_type=self.source_type, external_job_id=str(raw.get("id")), url=raw.get("url"),
            title=clean_line(raw.get("title")), company=clean_line(org.get("name")) or self.company.company_name,
            location=clean_line(self.location_hint(raw)),
            remote_status="remote" if raw.get("work_mode") == "remote" else ("hybrid" if raw.get("work_mode") == "hybrid" else None),
            description="\n".join(d for d in details if d), seniority=raw.get("seniority"),
            career_page_url=self.company.careers_url,
            date_posted=datetime.fromtimestamp(created) if isinstance(created, (int, float)) else None,
            raw={"organization": org.get("name"), "skills": raw.get("skills"), "board": self.company.company_name},
        )
