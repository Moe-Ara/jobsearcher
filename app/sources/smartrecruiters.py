"""SmartRecruiters public Posting API: https://developers.smartrecruiters.com/docs/posting-api"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.core.text import clean_line, html_to_text
from app.sources.base import JobPosting, JobSourceAdapter

API = "https://api.smartrecruiters.com/v1/companies"


class SmartRecruitersAdapter(JobSourceAdapter):
    name = "smartrecruiters"

    def fetch_jobs(self) -> list[dict[str, Any]]:
        cid = self.company.identifier
        if not cid:
            raise ValueError("SmartRecruiters adapter needs ats_identifier (company id)")
        out: list[dict[str, Any]] = []
        offset, limit = 0, 100
        params: dict[str, Any] = {"limit": limit}
        # Large employers have thousands of postings; optional server-side filters keep it sane.
        for key in ("country", "q", "city", "department"):
            if self.company.options.get(key):
                params[key] = self.company.options[key]
        max_jobs = int(self.company.options.get("max_jobs", 1000))
        while True:
            params["offset"] = offset
            page = self.http.get_json(f"{API}/{cid}/postings", params=params)
            content = page.get("content") or []
            out.extend(content)
            offset += limit
            if not content or offset >= int(page.get("totalFound", 0)) or len(out) >= max_jobs:
                return out[:max_jobs]

    def fetch_details(self, raw: dict[str, Any]) -> dict[str, Any]:
        detail = self.http.get_json(f"{API}/{self.company.identifier}/postings/{raw['id']}")
        return {**raw, **detail}

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        loc = raw.get("location") or {}
        sections = (raw.get("jobAd") or {}).get("sections") or {}

        def sec(key: str) -> str:
            return html_to_text((sections.get(key) or {}).get("text"))

        remote = "remote" if loc.get("remote") else ("hybrid" if loc.get("hybrid") else None)
        released = raw.get("releasedDate")
        try:
            posted = (
                datetime.fromisoformat(released.replace("Z", "+00:00")).replace(tzinfo=None) if released else None
            )
        except ValueError:
            posted = None
        company_slug = (raw.get("company") or {}).get("identifier") or self.company.identifier
        location = loc.get("fullLocation") or ", ".join(
            x for x in (loc.get("city"), loc.get("region"), loc.get("country")) if x
        )
        return JobPosting(
            source=self.name,
            source_type=self.source_type,
            external_job_id=str(raw.get("id")),
            url=raw.get("postingUrl") or f"https://jobs.smartrecruiters.com/{company_slug}/{raw.get('id')}",
            title=clean_line(raw.get("name")),
            company=self.company.company_name or clean_line((raw.get("company") or {}).get("name")),
            location=clean_line(location),
            country=(loc.get("country") or "").upper() or None,
            remote_status=remote,
            description="\n\n".join(
                x for x in (sec("companyDescription"), sec("jobDescription"), sec("additionalInformation")) if x
            ),
            requirements=sec("qualifications"),
            employment_type=(raw.get("typeOfEmployment") or {}).get("label"),
            seniority=(raw.get("experienceLevel") or {}).get("label"),
            department=(raw.get("department") or {}).get("label") or (raw.get("function") or {}).get("label"),
            career_page_url=self.company.careers_url,
            date_posted=posted,
            raw={k: v for k, v in raw.items() if k != "jobAd"},
        )
