"""Lever public Postings API: https://github.com/lever/postings-api"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.core.text import clean_line, clean_text, html_to_text
from app.sources.base import JobPosting, JobSourceAdapter


def _api_base(options: dict[str, Any]) -> str:
    return "https://api.eu.lever.co/v0/postings" if options.get("region") == "eu" else "https://api.lever.co/v0/postings"


class LeverAdapter(JobSourceAdapter):
    name = "lever"

    def fetch_jobs(self) -> list[dict[str, Any]]:
        slug = self.company.identifier
        if not slug:
            raise ValueError("Lever adapter needs ats_identifier (company slug)")
        out: list[dict[str, Any]] = []
        skip, limit = 0, 100
        while True:
            page = self.http.get_json(
                f"{_api_base(self.company.options)}/{slug}", params={"mode": "json", "skip": skip, "limit": limit}
            )
            if not isinstance(page, list):
                raise ValueError(f"Unexpected Lever response for {slug}")
            out.extend(page)
            if len(page) < limit or len(out) >= 2000:
                return out
            skip += limit

    def fetch_single(self, posting_id: str) -> dict[str, Any]:
        return self.http.get_json(f"{_api_base(self.company.options)}/{self.company.identifier}/{posting_id}", params={"mode": "json"})

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        cats = raw.get("categories") or {}
        lists = raw.get("lists") or []
        requirements, responsibilities, other = [], [], []
        for block in lists:
            heading = clean_line(block.get("text"))
            body = html_to_text(block.get("content"))
            h = heading.lower()
            if any(k in h for k in ("require", "qualif", "you have", "about you", "profile", "skills", "what you bring", "must")):
                requirements.append(f"{heading}\n{body}")
            elif any(k in h for k in ("responsib", "you will", "what you'll do", "what you will do", "role", "tasks", "aufgaben")):
                responsibilities.append(f"{heading}\n{body}")
            else:
                other.append(f"{heading}\n{body}")
        description = "\n\n".join(
            x for x in [
                clean_text(raw.get("descriptionPlain")) or html_to_text(raw.get("description")),
                *other,
                clean_text(raw.get("additionalPlain")) or html_to_text(raw.get("additional")),
            ] if x
        )
        salary = raw.get("salaryRange") or {}
        created = raw.get("createdAt")
        all_locations = cats.get("allLocations") or []
        location = cats.get("location") or ", ".join(all_locations)
        workplace = (raw.get("workplaceType") or "").lower()
        remote = {"remote": "remote", "hybrid": "hybrid", "onsite": "onsite", "on-site": "onsite"}.get(workplace)
        return JobPosting(
            source=self.name,
            source_type=self.source_type,
            external_job_id=str(raw.get("id")),
            url=raw.get("hostedUrl"),
            title=clean_line(raw.get("text")),
            company=self.company.company_name,
            location=clean_line(location) or self._fallback_location(),
            country=raw.get("country"),
            remote_status=remote,
            description=description,
            requirements="\n\n".join(requirements),
            responsibilities="\n\n".join(responsibilities),
            salary_min=salary.get("min"),
            salary_max=salary.get("max"),
            currency=salary.get("currency"),
            employment_type=cats.get("commitment"),
            department=cats.get("department") or cats.get("team"),
            career_page_url=self.company.careers_url,
            date_posted=datetime.fromtimestamp(created / 1000, UTC).replace(tzinfo=None) if isinstance(created, (int, float)) else None,
            raw={k: v for k, v in raw.items() if k not in ("description", "descriptionPlain", "lists", "additional", "additionalPlain")},
        )
