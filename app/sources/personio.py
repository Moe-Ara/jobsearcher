"""Personio public XML feed: https://{company}.jobs.personio.de/xml"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup

from app.core.text import clean_line, html_to_text
from app.sources.base import JobPosting, JobSourceAdapter

_REQ_WORDS = ("profil", "profile", "qualif", "require", "you bring", "about you", "skills", "anforderung", "mitbringst")
_RESP_WORDS = ("aufgaben", "tasks", "responsib", "your role", "you will", "what you", "deine rolle")


class PersonioAdapter(JobSourceAdapter):
    name = "personio"

    def _host(self) -> str:
        domain = self.company.options.get("domain", "jobs.personio.de")  # or jobs.personio.com
        return f"https://{self.company.identifier}.{domain}"

    def fetch_jobs(self) -> list[dict[str, Any]]:
        if not self.company.identifier:
            raise ValueError("Personio adapter needs ats_identifier (subdomain)")
        params = {"language": self.company.options.get("language", "en")}
        xml = self.http.get(f"{self._host()}/xml", params=params, accept="application/xml").text
        return parse_personio_xml(xml)

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        description, requirements, responsibilities = [], [], []
        for heading, body in raw.get("sections") or []:
            h = heading.lower()
            text = f"{heading}\n{body}" if heading else body
            if any(k in h for k in _REQ_WORDS):
                requirements.append(text)
            elif any(k in h for k in _RESP_WORDS):
                responsibilities.append(text)
            else:
                description.append(text)
        loc = ", ".join(x for x in (raw.get("office"), *raw.get("additional_offices", [])) if x)
        return JobPosting(
            source=self.name,
            source_type=self.source_type,
            external_job_id=str(raw["id"]),
            url=f"{self._host()}/job/{raw['id']}",
            title=clean_line(raw.get("name")),
            company=self.company.company_name or clean_line(raw.get("subcompany")),
            location=clean_line(loc) or self._fallback_location(),
            description="\n\n".join(description),
            requirements="\n\n".join(requirements),
            responsibilities="\n\n".join(responsibilities),
            employment_type=", ".join(x for x in (raw.get("employmentType"), raw.get("schedule")) if x) or None,
            seniority=raw.get("seniority") or None,
            department=raw.get("department") or None,
            career_page_url=self.company.careers_url or self._host(),
            date_posted=raw.get("createdAt"),
            raw={k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in raw.items() if k != "sections"},
        )


def parse_personio_xml(xml: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(xml, "xml")
    out: list[dict[str, Any]] = []
    for pos in soup.find_all("position"):

        def tag(name: str, pos=pos) -> str:
            el = pos.find(name, recursive=False)
            return el.get_text(strip=True) if el else ""

        sections = []
        for jd in pos.find_all("jobDescription"):
            name_el, value_el = jd.find("name"), jd.find("value")
            sections.append(
                (
                    clean_line(name_el.get_text() if name_el else ""),
                    html_to_text(value_el.get_text() if value_el else ""),
                )
            )
        created = tag("createdAt")
        try:
            created_dt = (
                datetime.fromisoformat(created.replace("Z", "+00:00")).replace(tzinfo=None) if created else None
            )
        except ValueError:
            created_dt = None
        extra = pos.find("additionalOffices")
        out.append(
            {
                "id": tag("id"),
                "subcompany": tag("subcompany"),
                "office": tag("office"),
                "additional_offices": [o.get_text(strip=True) for o in extra.find_all("office")] if extra else [],
                "department": tag("department"),
                "recruitingCategory": tag("recruitingCategory"),
                "name": tag("name"),
                "employmentType": tag("employmentType"),
                "seniority": tag("seniority"),
                "schedule": tag("schedule"),
                "yearsOfExperience": tag("yearsOfExperience"),
                "keywords": tag("keywords"),
                "occupation": tag("occupation"),
                "occupationCategory": tag("occupationCategory"),
                "createdAt": created_dt,
                "sections": sections,
            }
        )
    return [p for p in out if p["id"]]
