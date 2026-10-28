"""Workday career sites (<tenant>.wd<N>.myworkdayjobs.com/<site>) through their public CXS JSON API.

ats_identifier: "<tenant>.wd<N>/<site>", e.g. "nvidia.wd5/NVIDIAExternalCareerSite"
ats_config:
  search:   search texts (default: your target countries — Workday matches them against the job location)
  max_jobs: cap per search text (default 300; 20 per request)
Large employers list thousands of jobs worldwide: searching by country keeps the feed to the places you want.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.core.text import clean_line, html_to_text
from app.sources.base import JobPosting, JobSourceAdapter

PAGE = 20


def parse_identifier(identifier: str) -> tuple[str, str, str]:
    """'nvidia.wd5/NVIDIAExternalCareerSite' → (host, tenant, site)."""
    host_part, _, site = (identifier or "").partition("/")
    tenant, _, wd = host_part.partition(".")
    if not (tenant and wd and site):
        raise ValueError("Workday ats_identifier must look like 'tenant.wd5/SiteName'")
    return f"{tenant}.{wd}.myworkdayjobs.com", tenant, site


class WorkdayAdapter(JobSourceAdapter):
    name = "workday"

    def _api(self) -> str:
        host, tenant, site = parse_identifier(self.company.identifier or "")
        return f"https://{host}/wday/cxs/{tenant}/{site}"

    def _searches(self) -> list[str]:
        searches = self.company.options.get("search")
        if searches:
            return [searches] if isinstance(searches, str) else list(searches)
        from app.config_files import get_prefs

        return list(get_prefs().locations.target_countries)

    def fetch_jobs(self) -> list[dict[str, Any]]:
        api, seen, out = self._api(), set(), []
        cap = int(self.company.options.get("max_jobs", 300))
        for text in self._searches():
            for offset in range(0, cap, PAGE):
                data = self.http.post_json(f"{api}/jobs", {"appliedFacets": {}, "limit": PAGE, "offset": offset,
                                                           "searchText": "" if text == "*" else text})
                postings = data.get("jobPostings") or []
                for p in postings:
                    path = p.get("externalPath")
                    if path and path not in seen:
                        seen.add(path)
                        out.append(p)
                if len(postings) < PAGE or offset + PAGE >= int(data.get("total") or 0):
                    break
            else:
                self.truncated = True  # stopped at max_jobs: more postings exist than were read
        return out

    def external_id(self, raw: dict[str, Any]) -> str | None:
        return raw.get("externalPath") or raw.get("jobPostingId")

    def location_hint(self, raw: dict[str, Any]) -> str:
        """'/job/Germany-Munich/Title_JR123' → 'Germany Munich' (the listing has no other location data)."""
        info = raw.get("jobPostingInfo") or {}
        if info:
            return ", ".join(x for x in (info.get("location"), (info.get("country") or {}).get("descriptor")) if x)
        parts = (raw.get("externalPath") or "").split("/")
        return parts[2].replace("-", " ") if len(parts) > 2 else ""

    def fetch_details(self, raw: dict[str, Any]) -> dict[str, Any]:
        data = self.http.get_json(f"{self._api()}{raw['externalPath']}")
        return {**raw, "jobPostingInfo": data.get("jobPostingInfo") or {}}

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        info = raw.get("jobPostingInfo") or {}
        extra = info.get("additionalLocations") or []
        location = "; ".join(x for x in [self.location_hint(raw), *extra] if x)
        posted = None
        if info.get("startDate"):
            try:
                posted = datetime.fromisoformat(info["startDate"][:10])
            except ValueError:
                posted = None
        host, _, site = parse_identifier(self.company.identifier or "")
        return JobPosting(
            source=self.name, source_type=self.source_type, external_job_id=self.external_id(raw),
            url=info.get("externalUrl") or f"https://{host}/{site}{raw.get('externalPath', '')}",
            title=clean_line(info.get("title") or raw.get("title")), company=self.company.company_name,
            location=clean_line(location) or self._fallback_location(), description=html_to_text(info.get("jobDescription")),
            employment_type=info.get("timeType"), career_page_url=self.company.careers_url, date_posted=posted,
            raw={"externalPath": raw.get("externalPath"), "jobReqId": info.get("jobReqId")},
        )
