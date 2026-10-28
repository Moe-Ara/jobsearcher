"""Generic career page adapter.

Strategy (no anti-bot fighting, no JS rendering):
1. JobPosting JSON-LD embedded on the careers page itself.
2. If `ats_config.job_link_pattern` (regex) is set: collect matching links on the careers page and read
   JSON-LD (or the page text) from each detail page — only for links not seen before.
   `ats_config.listing_urls`: more listing pages (e.g. one per job category) read the same way.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from app.core.text import clean_line
from app.sources.base import JobPosting, JobSourceAdapter
from app.sources.jsonld import extract_jsonld_postings, extract_page_fallback, jsonld_to_fields


class GenericCareerPageAdapter(JobSourceAdapter):
    name = "career_page"

    def fetch_jobs(self) -> list[dict[str, Any]]:
        url = self.company.careers_url
        if not url:
            raise ValueError("Generic adapter needs careers_url")
        out: list[dict[str, Any]] = []
        pattern = self.company.options.get("job_link_pattern")
        regex = re.compile(pattern) if pattern else None
        seen: set[str] = set()
        max_links = int(self.company.options.get("max_links", 200))
        for page_url in [url, *(self.company.options.get("listing_urls") or [])]:
            html = self.http.get_text(page_url)
            out += [{"kind": "jsonld", "fields": jsonld_to_fields(node), "page_url": page_url}
                    for node in extract_jsonld_postings(html)]
            if regex is None:
                continue
            for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
                href = urljoin(page_url, a["href"]).split("#")[0]
                if len(seen) >= max_links and regex.search(href) and href not in seen:
                    self.truncated = True  # more posting links than max_links
                if href in seen or not regex.search(href) or len(seen) >= max_links:
                    continue
                seen.add(href)
                out.append({"kind": "link", "url": href, "anchor": clean_line(a.get_text(" "))})
        return out

    def external_id(self, raw: dict[str, Any]) -> str | None:
        if raw["kind"] in ("link", "page") or raw.get("link_id"):
            return raw.get("link_id") or raw["url"]
        f = raw["fields"]
        return f.get("external_job_id") or f.get("url") or f"{f.get('title')}|{f.get('location')}"

    def fetch_details(self, raw: dict[str, Any]) -> dict[str, Any]:
        if raw["kind"] != "link":
            return raw
        html = self.http.get_text(raw["url"])
        nodes = extract_jsonld_postings(html)
        if nodes:
            return {"kind": "jsonld", "fields": jsonld_to_fields(nodes[0]), "page_url": raw["url"], "link_id": raw["url"]}
        page = extract_page_fallback(html)
        return {"kind": "page", "page": page, "url": raw["url"], "anchor": raw.get("anchor", ""), "link_id": raw["url"]}

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        kind = raw["kind"]
        common = {
            "source": self.name,
            "source_type": self.source_type,
            "career_page_url": self.company.careers_url,
        }
        if kind == "link":  # already known: detail fetch skipped, minimal record keeps last_seen fresh
            return JobPosting(
                **common, external_job_id=raw["url"], url=raw["url"],
                title=raw.get("anchor") or raw["url"], company=self.company.company_name,
            )
        if kind == "page":
            page = raw["page"]
            return JobPosting(
                **common, external_job_id=raw["link_id"], url=raw["url"],
                title=raw.get("anchor") or page["title"] or raw["url"], company=self.company.company_name,
                location=self._fallback_location(), description=page["text"], raw={"page_title": page["title"]},
            )
        f = dict(raw["fields"])
        ext = self.external_id(raw)
        url = f.pop("url", None) or raw.get("page_url")
        f.pop("external_job_id", None)
        scraped_company = f.pop("company", "")
        title = f.pop("title", "") or raw.get("anchor") or "Untitled position"
        if not f.get("location"):
            f["location"] = self._fallback_location()
        return JobPosting(
            **common, external_job_id=ext, url=url, title=title,
            company=self.company.company_name or scraped_company,
            raw={"jsonld": True}, **{k: v for k, v in f.items() if v not in (None, "")},
        )
