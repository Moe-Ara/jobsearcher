"""RSS / Atom job feeds (job boards that publish feeds, e.g. Berlin Startup Jobs).

ats_config:
  feeds:            feed URLs (careers_url is read too)
  title_separator:  "Title // Company" style titles are split on this (e.g. " // ")
  fetch_page:       read each new posting's page for the full text (JSON-LD or page text) — feeds often carry teasers
"""

from __future__ import annotations

from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

from bs4 import BeautifulSoup

from app.core.text import clean_line, html_to_text
from app.enums import SourceType
from app.sources.base import JobPosting, JobSourceAdapter
from app.sources.jsonld import extract_jsonld_postings, extract_page_fallback, jsonld_to_fields


class RssFeedAdapter(JobSourceAdapter):
    name = "rss"
    source_type = SourceType.JOB_BOARD
    partial_listing = True  # a feed shows the newest items only

    def fetch_jobs(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        feeds = [u for u in [self.company.careers_url, *(self.company.options.get("feeds") or [])] if u]
        for url in dict.fromkeys(feeds):
            soup = BeautifulSoup(self.http.get_text(url), "xml")
            for item in soup.find_all(["item", "entry"]):
                link = item.find("link")
                href = (link.get("href") or link.get_text(strip=True)) if link else ""
                out.append({
                    "title": item.find("title").get_text(" ", strip=True) if item.find("title") else "",
                    "url": href.strip(),
                    "summary": (item.find("description") or item.find("summary") or item.find("content")
                                or BeautifulSoup("", "xml")).get_text(" ", strip=True),
                    "published": (item.find("pubDate") or item.find("published") or item.find("updated")
                                  or BeautifulSoup("", "xml")).get_text(strip=True),
                    "categories": [c.get_text(strip=True) for c in item.find_all("category")],
                })
        return [r for r in out if r["url"] and r["title"]]

    def external_id(self, raw: dict[str, Any]) -> str | None:
        return raw.get("url")

    def fetch_details(self, raw: dict[str, Any]) -> dict[str, Any]:
        if not self.company.options.get("fetch_page"):
            return raw
        try:
            html = self.http.get_text(raw["url"])
        except Exception:  # noqa: BLE001 - the feed item alone is still a posting
            return raw
        nodes = extract_jsonld_postings(html)
        if nodes:
            return {**raw, "fields": jsonld_to_fields(nodes[0])}
        return {**raw, "page_text": extract_page_fallback(html)["text"][:20000]}

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        from html import unescape

        f = raw.get("fields") or {}
        title, company = unescape(f.get("title") or raw["title"]), ""
        sep = self.company.options.get("title_separator")
        if sep and sep in title:  # "Senior Backend Developer – Go & Kubernetes // DATATRONiQ"
            title, company = (x.strip() for x in title.rsplit(sep, 1))
        posted = None
        try:
            posted = parsedate_to_datetime(raw.get("published") or "").replace(tzinfo=None)
        except (TypeError, ValueError):
            try:
                posted = datetime.fromisoformat((raw.get("published") or "")[:19])
            except ValueError:
                posted = None
        return JobPosting(
            source=self.company.options.get("source_name") or self.name, source_type=self.source_type,
            external_job_id=raw["url"], url=raw["url"], title=clean_line(title),
            company=clean_line(f.get("company") or company), location=clean_line(f.get("location") or ""),
            country=f.get("country"), remote_status=f.get("remote_status"),
            # fetch_page sources: a known item (page not re-read) carries no text, so the stored full posting is kept
            description=f.get("description") or raw.get("page_text") or (
                "" if self.company.options.get("fetch_page") else html_to_text(raw.get("summary"))),
            requirements=f.get("requirements") or "", date_posted=posted,
            raw={"categories": raw.get("categories") or []},
        )
