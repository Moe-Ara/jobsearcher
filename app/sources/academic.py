"""Academic listing adapter — doctoral / research positions from university, institute and academic job boards.

Funded PhD positions are rarely on the usual job boards; they are announced on university job portals, research
group pages and academic boards (academics.de, EURAXESS, ELLIS, ...). This adapter reads one or more listing pages,
follows links whose URL matches `ats_config.job_link_pattern` (and, optionally, whose link text matches
`anchor_pattern`), and reads each posting from JSON-LD, the page text or a linked PDF.

ats_config:
  listing_urls:       extra listing pages (search queries) besides careers_url
  job_link_pattern:   regex a posting URL must match (required)
  anchor_pattern:     regex the link text must match (default: doctoral / research-position words)
  employer_from_posting: true for aggregators — the posting's own employer becomes the job's company
  board:              true when the source is a job board (not the employer's own site)
  max_links:          cap per listing page (default 60)

Whether a position is funded and in your field is decided later by the hard filters (`phd` in preferences.yaml).
"""

from __future__ import annotations

import io
import logging
import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from app.core.normalize import infer_country, infer_country_from_title
from app.core.text import clean_line, clean_text
from app.enums import SourceType
from app.sources.base import JobPosting
from app.sources.generic import GenericCareerPageAdapter
from app.sources.jsonld import extract_jsonld_postings, extract_page_fallback, jsonld_to_fields

log = logging.getLogger(__name__)

DEFAULT_ANCHOR_PATTERN = (
    r"ph\.?\s?d|doktorand|doctoral|doctorate|promotion|promovier|"
    r"(wissenschaftlich|akademisch)\w*(?:[*/:]\w*)?\s+mitarbeiter|research (associate|assistant|scientist)"
)

# "Label value" pairs on EURAXESS-style pages (plain text after HTML stripping).
_LABELS: dict[str, list[str]] = {
    "company": [r"Organisation/Company\s+(.{3,120}?)\s+(?:Department|Research Field|Researcher Profile)"],
    "country": [r"\bCountry\s+([A-Z][A-Za-z ]{2,40}?)\s+(?:Type of Contract|City|State|Postal Code|Job Status|Application)"],
    "city": [r"\bCity\s+([A-ZÄÖÜ][\w .\-]{1,40}?)\s+(?:Postal Code|Street|Website|State|Type)"],
}
_ABROAD = re.compile(r"andere länder|other countries|\bausland\b", re.IGNORECASE)
_TITLE_EMPLOYER =re.compile(r"\b(?:at|@)\s+(?:the\s+)?([A-Z][^,;()|]{3,80}?)\s*(?:\([^)]*\))?\s*$")


def pdf_text(data: bytes, max_pages: int = 6) -> str:
    """Plain text of a (short) PDF announcement; empty when it cannot be read."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages[:max_pages])
    except Exception as exc:  # noqa: BLE001 - a broken PDF must not stop the feed
        log.info("Could not read PDF posting: %s", exc)
        return ""


def _label(text: str, key: str) -> str:
    for pattern in _LABELS[key]:
        m = re.search(pattern, text)
        if m:
            return clean_line(m.group(1))
    return ""


def guess_country(title: str, text: str, location: str = "") -> str | None:
    """Country from labels, location, title (institution names often carry the city) or the text near the title."""
    labelled = _label(text, "country")
    for candidate in (labelled, location):
        if candidate and (c := infer_country(candidate)):
            return c
    if c := (infer_country(title) or infer_country_from_title(title)):
        return c
    idx = text.find(title[:40]) if title else -1
    window = text[idx + len(title): idx + len(title) + 300] if idx >= 0 else text[:300]
    return infer_country(window)


class AcademicListingAdapter(GenericCareerPageAdapter):
    name = "academic"
    partial_listing = True  # listing / search pages show a window of postings, not every one

    @property
    def source_type(self) -> SourceType:  # type: ignore[override]
        return SourceType.JOB_BOARD if self.company.options.get("board") else SourceType.EMPLOYER

    def _listing_urls(self) -> list[str]:
        urls = [self.company.careers_url, *(self.company.options.get("listing_urls") or [])]
        return [u for u in dict.fromkeys(urls) if u]

    def fetch_jobs(self) -> list[dict[str, Any]]:
        pattern = self.company.options.get("job_link_pattern")
        if not pattern:
            raise ValueError("Academic adapter needs ats_config.job_link_pattern")
        link_rx = re.compile(pattern)
        anchor_rx = re.compile(self.company.options.get("anchor_pattern") or DEFAULT_ANCHOR_PATTERN, re.IGNORECASE)
        max_links = int(self.company.options.get("max_links", 60))  # listing pages are searches: see partial_listing
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        errors: list[str] = []
        urls = self._listing_urls()
        if not urls:
            raise ValueError("Academic adapter needs careers_url or ats_config.listing_urls")
        for listing in urls:
            try:
                html = self.http.get_text(listing)
            except Exception as exc:  # noqa: BLE001 - one dead search page must not hide the others
                errors.append(f"{listing}: {exc}")
                continue
            taken = 0
            for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
                href = urljoin(listing, a["href"]).split("#")[0]
                anchor = clean_line(a.get_text(" "))
                if href in seen or not link_rx.search(href) or not anchor_rx.search(anchor or a.get("title", "")):
                    continue
                seen.add(href)
                out.append({"kind": "link", "url": href, "anchor": anchor})
                taken += 1
                if taken >= max_links:
                    break
        if errors and len(errors) == len(urls):
            raise RuntimeError("; ".join(errors[:3]))
        return out

    def fetch_details(self, raw: dict[str, Any]) -> dict[str, Any]:
        if raw["kind"] != "link":
            return raw
        url = raw["url"]
        resp = self.http.get(url)
        ctype = resp.headers.get("content-type", "").lower()
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            text = clean_text(pdf_text(resp.content))
            page = {"title": raw.get("anchor", ""), "site_name": "", "text": text}
            return {"kind": "page", "page": page, "url": url, "anchor": raw.get("anchor", ""), "link_id": url}
        html = resp.text
        nodes = extract_jsonld_postings(html)
        page = extract_page_fallback(html)
        if nodes:
            fields = jsonld_to_fields(nodes[0])
            if not fields.get("description") and page["text"]:
                fields["description"] = page["text"]
            return {"kind": "jsonld", "fields": fields, "page_url": url, "link_id": url, "anchor": raw.get("anchor", "")}
        return {"kind": "page", "page": page, "url": url, "anchor": raw.get("anchor", ""), "link_id": url}

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        posting = super().normalize_job(raw)
        from_posting = bool(self.company.options.get("employer_from_posting"))
        if raw["kind"] == "jsonld":
            scraped = clean_line(raw["fields"].get("company"))
            if from_posting and scraped:
                posting.company = scraped
            posting.url = raw.get("link_id") or posting.url
        elif raw["kind"] == "page":
            text = raw["page"]["text"]
            if from_posting:
                posting.company = _label(text, "company") or posting.company
            if not self.company.country:  # an employer's own portal keeps its configured city/country
                city = _label(text, "city")
                country = guess_country(posting.title, text, city)
                if country:
                    posting.country = country
                    posting.location = ", ".join(x for x in (city, country) if x)
        if not posting.country and not self.company.country:
            posting.country = guess_country(posting.title, posting.description, posting.location) or None
        if not posting.country and _ABROAD.search(posting.location or ""):
            posting.country = "Outside Germany"  # academics.de: "Aarhus, Andere Länder"
        if from_posting and posting.company == self.company.company_name:
            # an aggregator is never the employer; "PhD position in X at TU Darmstadt" names it in the title
            m = _TITLE_EMPLOYER.search(posting.title)
            posting.company = clean_line(m.group(1)) if m else ""
        posting.raw = {**posting.raw, "listing": self.company.company_name}
        return posting
