"""Adapter registry, ATS detection from URLs, and single-job URL fetching (manual ingestion).

To add a new ATS (Workday, SuccessFactors, ...): subclass JobSourceAdapter, register it in ADAPTERS,
and teach `detect_ats` its URL shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

from app.core.http import HttpClient, assert_public_http_url
from app.core.normalize import classify_url_host
from app.core.text import clean_line
from app.enums import SourceType
from app.sources.base import CompanyFeedConfig, JobPosting, JobSourceAdapter
from app.sources.academic import AcademicListingAdapter
from app.sources.adzuna import AdzunaAdapter
from app.sources.amazon import AmazonJobsAdapter
from app.sources.arbeitnow import ArbeitnowAdapter
from app.sources.arbeitsagentur import ArbeitsagenturAdapter
from app.sources.ashby import AshbyAdapter
from app.sources.careerjet import CareerjetAdapter
from app.sources.generic import GenericCareerPageAdapter
from app.sources.getro import GetroAdapter
from app.sources.greenhouse import GreenhouseAdapter
from app.sources.jooble import JoobleAdapter
from app.sources.jsonld import extract_jsonld_postings, extract_page_fallback, jsonld_to_fields
from app.sources.lever import LeverAdapter
from app.sources.personio import PersonioAdapter
from app.sources.recruitee import RecruiteeAdapter
from app.sources.reed import ReedAdapter
from app.sources.rss import RssFeedAdapter
from app.sources.smartrecruiters import SmartRecruitersAdapter
from app.sources.workable import WorkableAdapter
from app.sources.workable_board import WorkableBoardAdapter
from app.sources.workday import WorkdayAdapter

ADAPTERS: dict[str, type[JobSourceAdapter]] = {
    "greenhouse": GreenhouseAdapter,
    "lever": LeverAdapter,
    "personio": PersonioAdapter,
    "smartrecruiters": SmartRecruitersAdapter,
    "ashby": AshbyAdapter,
    "workable": WorkableAdapter,
    "recruitee": RecruiteeAdapter,
    "workday": WorkdayAdapter,
    "amazon": AmazonJobsAdapter,
    "getro": GetroAdapter,
    "arbeitsagentur": ArbeitsagenturAdapter,
    "workable_board": WorkableBoardAdapter,
    "careerjet": CareerjetAdapter,
    "jooble": JoobleAdapter,
    "arbeitnow": ArbeitnowAdapter,
    "rss": RssFeedAdapter,
    "reed": ReedAdapter,
    "adzuna": AdzunaAdapter,
    "generic": GenericCareerPageAdapter,
    "academic": AcademicListingAdapter,
}

ATS_TYPES = [*ADAPTERS.keys(), "none"]


def build_adapter(ats_type: str, company: CompanyFeedConfig, http: HttpClient) -> JobSourceAdapter:
    cls = ADAPTERS.get(ats_type)
    if cls is None:
        raise ValueError(f"No adapter for ATS type '{ats_type}'")
    return cls(company, http)


@dataclass
class AtsMatch:
    ats_type: str
    identifier: str | None
    job_id: str | None = None
    options: dict | None = None


def detect_ats(url: str) -> AtsMatch:
    """Recognise ATS-hosted career/job URLs. Falls back to the generic adapter."""
    parts = urlsplit(url.strip())
    host = parts.netloc.lower()
    path = [p for p in parts.path.split("/") if p]
    qs = parse_qs(parts.query)

    if "greenhouse.io" in host:
        options = {"region": "eu"} if ".eu." in host else {}
        token = (qs.get("for") or [None])[0]
        job_id = (qs.get("gh_jid") or qs.get("token") or [None])[0]
        if not token and path and path[0] not in ("embed", "v1"):
            token = path[0]
        if "jobs" in path:
            idx = path.index("jobs")
            if idx + 1 < len(path):
                job_id = path[idx + 1]
        return AtsMatch("greenhouse", token, job_id, options)
    if "lever.co" in host:
        options = {"region": "eu"} if ".eu." in host else {}
        return AtsMatch("lever", path[0] if path else None, path[1] if len(path) > 1 else None, options)
    m = re.match(r"^([a-z0-9-]+)\.jobs\.personio\.(de|com)$", host)
    if m:
        job_id = path[1] if len(path) > 1 and path[0] == "job" else None
        return AtsMatch("personio", m.group(1), job_id, {"domain": f"jobs.personio.{m.group(2)}"})
    if "smartrecruiters.com" in host and path:
        if host.startswith("api."):
            ident = path[2] if len(path) > 2 else None
            return AtsMatch("smartrecruiters", ident)
        job_id = path[1].split("-")[0] if len(path) > 1 and path[1][:1].isdigit() else None
        return AtsMatch("smartrecruiters", path[0], job_id)
    if host in ("jobs.ashbyhq.com", "api.ashbyhq.com") and path:
        board = path[2] if host.startswith("api.") and len(path) > 2 else path[0]
        return AtsMatch("ashby", board, path[1] if host.startswith("jobs.") and len(path) > 1 else None)
    if host == "apply.workable.com" and path:
        return AtsMatch("workable", path[0], path[2] if len(path) > 2 and path[1] == "j" else None)
    m = re.match(r"^([a-z0-9-]+)\.recruitee\.com$", host)
    if m:
        return AtsMatch("recruitee", m.group(1))
    m = re.match(r"^([a-z0-9-]+)\.(wd\d+)\.myworkdayjobs\.com$", host)
    if m and path:
        site = path[1] if len(path) > 1 and re.fullmatch(r"[a-z]{2}-[A-Z]{2}", path[0]) else path[0]
        return AtsMatch("workday", f"{m.group(1)}.{m.group(2)}/{site}")
    # Greenhouse-powered career sites often carry ?gh_jid=...
    if qs.get("gh_jid"):
        return AtsMatch("generic", None, qs["gh_jid"][0])
    return AtsMatch("generic", None)


class UrlFetchError(RuntimeError):
    pass


def fetch_posting_from_url(url: str, http: HttpClient, company_hint: str = "") -> JobPosting:
    """Fetch one job from a URL: ATS API when recognised, else JSON-LD, else page text."""
    assert_public_http_url(url)
    match = detect_ats(url)
    cfg = CompanyFeedConfig(company_name=company_hint, identifier=match.identifier, options=match.options or {})

    if match.ats_type == "greenhouse" and match.identifier and match.job_id:
        adapter = GreenhouseAdapter(cfg, http)
        return adapter.normalize_job(adapter.fetch_single(match.job_id))
    if match.ats_type == "lever" and match.identifier and match.job_id:
        adapter = LeverAdapter(cfg, http)
        return adapter.normalize_job(adapter.fetch_single(match.job_id))
    if match.ats_type == "smartrecruiters" and match.identifier and match.job_id:
        adapter = SmartRecruitersAdapter(cfg, http)
        return adapter.normalize_job(adapter.fetch_details({"id": match.job_id}))

    board = classify_url_host(url)
    source, source_type = (board[0], SourceType(board[1])) if board else ("career_page", SourceType.EMPLOYER)
    if match.ats_type in ("personio", "greenhouse", "lever", "smartrecruiters"):
        source, source_type = match.ats_type, SourceType.EMPLOYER

    html = http.get_text(url)
    nodes = extract_jsonld_postings(html)
    if nodes:
        fields = jsonld_to_fields(nodes[0])
        fields.pop("url", None)
        ext = fields.pop("external_job_id", None)
        title = fields.pop("title", "") or "Untitled position"
        company = fields.pop("company", "") or company_hint
        return JobPosting(
            source=source, source_type=source_type, external_job_id=ext, url=url, title=title,
            company=company, raw={"jsonld": True}, **{k: v for k, v in fields.items() if v not in (None, "")},
        )
    page = extract_page_fallback(html)
    if len(page["text"]) < 200:
        raise UrlFetchError(
            "The page returned almost no text (probably requires login or JavaScript). "
            "Paste the job description into the form instead."
        )
    return JobPosting(
        source=source, source_type=source_type, url=url, title=page["title"] or "Untitled position",
        company=company_hint or clean_line(page.get("site_name")), description=page["text"],
        raw={"page_title": page["title"]},
    )
