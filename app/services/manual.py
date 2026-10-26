"""Manual ingestion from the dashboard: a job URL and/or pasted description, or a company careers URL."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.core.http import FetchError, get_http
from app.core.normalize import classify_url_host, infer_country
from app.core.text import clean_line, clean_text
from app.db import session_scope
from app.enums import SourceType
from app.models import SourceRun, utcnow
from app.services.analysis import extract_fields
from app.services.companies import upsert_company
from app.services.ingest import ingest_posting
from app.sources.base import JobPosting
from app.sources.registry import UrlFetchError, detect_ats, fetch_posting_from_url

log = logging.getLogger(__name__)


class ManualIngestError(ValueError):
    pass


@dataclass
class ManualResult:
    job_id: int
    outcome: str
    message: str


def ingest_manual(
    *, url: str = "", title: str = "", company: str = "", location: str = "", description: str = ""
) -> ManualResult:
    url, title, company, location = url.strip(), clean_line(title), clean_line(company), clean_line(location)
    description = clean_text(description)
    if not url and not description:
        raise ManualIngestError("Provide a job URL, a description, or both.")

    posting: JobPosting | None = None
    fetch_note = ""
    if url and not description:
        try:
            posting = fetch_posting_from_url(url, get_http(), company_hint=company)
        except (FetchError, UrlFetchError) as exc:
            raise ManualIngestError(f"Could not read that URL ({exc}). Paste the job description as well.") from exc
    elif url:
        fetch_note = " (description taken from your paste)"

    if posting is None:
        board = classify_url_host(url) if url else None
        if board:
            source, source_type = board[0], SourceType(board[1])
        elif url and detect_ats(url).ats_type != "generic":
            source, source_type = detect_ats(url).ats_type, SourceType.EMPLOYER
        else:
            source, source_type = "manual", SourceType.MANUAL
        posting = JobPosting(
            source=source, source_type=source_type, url=url or None, title=title or "",
            company=company, location=location, description=description,
        )

    # user-provided fields win over scraped ones
    if title:
        posting.title = title
    if company:
        posting.company = company
    if location:
        posting.location = location

    with session_scope() as s:
        if not posting.title or not posting.company:
            extracted = extract_fields(s, f"{posting.title}\n{posting.company}\n{posting.location}\n{posting.description}")
            posting.title = posting.title or extracted.get("title", "")
            posting.company = posting.company or extracted.get("company", "")
            posting.location = posting.location or extracted.get("location", "")
            posting.country = posting.country or infer_country(None, extracted.get("country"))
        if not posting.title:
            first_line = next((ln for ln in posting.description.splitlines() if ln.strip()), "")
            posting.title = first_line[:150] or "Untitled position"

        result = ingest_posting(s, posting, enforce_retention=False)  # explicit user action
        s.add(SourceRun(
            source_name=f"manual:{posting.source}", adapter="manual", started_at=utcnow(), finished_at=utcnow(),
            success=True, jobs_found=1, jobs_new=int(result.outcome == "new"), jobs_merged=int(result.outcome == "merged"),
            jobs_updated=int(result.outcome == "updated"), errors=[],
        ))
        s.flush()
        msg = {
            "new": "Job added",
            "merged": f"Matched an existing job ({result.dedup_reason}) — source added",
            "updated": "Existing job updated",
            "unchanged": "Job already known",
        }[result.outcome]
        return ManualResult(result.job.id, result.outcome, msg + fetch_note)


def add_company_from_url(careers_url: str, name: str = "", priority: str = "B", country: str = "", city: str = "", sector: str = "") -> tuple[int, str]:
    careers_url = careers_url.strip()
    if not careers_url.lower().startswith(("http://", "https://")):
        raise ManualIngestError("Careers URL must start with http(s)://")
    match = detect_ats(careers_url)
    name = clean_line(name) or (match.identifier or "").replace("-", " ").title()
    if not name:
        raise ManualIngestError("Couldn't infer the company name — please enter it.")
    with session_scope() as s:
        tc, created = upsert_company(
            s,
            {"company_name": name, "careers_url": careers_url, "priority": priority, "country": country,
             "city": city, "sector": sector},
            overwrite=True,
        )
        s.flush()
        how = f"{tc.ats_type} feed" + (f" '{tc.ats_identifier}'" if tc.ats_identifier else "")
        return tc.id, f"{'Added' if created else 'Updated'} {tc.company_name} ({how})"
