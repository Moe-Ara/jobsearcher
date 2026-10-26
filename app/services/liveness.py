"""Is the posting still open? Checked right before the agents prepare an application (single run and queue).

  1. the job's feed no longer lists it (job.active is False)                      → closed
  2. an application deadline in the posting text has passed                       → closed
  3. the source itself: the ATS API (Greenhouse, Lever, SmartRecruiters, Workday, Personio, Ashby, Workable,
     Recruitee, Bundesagentur) no longer has the job, or the posting page answers 404/410 or says it is closed
     ("no longer accepting applications", "Stelle ist bereits besetzt", "Bewerbungsfrist ist abgelaufen", …) → closed
Anything that cannot be checked (LinkedIn / Indeed / StepStone often block automated requests, timeouts) is
"unknown" — the run continues; only a clear signal stops it.
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass
from datetime import date

from app.models import utcnow

from app.core.text import clean_text

log = logging.getLogger(__name__)

CLOSED_PHRASES = re.compile(
    r"no longer accepting applications|no longer (available|active|open)|this (job|position|posting|vacancy) (is|has) "
    r"(been )?(closed|filled|expired|removed)|position has been filled|(job|posting|vacancy) (has )?expired|"
    r"applications? (are|is) (now )?closed|we are no longer recruiting|"
    r"stelle (ist )?(bereits |leider )?(besetzt|vergeben)|(nicht|nicht mehr) (mehr )?verfügbar|"
    r"(stellen)?anzeige (ist )?(nicht mehr aktiv|abgelaufen|deaktiviert)|bewerbungsfrist (ist )?(abgelaufen|beendet)|"
    r"(ausschreibung|bewerbungsphase) (ist )?(beendet|abgeschlossen)|bewerbungen werden nicht mehr angenommen|"
    r"keine bewerbungen mehr|diese stelle (ist|wurde) (leider )?(nicht mehr|bereits)", re.I)

_DEADLINE_LABEL = r"(bewerbungs(frist|schluss|ende)|bewerbungen bis( zum)?|bewerben sie sich bis( zum)?|einsendeschluss|" \
                  r"application deadline|apply by|closing date|deadline( for applications)?|applications? (close|due))"
_MONTHS = {m: i for i, m in enumerate(
    ["januar", "februar", "märz", "april", "mai", "juni", "juli", "august", "september", "oktober", "november", "dezember"], 1)}
_MONTHS.update({m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)})


@dataclass
class Liveness:
    status: str  # open | closed | unknown
    reason: str = ""
    url: str | None = None

    @property
    def closed(self) -> bool:
        return self.status == "closed"


def deadline_passed(text: str, today: date | None = None) -> date | None:
    """The application deadline named in the posting, if it is in the past."""
    today = today or date.today()
    for m in re.finditer(_DEADLINE_LABEL + r"[^0-9a-zäöü]{0,12}(?:dem\s+)?(.{0,30})", text or "", re.I):
        when = _parse_date(m.group(m.lastindex))
        if when:
            return when if when < today else None
    return None


def _parse_date(s: str) -> date | None:
    s = s.strip().lower()
    for rx, order in ((r"(\d{1,2})\.\s?(\d{1,2})\.\s?(\d{4})", "dmy"), (r"(\d{4})-(\d{2})-(\d{2})", "ymd"),
                      (r"(\d{1,2})/(\d{1,2})/(\d{4})", "dmy")):
        m = re.match(rx, s)
        if m:
            a, b, c = (int(x) for x in m.groups())
            try:
                return date(a, b, c) if order == "ymd" else date(c, b, a)
            except ValueError:
                return None
    m = re.match(r"(\d{1,2})\.?\s+([a-zä]+)\s+(\d{4})", s) or None
    if m and m.group(2) in _MONTHS:
        try:
            return date(int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))
        except ValueError:
            return None
    m = re.match(r"([a-z]+)\s+(\d{1,2}),?\s+(\d{4})", s)
    if m and m.group(1) in _MONTHS:
        try:
            return date(int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2)))
        except ValueError:
            return None
    return None


def _ats_check(job, http) -> Liveness | None:
    """Ask the ATS whether the job still exists (None = this source has no cheap check)."""
    from app.core.http import FetchError

    sources = {s.source: s for s in getattr(job, "sources", []) or []}
    tc = getattr(job, "target_company", None)
    ident = (tc.ats_identifier if tc else None) or ""
    try:
        if "greenhouse" in sources and ident and job.external_job_id:
            region = (tc.ats_config or {}).get("region") if tc else None
            base = "https://boards-api.eu.greenhouse.io" if region == "eu" else "https://boards-api.greenhouse.io"
            http.get_json(f"{base}/v1/boards/{ident}/jobs/{job.external_job_id}")
            return Liveness("open", "still on the Greenhouse board")
        if "lever" in sources and ident and job.external_job_id:
            http.get_json(f"https://api.lever.co/v0/postings/{ident}/{job.external_job_id}")
            return Liveness("open", "still on the Lever board")
        if "smartrecruiters" in sources and ident and job.external_job_id:
            http.get_json(f"https://api.smartrecruiters.com/v1/companies/{ident}/postings/{job.external_job_id}")
            return Liveness("open", "still on SmartRecruiters")
        if "workday" in sources and ident and job.external_job_id:
            host_part, _, site = ident.partition("/")
            tenant, _, wd = host_part.partition(".")
            data = http.get_json(f"https://{tenant}.{wd}.myworkdayjobs.com/wday/cxs/{tenant}/{site}{job.external_job_id}")
            return Liveness("open" if (data or {}).get("jobPostingInfo") else "closed", "Workday posting")
        if "arbeitsagentur" in sources and job.external_job_id:
            ref = base64.b64encode(job.external_job_id.encode()).decode()
            http.get_json_with_headers(f"https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobdetails/{ref}",
                                       headers={"X-API-Key": "jobboerse-jobsuche"})
            return Liveness("open", "still listed by the Bundesagentur")
        for name, url, key in (("ashby", f"https://api.ashbyhq.com/posting-api/job-board/{ident}", "jobs"),
                               ("workable", f"https://apply.workable.com/api/v1/widget/accounts/{ident}", "jobs"),
                               ("recruitee", f"https://{ident}.recruitee.com/api/offers/", "offers")):
            if name in sources and ident and job.external_job_id:
                items = http.get_json(url).get(key) or []
                ids = {str(i.get("id") or i.get("shortcode") or "") for i in items}
                return Liveness("open" if job.external_job_id in ids else "closed",
                                f"{'still' if job.external_job_id in ids else 'no longer'} on the {name.title()} board")
    except FetchError as exc:
        if re.search(r"HTTP (404|410)\b", str(exc)):
            return Liveness("closed", "the employer's job board no longer has this posting")
        return None
    except Exception as exc:  # noqa: BLE001 - a check must never break the run
        log.info("ATS liveness check failed for job %s: %s", getattr(job, "id", "?"), exc)
        return None
    return None


def _page_check(url: str, http) -> Liveness | None:
    from app.core.http import FetchError

    try:
        html = http.get_text(url)
    except FetchError as exc:
        if re.search(r"HTTP (404|410)\b", str(exc)):
            return Liveness("closed", "the posting page is gone (HTTP 404/410)", url)
        return None  # 403 / 429 / timeouts: cannot tell
    text = clean_text(html)[:40000]
    m = CLOSED_PHRASES.search(text)
    if m:
        return Liveness("closed", f"the posting page says “{m.group(0)}”", url)
    passed = deadline_passed(text)
    if passed:
        return Liveness("closed", f"the application deadline ({passed:%d.%m.%Y}) has passed", url)
    return Liveness("open", "the posting page is still online", url)


def check(job, http=None) -> Liveness:
    """open / closed / unknown — see the module docstring."""
    from app.core.http import get_http

    http = http or get_http()
    if not getattr(job, "active", True):
        return Liveness("closed", "its job feed no longer lists it")
    passed = deadline_passed(" ".join(x or "" for x in (job.description, job.requirements, job.responsibilities)))
    if passed:
        return Liveness("closed", f"the application deadline ({passed:%d.%m.%Y}) has passed")
    ats = _ats_check(job, http)
    if ats:
        return ats
    for url in dict.fromkeys(u for u in (job.career_page_url if _looks_like_posting(job.career_page_url) else None,
                                         job.source_url, job.canonical_url) if u):
        if re.search(r"indeed\.|stepstone\.|glassdoor\.", url):
            continue  # these block automated requests: no signal either way
        result = _page_check(url, http)
        if result:
            return result
    return Liveness("unknown", "could not be checked — the agents run anyway")


def _looks_like_posting(url: str | None) -> bool:
    """career_page_url is sometimes a whole careers page (lists every job) — only a posting URL tells anything."""
    return bool(url) and bool(re.search(r"\d{4,}|/jobs?/[^/]+|/stellen\w*/[^/]+|/o/[a-z0-9-]+|/offer/", url or "", re.I))


def mark_closed(session, job, result: Liveness) -> None:
    """Closed posting: the job is set to Expired (with the reason), so it leaves your lists."""
    from app.enums import JobStatus
    from app.services.jobs import set_status

    raw = dict(job.raw_source_data or {})
    raw["liveness"] = {"status": result.status, "reason": result.reason, "at": utcnow().isoformat(timespec="seconds")}
    job.raw_source_data = raw
    if job.status in (JobStatus.NEW, JobStatus.REVIEW, JobStatus.PRIORITY, JobStatus.APPLYING):
        set_status(session, job, JobStatus.EXPIRED, note=f"Posting closed: {result.reason}")
