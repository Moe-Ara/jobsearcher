"""Which link to open for a job.

Email alerts (Indeed, StepStone) only contain tracking redirects (cts.indeed.com, click.stepstone.de). They
expire after a while and Indeed's wrap sponsored-ad click URLs, so opening them later often ends on "job no longer
available" or a bot/verification page. We therefore prefer, in order:
  1. a direct link from any non-tracking source of the same vacancy (employer feed, LinkedIn, ...);
  2. a search on the job board for this exact title + company + location (a normal page, no tracking token);
and always offer a web search (finds the employer's own posting) plus the original email link as a fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote_plus, urlsplit

from app.core.normalize import strip_accents
from app.enums import SOURCE_RANK

_TRACKING_HOSTS = ("cts.indeed.com", "click.stepstone.de", "click.linkedin.com", "email.stepstone.de", "links.indeed.com")
_TRACKING_PATHS = re.compile(r"/(pagead|rc)/clk|/f/a/", re.I)
_GENDER = re.compile(r"\(\s*[a-z/|,*\s.-]{1,15}\)|\b[mwfdx]\s*/\s*[mwfdx](\s*/\s*[mwfdx])?\b|\(?\b(all genders?|gn)\b\)?", re.I)


@dataclass
class JobLink:
    label: str
    url: str
    hint: str = ""


def is_tracking_url(url: str | None) -> bool:
    if not url:
        return True
    parts = urlsplit(url)
    return parts.netloc.lower().endswith(_TRACKING_HOSTS) or bool(_TRACKING_PATHS.search(parts.path))


def _clean_title(title: str) -> str:
    t = _GENDER.sub(" ", title or "")
    t = re.sub(r"\s*(\.\.\.|…)\s*$", "", t)
    return re.sub(r"\s+", " ", t).strip(" -–|")


def _city(location: str | None) -> str:
    loc = re.sub(r"\b\d{5}\b", " ", location or "")  # postcode
    loc = re.sub(r"\((on-site|hybrid|remote)\)", " ", loc, flags=re.I)
    first = re.split(r"[,;/|]", loc)[0]
    return re.sub(r"\s+", " ", first).strip()


def _company(company: str) -> str:
    from app.email_ingest.parsers.base import UI_LABELS

    return "" if (company or "").strip().casefold() in UI_LABELS else (company or "").strip()


def board_search_url(source: str, title: str, company: str, location: str | None) -> str | None:
    what = quote_plus(f"{_clean_title(title)} {_company(company)}".strip())
    where = quote_plus(_city(location))
    if source == "indeed":
        return f"https://de.indeed.com/jobs?q={what}&l={where}"
    if source == "stepstone":
        return f"https://www.stepstone.de/work/?ke={what}&ws={where}"
    return None


def web_search_url(title: str, company: str, location: str | None) -> str:
    q = f"{_clean_title(title)} {_company(company)} {_city(location)} Stellenangebot"
    return "https://www.google.com/search?q=" + quote_plus(strip_accents(q) if not q.isascii() else q)


def job_links(job) -> dict[str, object]:
    """{'primary': JobLink, 'extras': [JobLink...]} for templates/API."""
    sources = sorted(job.sources or [], key=lambda s: SOURCE_RANK.get(s.source_type, 9))
    title, company, location = job.title or "", job.company or "", job.location
    extras: list[JobLink] = []

    direct = next((s.url for s in sources if s.url and not is_tracking_url(s.url)), None)
    if direct is None and job.canonical_url and not is_tracking_url(job.canonical_url):
        direct = job.canonical_url
    if direct:
        primary = JobLink("Open job ↗", direct)
    else:
        board = next((s.source for s in sources if board_search_url(s.source, title, company, location)), None)
        if board:
            primary = JobLink(f"Find on {board.capitalize()} ↗", board_search_url(board, title, company, location),
                              "Searches the board for this title, company and location (the email link expires).")
        else:
            primary = JobLink("Search job ↗", web_search_url(title, company, location))

    extras.append(JobLink("Web search", web_search_url(title, company, location), "Finds the employer's own posting"))
    tracking = next((s.url for s in sources if s.url and is_tracking_url(s.url)), None)
    if tracking and primary.url != tracking:
        extras.append(JobLink("Email link", tracking, "Original alert link — may have expired or trigger a bot check"))
    return {"primary": primary, "extras": extras}
