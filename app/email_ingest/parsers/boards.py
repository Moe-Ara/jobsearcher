"""Alert parsers for LinkedIn, Indeed, StepStone, Bundesagentur für Arbeit and a generic fallback.

Email layouts change; these parsers rely on stable job-URL shapes. Tune them with real samples using
the .eml upload (preview mode) on the "Add" page.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qs, urlsplit

from app.email_ingest.parsers.base import EmailJobAlertParser, split_company_location
from app.enums import SourceType
from app.sources.base import JobPosting


class LinkedInAlertParser(EmailJobAlertParser):
    name = "linkedin"
    source = "linkedin"
    sender_patterns = ["linkedin.com"]

    @staticmethod
    def job_id(url: str) -> str | None:
        m = re.search(r"linkedin\.com/(?:comm/)?jobs/view/(?:[^/?]*-)?(\d{6,})", url)
        return m.group(1) if m else None

    def parse(self, html: str, text: str = "") -> list[JobPosting]:
        out = []
        for card in self.cards_by_link(html, self.job_id):
            company, location, snippet = split_company_location(card.lines)
            out.append(self.posting(
                job_id=card.job_id, url=f"https://www.linkedin.com/jobs/view/{card.job_id}",
                title=card.title, company=company, location=location, snippet=snippet,
            ))
        return out


def _card_postings(parser: EmailJobAlertParser, cards, id_prefix: str) -> list[JobPosting]:
    """Turn title-anchored cards into postings. Opaque tracking links carry no job id → stable content hash."""
    out = []
    for card in cards:
        company, location = card.lines[0], (card.lines[1] if len(card.lines) > 1 else "")
        snippet_lines = [ln for ln in card.lines[2:12] if not re.match(r"^(vorschau der stellenbeschreibung|gehalt|anstellungsarten|arbeitszeiten|leistungen)$", ln, re.I)]
        key = f"{card.title}|{company}|{location}".lower()
        job_id = id_prefix + hashlib.sha1(key.encode()).hexdigest()[:16]
        posting = parser.posting(job_id=job_id, url=card.url, title=card.title, company=company, location=location,
                                 snippet=" · ".join(snippet_lines))
        posting.raw["truncated_title"] = card.title.endswith(("...", "…")) or len(card.title) < 6
        out.append(posting)
    return out


_JOB_KEY = re.compile(r"(?:[?&](?:jk|vjk)=|jrtk=[^&#]*-)([0-9a-f]{16})(?=[&#]|$)")
_SALARY = re.compile(r"(€|\beur\b|\$|pro (jahr|monat|stunde)|per (year|month|hour)|/\s*(jahr|monat|stunde|year|hour))", re.I)
_NOT_A_PLACE = re.compile(r"^(stellenbeschreibung|job description|vollzeit|teilzeit|festanstellung|befristet|"
                          r"wurde seit .*|mindestgrundgehalt|arbeitsort|möchten sie .*|mehr erfahren|gehalt|"
                          r"vorschau der stellenbeschreibung|aktiver arbeitgeber|einfach bewerben|leistungen)$", re.I)
_YEARLY = re.compile(r"([\d.]{4,})\s*€(?:\s*[–-]\s*([\d.]{4,})\s*€)?\s*pro jahr", re.I)


def yearly_salary(line: str) -> tuple[float | None, float | None]:
    """'55.000 € – 65.000 € pro Jahr' → (55000, 65000); other periods are left in the text."""
    m = _YEARLY.search(line or "")
    if not m:
        return None, None
    lo = float(m.group(1).replace(".", ""))
    return lo, float(m.group(2).replace(".", "")) if m.group(2) else lo


def indeed_job_key(url: str) -> str | None:
    """Indeed's 16-hex job key: in jk= / vjk=, at the end of jrtk=, or inside a cts.indeed.com/v3/<base64 gzip JSON>
    tracking wrapper (decoded offline — Indeed itself blocks automated requests)."""
    import base64
    import gzip
    import html as html_lib
    import json

    url = html_lib.unescape(url or "")
    if m := _JOB_KEY.search(url):
        return m.group(1)
    if "/v3/" in url:
        token = url.split("/v3/", 1)[1].split("?")[0].split("/")[0]
        try:
            inner = json.loads(gzip.decompress(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))))
        except Exception:  # noqa: BLE001 - truncated / foreign payload: no job key
            return None
        if m := _JOB_KEY.search(str(inner.get("u", ""))):
            return m.group(1)
    return None


class IndeedAlertParser(EmailJobAlertParser):
    """Indeed alert + "match" emails. Links are cts.indeed.com tracking URLs whose payload (base64 gzip JSON) holds
    the job's ad-click URL; its jrtk parameter ends with the job key. A card counts as a job ONLY if its link yields a
    job key — that drops the e-mail's UI cards ("Mindestgrundgehalt", "Arbeitsort", "Anmelden", "Profil überprüfen")."""

    name = "indeed"
    source = "indeed"
    sender_patterns = ["match.indeed.com", "alert.indeed.com", "alert@indeed", "jobalert.indeed", "jobalerts@indeed",
                       "donotreply@jobalert.indeed.com", "indeedemail.com"]

    @staticmethod
    def job_id(url: str) -> str | None:
        return indeed_job_key(url)

    def parse(self, html: str, text: str = "") -> list[JobPosting]:
        out, seen = [], set()
        for card in self.cards_by_title_anchor(html):
            jk = indeed_job_key(card.url)
            if not jk or jk in seen:
                continue
            seen.add(jk)
            company, location = card.lines[0], ""
            if " · " in company:  # "Plath Solutions GmbH · 82205 Gilching"
                company, location = (x.strip() for x in company.split(" · ", 1))
            details, salary = [], ""
            for line in card.lines[1:12]:
                if _SALARY.search(line) and not salary:
                    salary = line
                elif not location and not _NOT_A_PLACE.match(line) and len(line) <= 60:
                    location = line
                elif not _NOT_A_PLACE.match(line):
                    details.append(line)
            lo, hi = yearly_salary(salary)
            snippet = " · ".join(([f"Gehalt: {salary}"] if salary and lo is None else []) + details)
            p = self.posting(job_id=jk, url=f"https://de.indeed.com/viewjob?jk={jk}", title=card.title, company=company,
                             location=location, snippet=snippet)
            if lo is not None:
                p.salary_min, p.salary_max, p.currency = lo, hi, "EUR"
            p.raw["truncated_title"] = card.title.endswith(("...", "…"))
            out.append(p)
        return out


_STEPSTONE_UI = re.compile(r"^(weil du|diesen job melden|du passt|erhöhe deine|mehr jobs|alle jobs|jetzt bewerben|"
                          r"top-fähigkeiten|bei diesem job|diese e-mail|©|job merken|zum job|profil)", re.I)
_MAGIC_LINK = re.compile(r"magiclink|/login|/account|token=", re.I)


class StepStoneAlertParser(EmailJobAlertParser):
    """StepStone job-agent emails ("X und N andere Firmen suchen Kandidaten wie Dich"). Links are click.stepstone.de
    tracking URLs (some resolve to personal auto-login "magic links" — never followed by the app); titles are often
    truncated with "..." in the email (the enrichment tool later finds the full posting)."""

    name = "stepstone"
    source = "stepstone"
    sender_patterns = ["stepstone.de", "stepstone.com", "stepstone.at"]

    @staticmethod
    def job_id(url: str) -> str | None:
        m = re.search(r"--(\d{6,})(?:-inline)?\.html", url) or re.search(r"/(?:job|stellenangebote?)[^?]*?[/-](\d{6,})", url)
        return m.group(1) if m and "stepstone" in url else None

    def parse(self, html: str, text: str = "") -> list[JobPosting]:
        out = []
        cards = [c for c in self.cards_by_title_anchor(html)
                 if not _STEPSTONE_UI.match(c.title) and not _STEPSTONE_UI.match(c.lines[0] if c.lines else "")]
        for p in _card_postings(self, cards, "t"):
            sid = self.job_id(p.url or "")
            if sid:
                p.external_job_id, p.url = sid, p.url.split("?")[0]
            elif _MAGIC_LINK.search(p.url or ""):
                p.url = None  # a personal login link is never stored as the job's URL
            out.append(p)
        return out


class ArbeitsagenturAlertParser(EmailJobAlertParser):
    name = "arbeitsagentur"
    source = "arbeitsagentur"
    sender_patterns = ["arbeitsagentur.de"]

    @staticmethod
    def job_id(url: str) -> str | None:
        m = re.search(r"arbeitsagentur\.de/jobsuche/jobdetail/([0-9A-Za-z-]+)", url)
        return m.group(1) if m else None

    def parse(self, html: str, text: str = "") -> list[JobPosting]:
        out = []
        for card in self.cards_by_link(html, self.job_id):
            company, location, snippet = split_company_location(card.lines)
            out.append(self.posting(
                job_id=card.job_id, url=f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{card.job_id}",
                title=card.title, company=company, location=location, snippet=snippet,
            ))
        return out


class GenericAlertParser(EmailJobAlertParser):
    """Fallback for senders listed in preferences.email.generic_senders (company newsletters, other boards)."""

    name = "generic"
    source = "email"
    source_type = SourceType.AGGREGATOR
    sender_patterns: list[str] = []
    _JOBISH = re.compile(r"/(jobs?|stellen\w*|careers?|karriere|vacanc\w*|positions?|openings?)(/|$|\?|-)", re.I)

    def __init__(self, senders: list[str] | None = None) -> None:
        self.sender_patterns = [s.lower() for s in (senders or [])]

    def job_id(self, url: str) -> str | None:
        if not url.startswith("http") or not self._JOBISH.search(url):
            return None
        return hashlib.sha1(url.split("?")[0].encode()).hexdigest()[:16]

    def parse(self, html: str, text: str = "") -> list[JobPosting]:
        out = []
        for card in self.cards_by_link(html, self.job_id):
            if not (8 <= len(card.title) <= 140):
                continue
            company, location, snippet = split_company_location(card.lines)
            out.append(self.posting(job_id=card.job_id, url=card.url, title=card.title, company=company, location=location, snippet=snippet))
        return out


class _BoardParser(EmailJobAlertParser):
    """Job boards that don't allow automated searches (Gulf boards, Xing, the StepStone group's UK boards, Glassdoor …):
    their job-alert e-mails are how their jobs reach you. Direct job links carry the board's job id (`url_rx` with an
    `id` group); tracking links fall back to title-anchored cards."""

    source_type = SourceType.JOB_BOARD
    url_rx: re.Pattern[str]
    canonical: str = ""

    def job_id(self, url: str) -> str | None:
        m = self.url_rx.search(url or "")
        return m.group("id") if m else None

    def parse(self, html: str, text: str = "") -> list[JobPosting]:
        out = []
        for card in self.cards_by_link(html, self.job_id):
            company, location, snippet = split_company_location(card.lines)
            m = self.url_rx.search(card.url)
            url = (self.canonical.format(**m.groupdict()) if self.canonical and m else card.url.split("?")[0])
            out.append(self.posting(job_id=card.job_id, url=url, title=card.title, company=company,
                                    location=location, snippet=snippet))
        if out:
            return out
        return [p for p in _card_postings(self, self.cards_by_title_anchor(html), self.name[:2] + "-")
                if 6 <= len(p.title) <= 140]


_GulfBoardParser = _BoardParser  # older name


class BaytAlertParser(_BoardParser):
    """bayt.com/en/uae/jobs/<title>-<company>-<city>-74414826/"""

    name = "bayt"
    source = "bayt"
    sender_patterns = ["bayt.com"]
    url_rx = re.compile(r"bayt\.com/(?P<lang>[a-z]{2})/(?P<country>[a-z-]+)/jobs/(?P<slug>[a-z0-9-]+?)-(?P<id>\d{6,})/?", re.I)
    canonical = "https://www.bayt.com/{lang}/{country}/jobs/{slug}-{id}/"


class GulfTalentAlertParser(_GulfBoardParser):
    """gulftalent.com/uae/jobs/senior-software-engineer-java-full-stack-577125"""

    name = "gulftalent"
    source = "gulftalent"
    sender_patterns = ["gulftalent.com"]
    url_rx = re.compile(r"gulftalent\.com/(?:mobile/)?(?P<country>[a-z-]+)/jobs/(?P<slug>(?!title/)[a-z0-9-]+?)-(?P<id>\d{5,})(?![\w-])", re.I)
    canonical = "https://www.gulftalent.com/{country}/jobs/{slug}-{id}"


class NaukriGulfAlertParser(_GulfBoardParser):
    """naukrigulf.com/<title>-jobs-in-<city>-<country>-in-<company>-...-jid-<id>"""

    name = "naukrigulf"
    source = "naukrigulf"
    sender_patterns = ["naukrigulf.com"]
    url_rx = re.compile(r"naukrigulf\.com/[^?\s\"']*?-jid-(?P<id>\d{6,})", re.I)


def _board(name: str, senders: list[str], url_rx: str, doc: str) -> type[_BoardParser]:
    """A job-alert parser for one more board: its sender domains and the shape of its job links."""
    return type(f"{name.title().replace('_', '')}AlertParser", (_BoardParser,), {
        "__doc__": doc, "name": name, "source": name, "sender_patterns": senders, "url_rx": re.compile(url_rx, re.I)})


# Germany and the UK — boards whose terms don't allow automated searches; their alert e-mails do. The link shapes are
# taken from the boards' public job URLs; anything else in an alert still comes through the title-anchored cards.
XingAlertParser = _board("xing", ["xing.com"], r"xing\.com/jobs/[a-z0-9-]*?-?(?P<id>\d{6,})(?![\w-])", "xing.com/jobs/<slug>-<id>")
TotaljobsAlertParser = _board("totaljobs", ["totaljobs.com"], r"totaljobs\.com/job/[^\s\"'?#]*?job(?P<id>\d{6,})",
                              "totaljobs.com/job/<title>/<company>-job<id> (StepStone group)")
CwjobsAlertParser = _board("cwjobs", ["cwjobs.co.uk"], r"cwjobs\.co\.uk/job/[^\s\"'?#]*?job(?P<id>\d{6,})",
                           "cwjobs.co.uk/job/<title>/<company>-job<id> (IT jobs, StepStone group)")
JobsiteAlertParser = _board("jobsite", ["jobsite.co.uk"], r"jobsite\.co\.uk/job/[^\s\"'?#]*?job(?P<id>\d{6,})",
                            "jobsite.co.uk/job/<title>/<company>-job<id>")
ReedAlertParser = _board("reed", ["reed.co.uk"], r"reed\.co\.uk/jobs/[a-z0-9-]+/(?P<id>\d{6,})", "reed.co.uk/jobs/<title>/<id>")
CvLibraryAlertParser = _board("cvlibrary", ["cv-library.co.uk"], r"cv-library\.co\.uk/job/(?P<id>\d{6,})",
                              "cv-library.co.uk/job/<id>/<title>")
GlassdoorAlertParser = _board("glassdoor", ["glassdoor."], r"glassdoor\.[a-z.]+/[^\s\"']*?[?&](?:jl|jobListingId)=(?P<id>\d{6,})",
                              "glassdoor.*/job-listing/…?jl=<id>")
MonsterAlertParser = _board("monster", ["monster.de", "monster.co.uk", "monster.com"],
                            r"monster\.[a-z.]+/(?:stellenangebot|job-openings|jobs/[a-z-]+)/[^\s\"'?#]*?--(?P<id>[0-9a-f-]{20,})",
                            "monster.*/stellenangebot/<slug>--<uuid>")
JobsAcUkAlertParser = _board("jobs_ac_uk", ["jobs.ac.uk"], r"jobs\.ac\.uk/job/(?P<id>[A-Z0-9]{4,})",
                             "jobs.ac.uk/job/<ID>/<slug> (UK academic jobs, PhD studentships)")
GuardianJobsAlertParser = _board("guardian_jobs", ["jobs.theguardian.com", "guardianjobs"],
                                 r"jobs\.theguardian\.com/job/(?P<id>\d{5,})", "jobs.theguardian.com/job/<id>/<slug>")
WttjAlertParser = _board("wttj", ["welcometothejungle.com", "otta.com"],
                         r"welcometothejungle\.com/[a-z]{2}/companies/[^/\s]+/jobs/(?P<id>[a-z0-9_-]{6,})",
                         "welcometothejungle.com/<lang>/companies/<company>/jobs/<slug> (Otta)")
StellenanzeigenAlertParser = _board("stellenanzeigen", ["stellenanzeigen.de"], r"stellenanzeigen\.de/job/[^\s\"'?#]*?-(?P<id>\d{6,})",
                                    "stellenanzeigen.de/job/<slug>-<id>/")
JobwareAlertParser = _board("jobware", ["jobware.de"], r"jobware\.de/[^\s\"'?#]*?(?P<id>\d{6,})", "jobware.de/…/<id>")
GetInItAlertParser = _board("get_in_it", ["get-in-it.de"], r"get-in-it\.de/jobsuche/p(?P<id>\d{4,})", "get-in-it.de/jobsuche/p<id>")

BOARD_PARSERS = [XingAlertParser, TotaljobsAlertParser, CwjobsAlertParser, JobsiteAlertParser, ReedAlertParser,
                 CvLibraryAlertParser, GlassdoorAlertParser, MonsterAlertParser, JobsAcUkAlertParser,
                 GuardianJobsAlertParser, WttjAlertParser, StellenanzeigenAlertParser, JobwareAlertParser, GetInItAlertParser]


def all_parsers(enabled: list[str], generic_senders: list[str]) -> list[EmailJobAlertParser]:
    parsers: list[EmailJobAlertParser] = [
        LinkedInAlertParser(), IndeedAlertParser(), StepStoneAlertParser(), ArbeitsagenturAlertParser(),
        BaytAlertParser(), GulfTalentAlertParser(), NaukriGulfAlertParser(), *(cls() for cls in BOARD_PARSERS),
        GenericAlertParser(generic_senders),
    ]
    return [p for p in parsers if p.name in enabled]


def pick_parser(parsers: list[EmailJobAlertParser], from_addr: str, subject: str) -> EmailJobAlertParser | None:
    return next((p for p in parsers if p.matches(from_addr, subject)), None)
