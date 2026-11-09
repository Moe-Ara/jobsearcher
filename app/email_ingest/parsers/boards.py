"""Alert parsers for LinkedIn, Indeed, StepStone, Bundesagentur für Arbeit and a generic fallback.

Email layouts change; these parsers rely on stable job-URL shapes. Tune them with real samples using
the .eml upload (preview mode) on the "Add" page.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qs, urlsplit

from bs4 import BeautifulSoup

from app.core.text import clean_line
from app.email_ingest.parsers.base import EmailJobAlertParser, looks_like_title, split_company_location, unwrap_redirect
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
        cards = self.cards_by_title_anchor(html)
        page = re.sub(r"\s+", " ", BeautifulSoup(html, "lxml").get_text(" ")) if cards else ""
        for card in cards:
            jk = indeed_job_key(card.url)
            if not jk and card.lines and f"{card.title} bei {card.lines[0]}" in page:
                # A one-job "match" e-mail ("<title> bei <company>") links through an ad click without a job key:
                # the header line proves the card is the job, so it gets a stable content id instead.
                jk = "in-" + hashlib.sha1(f"{card.title}|{card.lines[0]}".lower().encode()).hexdigest()[:16]
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
            url = None if jk.startswith("in-") else f"https://de.indeed.com/viewjob?jk={jk}"
            p = self.posting(job_id=jk, url=url, title=card.title, company=company, location=location, snippet=snippet)
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
        return out or self._recommendation(html)

    def _recommendation(self, html: str) -> list[JobPosting]:
        """"Max, unsere Empfehlung: <title>" — one job as plain text above the "Ich bin interessiert" button (that
        button registers interest on StepStone: its link is never kept; the job keeps no URL)."""
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["style", "script", "head", "title"]):
            tag.decompose()
        lines = [clean_line(s) for s in soup.stripped_strings]
        lines = [ln for ln in lines if ln and re.search(r"[^\W\d_]", ln)]
        try:
            end = next(i for i, ln in enumerate(lines) if ln.lower() == "ich bin interessiert")
        except StopIteration:
            return []
        start = max((i for i in range(end) if len(lines[i]) > 70 or lines[i].lower().startswith("hallo ")), default=-1)
        card = lines[start + 1:end]
        if len(card) < 2 or not looks_like_title(card[0]):
            return []
        title, company = card[0], re.sub(r"\s*(\.\.\.|…)\s*$", "", card[1])
        location = card[2] if len(card) > 2 else ""
        details = [ln for ln in card[3:9] if not re.match(r"^vor \d+|^\d[\d.]* ?-? ?\d*[\d.]* ?mitarbeiter", ln, re.I)]
        p = self.posting(job_id="t" + hashlib.sha1(f"{title}|{company}|{location}".lower().encode()).hexdigest()[:16],
                         url=None, title=title, company=company, location=location, snippet=" · ".join(details))
        return [p]


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
    # Gulf boards wrap links in tracking redirects: their cards are read by title. For boards whose alert format was
    # never seen, only real job links count — an account / networking e-mail never becomes a "job".
    title_fallback: bool = True

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
        if out or not self.title_fallback:
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
        "__doc__": doc, "name": name, "source": name, "sender_patterns": senders, "url_rx": re.compile(url_rx, re.I),
        "title_fallback": False})


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

_NOT_COMPANY_OR_PLACE = re.compile(r"(€|\beur\b|/jahr|pro jahr|stunden|homeoffice|remote möglich|vollzeit|teilzeit|"
                                   r"geschätzt|\bhours?\b)", re.I)


class _AnchorCardParser(EmailJobAlertParser):
    """Alerts whose every job card is ONE link holding title, company and place as separate text pieces
    (jobvector, Workwise, Jobijoba). Links that carry a personal token are never stored as the job's URL."""

    source_type = SourceType.JOB_BOARD
    link_rx: re.Pattern[str]
    canonical: str | None = None  # format with the link's named groups; None → the job keeps no URL

    def parse(self, html: str, text: str = "") -> list[JobPosting]:
        out, seen = [], set()
        for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
            m = self.link_rx.search(unwrap_redirect(a["href"]))
            parts = [clean_line(s) for s in a.stripped_strings]
            parts = [p for p in parts if p]
            if not m or len(parts) < 2 or not looks_like_title(parts[0]):
                continue
            title, company = parts[0], parts[1]
            places = [p for p in parts[2:] if not _NOT_COMPANY_OR_PLACE.search(p) and re.search(r"[^\W\d_]{2}", p)]
            location = re.sub(r"\s*\+\d+$", "", places[0]) if places else ""
            groups = {k: v for k, v in m.groupdict().items() if v}
            job_id = groups.get("id") or hashlib.sha1(f"{title}|{company}|{location}".lower().encode()).hexdigest()[:16]
            if job_id in seen:
                continue
            seen.add(job_id)
            url = self.canonical.format(**groups) if self.canonical and "id" in groups else None
            details = [p for p in parts[2:] if p not in places[:1]]
            out.append(self.posting(job_id=f"{self.name[:3]}-{job_id}", url=url, title=title, company=company,
                                    location=location, snippet=" · ".join(details)))
        return out


class JobvectorAlertParser(_AnchorCardParser):
    """jobvector JobMail (STEM jobs): …/jobs/…?jobId=<hex>; recommendation links carry a personal token."""

    name = "jobvector"
    source = "jobvector"
    sender_patterns = ["jobvector.de"]
    link_rx = re.compile(r"jobvector\.de/[^\s\"']*?[?&]jobId=(?P<id>[0-9a-f]{10,})", re.I)
    canonical = "https://www.jobvector.de/jobs/?jobId={id}"


class WorkwiseAlertParser(_AnchorCardParser):
    """Workwise job suggestions: every link is a personal auto-login link (eTT=…) — cards are read, links never kept."""

    name = "workwise"
    source = "workwise"
    sender_patterns = ["workwise.io"]
    link_rx = re.compile(r"workwise\.io/(jobsuche|job|stellen)\b", re.I)


class JobijobaAlertParser(_AnchorCardParser):
    """Jobijoba job alerts: tracking links carry your e-mail address — cards are read, links never kept."""

    name = "jobijoba"
    source = "jobijoba"
    sender_patterns = ["jobijoba.com", "jobijoba.de"]
    link_rx = re.compile(r"jobijoba\.(com|de)/clic/", re.I)


class HaystackAlertParser(EmailJobAlertParser):
    """Haystack job alerts: a title link and a "Company · Hybrid · City, Country" link per job (…/go?j=<uuid>)."""

    name = "haystack"
    source = "haystack"
    source_type = SourceType.JOB_BOARD
    sender_patterns = ["haystackapp.io"]
    _LINK = re.compile(r"haystackapp\.io/go\?j=(?P<id>[0-9a-f-]{20,})", re.I)

    def parse(self, html: str, text: str = "") -> list[JobPosting]:
        jobs: dict[str, dict[str, str]] = {}
        for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
            m = self._LINK.search(unwrap_redirect(a["href"]))
            label = clean_line(a.get_text(" "))
            if not m or not label:
                continue
            job = jobs.setdefault(m.group("id"), {})
            if " · " in label:
                job.setdefault("meta", label)
            elif looks_like_title(label):
                job.setdefault("title", label)
        out = []
        for jid, job in jobs.items():
            if "title" not in job:
                continue
            bits = [b.strip() for b in job.get("meta", "").split(" · ")]
            company = bits[0] if bits else ""
            location = bits[-1] if len(bits) > 1 else ""
            work_mode = " · ".join(bits[1:-1])
            out.append(self.posting(job_id=jid, url=f"https://haystackapp.io/go?j={jid}", title=job["title"],
                                    company=company, location=location, snippet=work_mode))
        return out


class CompanyAlertParser(EmailJobAlertParser):
    """An employer's own job alert from its SAP SuccessFactors career site (jobs2web: research centres, universities, manufacturers …):
    <career host>/job/<slug>/<id>/ — the employer is named by the e-mail, not the card."""

    name = "company_alerts"
    source = "company_alert"
    source_type = SourceType.EMPLOYER
    sender_patterns = ["jobs2web.com"]
    _LINK = re.compile(r"(?P<host>[a-z0-9.-]+\.[a-z]{2,})/job/(?P<slug>[^/?\s\"']+)/(?P<id>\d{3,}(?:-[a-z]{2}_[a-z]{2})?)/?", re.I)
    _OWNER = re.compile(r"\b(?:bei der|bei|an der|der|von)\s+(?P<c>[A-ZÄÖÜ][^–:|]{2,60}?)(?:\s+[–:|-]\s|\s*$)")

    def employer(self, host: str) -> str:
        subject = self.message_meta.get("subject", "")
        m = self._OWNER.search(subject)
        if m and not re.search(r"stelle|job|benachrichtigung", m.group("c"), re.I):
            return m.group("c").strip()
        name = self.message_meta.get("from_name", "")
        if name and "@" not in name and not re.search(r"noreply|no-reply|notification", name, re.I):
            return name
        label = host.lower().removeprefix("www.").split(".")
        core = label[-2] if len(label) >= 2 else host
        return core.upper() if len(core) <= 4 else core.replace("-", " ").title()

    def parse(self, html: str, text: str = "") -> list[JobPosting]:
        out, seen = [], set()
        for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
            m = self._LINK.search(unwrap_redirect(a["href"]))
            title = clean_line(a.get_text(" "))
            if not m or m.group("id") in seen or not looks_like_title(title):
                continue
            seen.add(m.group("id"))
            host = m.group("host")
            out.append(self.posting(job_id=m.group("id"), url=f"https://{host}/job/{m.group('slug')}/{m.group('id')}/",
                                    title=title, company=self.employer(host), location=""))
        return out


EXTRA_PARSERS = [JobvectorAlertParser, WorkwiseAlertParser, JobijobaAlertParser, HaystackAlertParser, CompanyAlertParser]

BOARD_PARSERS = [XingAlertParser, TotaljobsAlertParser, CwjobsAlertParser, JobsiteAlertParser, ReedAlertParser,
                 CvLibraryAlertParser, GlassdoorAlertParser, MonsterAlertParser, JobsAcUkAlertParser,
                 GuardianJobsAlertParser, WttjAlertParser, StellenanzeigenAlertParser, JobwareAlertParser, GetInItAlertParser]


def all_parsers(enabled: list[str], generic_senders: list[str]) -> list[EmailJobAlertParser]:
    parsers: list[EmailJobAlertParser] = [
        LinkedInAlertParser(), IndeedAlertParser(), StepStoneAlertParser(), ArbeitsagenturAlertParser(),
        BaytAlertParser(), GulfTalentAlertParser(), NaukriGulfAlertParser(), *(cls() for cls in BOARD_PARSERS),
        *(cls() for cls in EXTRA_PARSERS), GenericAlertParser(generic_senders),
    ]
    return [p for p in parsers if p.name in enabled]


def pick_parser(parsers: list[EmailJobAlertParser], from_addr: str, subject: str) -> EmailJobAlertParser | None:
    return next((p for p in parsers if p.matches(from_addr, subject)), None)
