"""Email job-alert parsing.

Alert emails are HTML lists of "job cards". Rather than depending on fragile, frequently-changing CSS/table
structure, parsers anchor on the job *links* (whose URL shapes are stable), then read the card text around
each link: title = link text, followed by company and location lines.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar
from urllib.parse import parse_qs, unquote, urlsplit

from bs4 import BeautifulSoup, NavigableString, Tag

from app.core.text import clean_line
from app.enums import SourceType
from app.sources.base import JobPosting

# Lines in alert cards that are UI chrome, not job data.
NOISE_RE = re.compile(
    r"^(new|neu|promoted|gesponsert|sponsored|easy apply|einfach bewerben|schnellbewerbung|actively recruiting|"
    r"be an early applicant|apply now|jetzt bewerben|view job|job ansehen|see (all|more) jobs?|alle jobs|"
    r"view|ansehen|anzeigen|details|more|mehr|mehr erfahren|learn more|show|apply|bewerben|anmelden|sign in|log in|"
    r"\d+\s+(applicants?|bewerber\w*|connections?|kontakte?)|.*alumni.*|top applicant|"
    r"\d+\s*(hours?|days?|weeks?|stunden?|tagen?|wochen?)\s*(ago)?|vor\s+\d+.*|today|heute|gestern|yesterday|"
    r"just posted|gerade veröffentlicht|urgently hiring|responsive employer|save|speichern|"
    r"€.*|\$.*|ab\s+\d.*|bis\s+\d.*|\d[\d.,]*\s*(€|eur|k)\b.*|full[- ]time|part[- ]time|vollzeit|teilzeit|"
    r"festanstellung|befristet|unbefristet|homeoffice möglich|remote|hybrid|on-site|vor ort)$",
    re.IGNORECASE,
)


NON_ALERT_SUBJECT = re.compile(
    r"bewerbung|application|angesehen|erhöhe deine chancen|sign in|anmelde|login|\bcode\b|passwort|password|appeal"
    r"|einspruch|bestätig|confirm|sicherheit|security|verifiz|verify|interview",
    re.IGNORECASE,
)

# Link texts that are buttons/footer items in alert emails, never job titles.
UI_LABELS = {
    s.casefold() for s in [
        "Job anzeigen", "Passt nicht", "Mehr erfahren", "Profil bearbeiten", "Diese E-Mails pausieren", "Hilfebereich",
        "Indeed", "Datenschutzerklärung", "Nutzungsbedingungen", "Abbestellen", "Weitere passende Jobs anzeigen",
        "E-Mail-Einstellungen verwalten", "Diese E-Mail abbestellen", "Kontakt", "Impressum", "Jetzt bewerben",
        "Bewerben", "Apply", "Apply now", "View job", "View all jobs", "See all jobs", "See more jobs", "Unsubscribe",
        "Privacy Policy", "Terms", "Help", "Manage alerts", "Alle Jobs anzeigen", "Job speichern", "Speichern",
        "Passt hervorragend", "Passt gut", "Neu", "Stepstone", "LinkedIn", "Jobs ansehen", "Einstellungen",
    ]
}
_CARD_NOISE = re.compile(
    r"^(\d[\d.,]*\s*[-–+]?\s*\d*[\d.,]*\+?|mitarbeiter(innen)?|employees|aktiver arbeitgeber|active employer|"
    r"bewerbungs-update verfügbar|passt (hervorragend|gut|nicht)|neu|new|top|gesponsert|sponsored|"
    r"andere bewerber interessierten sich auch für diese jobs)$",
    re.IGNORECASE,
)


def looks_like_title(text: str) -> bool:
    t = text.strip()
    return 6 <= len(t) <= 160 and t.casefold() not in UI_LABELS and not NOISE_RE.match(t) and not t.lower().startswith("http")


def _is_card_noise(text: str) -> bool:
    return text.casefold() in UI_LABELS or bool(_CARD_NOISE.match(text.strip()))


@dataclass
class ParsedCard:
    job_id: str
    url: str
    title: str
    lines: list[str] = field(default_factory=list)


class EmailJobAlertParser(ABC):
    name: ClassVar[str]
    source: ClassVar[str]
    source_type: ClassVar[SourceType] = SourceType.JOB_BOARD
    sender_patterns: ClassVar[list[str]] = []

    def matches(self, from_addr: str, subject: str) -> bool:
        f = (from_addr or "").lower()
        if NON_ALERT_SUBJECT.search(subject or ""):
            return False  # login codes, application updates, appeals... are not job alerts
        return any(p in f for p in self.sender_patterns)

    @abstractmethod
    def parse(self, html: str, text: str = "") -> list[JobPosting]:
        """Extract every job in the email (an alert usually contains several)."""

    # ---- shared helpers -------------------------------------------------------------------

    def cards_by_title_anchor(self, html: str) -> list[ParsedCard]:
        """For alerts whose links are opaque tracking URLs (Indeed, StepStone): a job card is a link whose text is a
        job title, followed by company and location text lines before the next job title link."""
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["style", "script", "head", "title"]):
            tag.decompose()
        nodes: list[tuple[str, Tag | None]] = []  # (line, enclosing <a> or None) in document order
        for s in soup.find_all(string=True):
            if type(s) is not NavigableString:  # skip comments (Outlook "[if mso]" blocks), CDATA, doctype...
                continue
            line = clean_line(str(s))
            if line and re.search(r"[^\W\d_]", line):
                nodes.append((line, s.find_parent("a")))
        cards: list[ParsedCard] = []
        seen: set[tuple[str, str]] = set()
        for i, (line, anchor) in enumerate(nodes):
            if anchor is None or not looks_like_title(line):
                continue
            if i > 0 and nodes[i - 1][1] is anchor:  # only the first text node of a link
                continue
            following = []
            for nxt, nxt_anchor in nodes[i + 1 : i + 30]:
                if nxt_anchor is not None and nxt_anchor is not anchor and looks_like_title(nxt):
                    break  # next card begins
                if nxt_anchor is anchor or _is_card_noise(nxt):
                    continue
                following.append(nxt)
            if len(following) < 2:
                continue  # footer / feedback links have no company + location after them
            title = re.sub(r"\s*(\.\.\.|…)\s*$", "", line)
            key = (title.lower(), following[0].lower())
            if key in seen:
                continue
            seen.add(key)
            cards.append(ParsedCard(job_id="", url=unwrap_redirect(anchor.get("href", "")), title=title, lines=following))
        return cards

    def cards_by_link(self, html: str, id_from_url) -> list[ParsedCard]:
        """Group anchors by job id; read the smallest enclosing block that holds exactly one job."""
        soup = BeautifulSoup(html, "lxml")
        anchors: dict[str, list[Tag]] = {}
        order: list[str] = []
        for a in soup.find_all("a", href=True):
            job_id = id_from_url(unwrap_redirect(a["href"]))
            if not job_id:
                continue
            if job_id not in anchors:
                anchors[job_id] = []
                order.append(job_id)
            anchors[job_id].append(a)

        cards: list[ParsedCard] = []
        for job_id in order:
            links = anchors[job_id]
            texts = [clean_line(a.get_text(" ")) for a in links]
            title = max((t for t in texts if t and not NOISE_RE.match(t)), key=len, default="")
            container = _card_container(links[0], job_id, id_from_url)
            lines = _card_lines(container) if container is not None else []
            if not title and lines:
                title = lines[0]
            if not title:
                continue
            # lines after the title (company, location, snippet...)
            if title in lines:
                lines = lines[lines.index(title) + 1 :]
            lines = [ln for ln in lines if ln != title]
            cards.append(ParsedCard(job_id, unwrap_redirect(links[0]["href"]), title, lines))
        return cards

    def posting(self, *, job_id: str, url: str, title: str, company: str, location: str, snippet: str = "") -> JobPosting:
        return JobPosting(
            source=self.source, source_type=self.source_type, external_job_id=job_id, url=url,
            title=clean_line(title), company=clean_line(company), location=clean_line(location),
            description=snippet, raw={"parser": self.name},
        )


def unwrap_redirect(href: str) -> str:
    """Follow common tracking-redirect encodings (?url=, ?u=, ?dest=...) without making requests."""
    href = href.strip()
    for _ in range(3):
        qs = parse_qs(urlsplit(href).query)
        target = next((qs[k][0] for k in ("url", "u", "dest", "destination", "target", "redirect", "r") if k in qs and qs[k][0].startswith("http")), None)
        if not target:
            break
        href = unquote(target)
    return href


def _card_container(anchor: Tag, job_id: str, id_from_url) -> Tag | None:
    node: Tag | None = anchor
    best = None
    for _ in range(10):
        node = node.parent if node is not None else None
        if node is None or node.name in ("body", "html", "[document]"):
            break
        ids = {id_from_url(unwrap_redirect(a["href"])) for a in node.find_all("a", href=True)}
        ids.discard(None)
        if ids - {job_id}:
            break  # grew into a neighbouring card
        best = node
        if len(_card_lines(node)) >= 4:
            break
    return best


def _card_lines(node: Tag) -> list[str]:
    lines = []
    for s in node.stripped_strings:
        line = clean_line(s)
        if line and len(line) > 1 and not NOISE_RE.match(line) and line not in lines:
            lines.append(line)
    return lines


def split_company_location(lines: list[str]) -> tuple[str, str, str]:
    """Heuristic: 'Company · Location' on one line, or company then location on separate lines."""
    if not lines:
        return "", "", ""
    first = lines[0]
    for sep in (" · ", " • ", " | ", " – ", " - "):
        if sep in first:
            company, location = first.split(sep, 1)
            return company.strip(), location.strip(), " ".join(lines[1:3])
    company = first
    location = lines[1] if len(lines) > 1 else ""
    return company, location, " ".join(lines[2:4])
