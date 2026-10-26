"""Hard filters — run before any LLM call. Pure functions over job fields + preferences.

Outcome:
  REJECTED  at least one hard rule matched (reason recorded, job hidden by default, never sent to LLM)
  FLAGGED   passes, but something needs a closer look (e.g. a mixed civil/defence company) — the LLM
            analysis gets an explicit defence check and can still reject
  PASSED    nothing matched
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from rapidfuzz import fuzz

from app.config_files import Preferences
from app.core.normalize import normalize_company
from app.enums import FilterStatus


class JobLike(Protocol):
    title: str
    company: str
    normalized_company: str
    location: str | None
    country: str | None
    remote_status: str | None
    description: str | None
    requirements: str | None
    responsibilities: str | None
    sector: str | None
    years_experience_required: float | None


@dataclass
class FilterOutcome:
    status: FilterStatus = FilterStatus.PASSED
    reasons: list[dict[str, str]] = field(default_factory=list)

    def reject(self, rule: str, detail: str) -> None:
        self.reasons.append({"rule": rule, "severity": "reject", "detail": detail})
        self.status = FilterStatus.REJECTED

    def flag(self, rule: str, detail: str) -> None:
        self.reasons.append({"rule": rule, "severity": "flag", "detail": detail})
        if self.status != FilterStatus.REJECTED:
            self.status = FilterStatus.FLAGGED

    @property
    def rejected(self) -> bool:
        return self.status == FilterStatus.REJECTED


@lru_cache(maxsize=256)
def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def _company_matches(normalized: str, names: list[str]) -> str | None:
    if not normalized:
        return None
    for name in names:
        n = normalize_company(name)
        if not n:
            continue
        if normalized == n or re.search(rf"\b{re.escape(n)}\b", normalized):
            return name
        if len(n) >= 6 and fuzz.ratio(normalized, n) >= 92:
            return name
    return None


def defence_check(job: JobLike, prefs: Preferences, out: FilterOutcome) -> None:
    d = prefs.defence
    company_norm = job.normalized_company or normalize_company(job.company)
    hit = _company_matches(company_norm, d.excluded_companies)
    if hit:
        out.reject("defence", f"Defence company ({hit})")
        return

    title = (job.title or "").lower()
    company = (job.company or "").lower()
    body = "\n".join(x for x in (job.description, job.requirements, job.responsibilities, job.sector) if x).lower()
    for phrase in d.benign_phrases:
        title = _rx(phrase).sub(" ", title)
        company = _rx(phrase).sub(" ", company)
        body = _rx(phrase).sub(" ", body)

    # The hiring entity itself is a defence business unit (e.g. "Airbus Defence and Space GmbH").
    for kw in d.strong_keywords:
        m = _rx(rf"\b{kw}\b").search(company)
        if m:
            out.reject("defence", f"Hiring company/business unit is defence-related: {job.company}")
            return

    for pattern in d.very_strong_patterns:
        m = _rx(pattern).search(title) or _rx(pattern).search(body)
        if m:
            out.reject("defence", f"Defence-related wording: “{m.group(0)}”")
            return
    for kw in d.strong_keywords:
        m = _rx(rf"\b{kw}\b").search(title)
        if m:
            out.reject("defence", f"Defence keyword in title: “{m.group(0)}”")
            return

    hits: list[str] = []
    for kw in d.strong_keywords:
        hits += [m.group(0) for m in _rx(rf"\b{kw}\b").finditer(body)]
    mixed = _company_matches(company_norm, d.mixed_companies)
    if len(hits) >= 3:
        out.reject("defence", f"Repeated defence keywords: {', '.join(sorted(set(hits)))}")
    elif hits:
        out.flag("defence_check", f"Mentions {', '.join(sorted(set(hits)))} — verify business unit")
    elif mixed:
        out.flag("defence_check", f"{mixed} has civil and defence business — verify business unit")


def location_check(job: JobLike, prefs: Preferences, out: FilterOutcome) -> None:
    loc = prefs.locations
    targets = {c.lower() for c in loc.target_countries}
    if job.country and job.country.lower() not in targets:
        if job.remote_status == "remote" and loc.remote_other_country != "reject":
            out.flag("country", f"Remote role based in {job.country}")
        else:
            out.reject("country", f"Country {job.country} not in target countries")
    if not job.country and loc.unknown_country in ("flag", "reject"):
        detail = f"Country unknown (location: {job.location or 'not given'}) — check before relying on it"
        out.reject("country", detail) if loc.unknown_country == "reject" else out.flag("country_unknown", detail)
    location = (job.location or "").lower()
    for bad in loc.excluded_locations:
        if bad and bad.lower() in location:
            out.reject("location", f"Excluded location: {bad}")


def title_check(job: JobLike, prefs: Preferences, out: FilterOutcome) -> None:
    hf = prefs.hard_filters
    title = job.title or ""
    for pattern in hf.excluded_title_patterns:
        m = _rx(pattern).search(title)
        if m:
            out.reject("seniority_or_type", f"Title contains “{m.group(0)}”")
            break
    unrelated = next((m for p in hf.unrelated_profession_patterns if (m := _rx(p).search(title))), None)
    if unrelated and not any(_rx(p).search(title) for p in hf.relevant_title_patterns):
        out.reject("profession", f"Unrelated profession (“{unrelated.group(0)}”)")


def experience_check(job: JobLike, prefs: Preferences, out: FilterOutcome) -> None:
    years = job.years_experience_required
    limit = prefs.hard_filters.max_years_experience_required
    if years is not None and limit and years > limit:
        out.reject("experience", f"Requires {years:g}+ years (limit {limit})")


def qualification_check(job: JobLike, prefs: Preferences, out: FilterOutcome) -> None:
    text = "\n".join(x for x in (job.title, job.description, job.requirements) if x)
    for pattern in prefs.hard_filters.mandatory_qualification_patterns:
        m = _rx(pattern).search(text)
        if m:
            out.reject("qualification", f"Requires “{m.group(0)}”")
            return


_GENERIC_SUFFIXES = {"ai", "io", "tech", "technologies", "hq", "inc", "global", "platform", "app", "labs ai"}
_URL_HOST = re.compile(r"https?://([^/\s\"'<>)]+)", re.IGNORECASE)
_BARE_DOMAIN = re.compile(r"\b((?:[a-z0-9-]+\.)+[a-z]{2,})\b", re.IGNORECASE)


def _job_hosts(job: JobLike) -> set[str]:
    """Hosts of the job's own URLs plus every domain-like token in its text (links to an application portal)."""
    urls = [getattr(job, "canonical_url", None), getattr(job, "source_url", None), getattr(job, "career_page_url", None)]
    try:
        urls += [src.url for src in (getattr(job, "sources", None) or [])]
    except Exception:  # noqa: BLE001 - detached ORM object: own fields are enough
        pass
    hosts = {m.group(1).lower().split(":")[0] for u in urls if u for m in [_URL_HOST.search(u)] if m}
    text = "\n".join(x for x in (job.description, job.requirements, job.responsibilities) if x)
    hosts |= {m.group(1).lower() for m in _BARE_DOMAIN.finditer(text)}
    return hosts


def untrusted_source_check(job: JobLike, prefs: Preferences, out: FilterOutcome) -> None:
    """Gig / AI-training marketplaces and talent funnels: skipped, whatever role they advertise."""
    u = prefs.untrusted_sources
    company = job.normalized_company or normalize_company(job.company)
    for name in u.companies:
        n = normalize_company(name)
        # exact name, or the name plus a generic suffix ("micro1 ai", "mercor tech") — never "Outlier Ventures"
        if n and (company == n or (company.startswith(n + " ") and company[len(n) + 1:] in _GENERIC_SUFFIXES)):
            out.reject("untrusted_source", f"{job.company} is on the untrusted-sources list (gig / AI-training marketplace)")
            return
    hosts = _job_hosts(job)
    for domain in u.domains:
        d = domain.lower().lstrip(".")
        if any(h == d or h.endswith("." + d) for h in hosts):
            out.reject("untrusted_source", f"Posting links to {d} (untrusted-sources list)")
            return
    title = job.title or ""
    for pattern in u.title_patterns:
        m = _rx(pattern).search(title)
        if m:
            out.reject("untrusted_source", f"AI-training gig title (“{m.group(0)}”)")
            return
    text = "\n".join(x for x in (job.title, job.description, job.requirements, job.responsibilities) if x)
    hits = [m.group(0) for p in u.posting_patterns if (m := _rx(p).search(text))]
    if hits and (len(hits) >= 2 or _rx(u.gig_signals).search(text)):
        out.reject("untrusted_source", f"AI-training gig posting (“{hits[0]}”)")


def is_doctoral_posting(job: JobLike, prefs: Preferences) -> bool:
    """Doctoral / research-staff title, or any posting from an academic source (those list research positions only)."""
    if getattr(job, "source", None) == "academic":
        return True
    return bool(prefs.phd.posting_pattern) and bool(_rx(prefs.phd.posting_pattern).search(job.title or ""))


def phd_check(job: JobLike, prefs: Preferences, out: FilterOutcome) -> None:
    """Doctoral / research positions: keep only funded ones in your field."""
    phd = prefs.phd
    if not phd.enabled or not is_doctoral_posting(job, prefs):
        return
    text = "\n".join(x for x in (job.title, job.description, job.requirements, job.responsibilities) if x)
    unfunded_spans = [m.span() for p in phd.unfunded_patterns for m in _rx(p).finditer(text)]
    # a funded signal inside an unfunded phrase ("self-funded PhD") does not count
    funded = next((m.group(0) for p in phd.funded_patterns for m in _rx(p).finditer(text)
                   if not any(a < m.end() and m.start() < b for a, b in unfunded_spans)), None)
    if not funded:
        unfunded = text[unfunded_spans[0][0]:unfunded_spans[0][1]] if unfunded_spans else None
        if unfunded:
            out.reject("phd_funding", f"Doctoral position without funding (“{unfunded}”)")
            return
        if phd.no_funding_info == "reject":
            out.reject("phd_funding", "Doctoral position: no salary, contract or stipend mentioned")
            return
        if phd.no_funding_info == "flag":
            out.flag("phd_funding", "Doctoral position: funding not stated — check salary/stipend before applying")
    if phd.field_title_pattern and _rx(phd.field_title_pattern).search(job.title or ""):
        return
    lowered = text.lower()
    hits = {k for k in phd.field_keywords if k and k.lower() in lowered}
    if len(hits) < phd.min_field_keyword_hits:
        out.reject("phd_field", "Doctoral position outside your fields (AI, ML, robotics, software, security…)")


# Gulf roles reserved for nationals (Saudization / Emiratisation programmes): not open to you as a German citizen.
_NATIONALS_TITLE = re.compile(
    r"\b(saudi|emirati|uae|qatari|omani|kuwaiti|bahraini)s?\s+(nationals?|citizens?)\b|\((saudi|emirati|uae)s?\s*(only|nationals?)\)|"
    r"\b(saudi|emirati|uae)s?[\s-]+only\b|\btamheer\b|\bnationals?\s+only\b", re.IGNORECASE)
_NATIONALS_TEXT = re.compile(
    r"(only|exclusively)\s+(open\s+to|for|available\s+to)\s+(saudi|emirati|uae|qatari|omani|kuwaiti|bahraini)s?\s+"
    r"(nationals?|citizens?)|(saudi|emirati|uae)s?\s+nationals?\s+only|must\s+be\s+an?\s+(saudi|emirati|uae)\s+national|"
    r"\bsaudi(zation)?\s+program(me)?\b|\bemiratisation\b|\bemiratization\b", re.IGNORECASE)


_NATIONALITY = {"saudi": "Saudi Arabia", "tamheer": "Saudi Arabia", "emirati": "UAE", "uae": "UAE", "qatari": "Qatar",
                "omani": "Oman", "kuwaiti": "Kuwait", "bahraini": "Bahrain"}


def nationals_only_check(job: JobLike, prefs: Preferences, out: FilterOutcome) -> None:
    """Gulf postings reserved for nationals ("(Saudi Only)", "Emirati National", Tamheer programme)."""
    m = _NATIONALS_TITLE.search(job.title or "")
    if not m:
        text = "\n".join(x for x in (job.description, job.requirements) if x)
        m = _NATIONALS_TEXT.search(text)
    if not m:
        return
    from app.config_files import get_profile
    from app.core.normalize import canonical_country

    hit = m.group(0).strip()
    words = hit.lower()
    country = next((c for key, c in _NATIONALITY.items() if key in words), None) or canonical_country(job.country)
    mine = {canonical_country(c) or c for c in get_profile().citizenships}
    if country and country in mine:
        return  # reserved for nationals — and you are one
    out.reject("nationals_only", f"Reserved for {country or 'local'} nationals (“{hit}”) — your profile lists no such citizenship")


# Free movement: an EU / EEA citizen can work anywhere in the EU, the EEA and Switzerland without a permit.
_EU_EEA = {"Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Czechia", "Denmark", "Estonia",
           "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
           "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden",
           "Iceland", "Liechtenstein", "Norway", "Switzerland"}
_NO_SPONSORSHIP = re.compile(
    r"\b(no|not offering|unable to (offer|provide)|not able to (offer|provide)|can ?not (offer|provide)|can't (offer|provide)|"
    r"do(es)? not (offer|provide)|don't (offer|provide)|will not (offer|provide)|won't (offer|provide))\s+(visa\s+|work permit\s+)?sponsorship|"
    r"\b(unable|not able|can ?not|can't|do not|does not|don't|will not|won't|are not able)\s+(to\s+)?sponsor\b|"
    r"\bsponsorship (is |will )?not (be )?(available|offered|provided|possible)|"
    r"\bwithout (the need for |requiring |needing )?(visa |work permit )?sponsorship|"
    r"\bmust (already )?(have|hold|possess)( the| a| an| full| existing| valid| current)* (right to work|work (permit|authori[sz]ation)|"
    r"(uk |british )?work visa)\b|"
    r"\b(must|need to) be (legally )?(eligible|entitled|authori[sz]ed) to work in the (uk|united kingdom)( without)?",
    re.IGNORECASE)


def work_permit_check(job: JobLike, prefs: Preferences, out: FilterOutcome) -> None:
    """Outside your free-movement area (citizenship(s) in profile.yaml; EU citizens: EU/EEA/CH) a posting that rules
    out visa sponsorship is closed to you — e.g. UK roles "must already have the right to work in the UK"."""
    from app.config_files import get_profile
    from app.core.normalize import canonical_country, infer_country

    mine = {canonical_country(c) or c for c in get_profile().citizenships}
    if not mine:
        return  # unknown citizenship: can't judge
    free = mine | (_EU_EEA if mine & _EU_EEA else set())
    country = canonical_country(job.country) or infer_country(job.location)
    if not country or country in free:
        return
    text = "\n".join(x for x in (job.description, job.requirements) if x)
    m = _NO_SPONSORSHIP.search(text)
    if m:
        out.reject("work_permit", f"No visa sponsorship in {country} (“{m.group(0).strip()}”) — you would need a work permit there")


def apply_hard_filters(job: JobLike, prefs: Preferences) -> FilterOutcome:
    out = FilterOutcome()
    for check in (untrusted_source_check, defence_check, location_check, title_check, experience_check,
                  qualification_check, phd_check, nationals_only_check, work_permit_check):
        check(job, prefs, out)
    return out


def reasons_text(reasons: list[dict[str, Any]], severity: str | None = None) -> str:
    return "; ".join(r["detail"] for r in reasons if severity is None or r.get("severity") == severity)
