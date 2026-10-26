"""Every job site that matters in your countries, and how JobSearcher reaches it (page: More → Job sites).

How a site is connected:
  search  JobSearcher searches it itself (official API, feed, or the employers' own job boards)
  key     the same, once you add a free API key to .env
  alerts  the site doesn't allow automated searches — create a job alert there; JobSearcher reads the e-mails
  none    no legitimate way in (blocked, no alerts) — listed so you know it was considered
A site can have several ways in (e.g. Reed: API key and alerts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Platform:
    name: str
    countries: tuple[str, ...]
    url: str
    sources: tuple[str, ...] = ()       # JobSource.source values the site's jobs arrive under
    search: tuple[str, ...] = ()        # ats_type of sources that search it (configured in companies.yaml)
    keys: tuple[str, ...] = ()          # settings that must be set for the search to run
    alerts: str | None = None           # e-mail parser name
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


DE, UK, AE, SA = "Germany", "United Kingdom", "UAE", "Saudi Arabia"
ALL = (DE, UK, AE, SA)

PLATFORMS: list[Platform] = [
    # ---- everywhere
    Platform("Company career pages", ALL, "", search=("greenhouse", "lever", "ashby", "workday", "smartrecruiters",
             "personio", "workable", "recruitee", "generic", "amazon"),
             sources=("greenhouse", "lever", "ashby", "workday", "smartrecruiters", "personio", "workable", "recruitee",
                      "career_page", "amazon"),
             note="The employers' own job boards (100+ companies) — the original postings"),
    Platform("LinkedIn", ALL, "https://www.linkedin.com/jobs/", sources=("linkedin",), alerts="linkedin",
             note="Automated access is not allowed; job alerts work"),
    Platform("Indeed", ALL, "https://www.indeed.com", sources=("indeed",), alerts="indeed",
             note="Blocks automated access; job alerts work (de., uk., ae., sa.indeed.com)"),
    Platform("Glassdoor", (DE, UK), "https://www.glassdoor.com", sources=("glassdoor",), alerts="glassdoor"),
    Platform("Workable job board", ALL, "https://jobs.workable.com", search=("workable_board",), sources=("workable_board",),
             note="Every company hiring through Workable"),
    Platform("Careerjet", ALL, "https://www.careerjet.com", search=("careerjet",), keys=("careerjet_api_key",),
             sources=("careerjet",), note="Aggregator — also indexes boards that can't be searched directly"),
    Platform("Jooble", ALL, "https://jooble.org", search=("jooble",), keys=("jooble_api_key",), sources=("jooble",),
             note="Aggregator — also indexes boards that can't be searched directly"),
    # ---- Germany
    Platform("Bundesagentur für Arbeit (Jobsuche)", (DE,), "https://www.arbeitsagentur.de/jobsuche/",
             search=("arbeitsagentur",), sources=("arbeitsagentur",), alerts="arbeitsagentur",
             note="Germany's largest job database — searched Germany-wide for your search terms (no temp agencies)"),
    Platform("StepStone", (DE,), "https://www.stepstone.de", sources=("stepstone",), alerts="stepstone",
             note="Automated access is not allowed; the Jobagent e-mails work"),
    Platform("XING", (DE,), "https://www.xing.com/jobs", sources=("xing",), alerts="xing"),
    Platform("Arbeitnow", (DE,), "https://www.arbeitnow.com", search=("arbeitnow",), sources=("arbeitnow",),
             note="Free public API"),
    Platform("Berlin Startup Jobs", (DE,), "https://berlinstartupjobs.com", search=("rss",), sources=("berlinstartupjobs",),
             note="RSS feed"),
    Platform("Adzuna", (DE, UK), "https://www.adzuna.de", search=("adzuna",), keys=("adzuna_app_id", "adzuna_app_key"),
             sources=("adzuna",), note="Aggregator, official API"),
    Platform("Monster", (DE, UK), "https://www.monster.de", sources=("monster",), alerts="monster"),
    Platform("stellenanzeigen.de", (DE,), "https://www.stellenanzeigen.de", sources=("stellenanzeigen",), alerts="stellenanzeigen"),
    Platform("Jobware", (DE,), "https://www.jobware.de", sources=("jobware",), alerts="jobware"),
    Platform("get in IT", (DE,), "https://www.get-in-it.de", sources=("get_in_it",), alerts="get_in_it", note="IT graduates"),
    Platform("academics.de / EURAXESS / ELLIS / universities", (DE,), "https://www.academics.de", search=("academic",),
             sources=("academic",), note="Funded PhD and research positions"),
    Platform("Kimeta, Talent.com, Joblift, Jobrapido", (DE, UK), "", note="Aggregators that mirror the boards above — not needed"),
    # ---- United Kingdom
    Platform("Reed", (UK,), "https://www.reed.co.uk", search=("reed",), keys=("reed_api_key",), sources=("reed",),
             alerts="reed", note="Official API (free key) — or its job alerts"),
    Platform("Totaljobs", (UK,), "https://www.totaljobs.com", sources=("totaljobs",), alerts="totaljobs", note="StepStone group"),
    Platform("CWJobs", (UK,), "https://www.cwjobs.co.uk", sources=("cwjobs",), alerts="cwjobs", note="IT jobs, StepStone group"),
    Platform("Jobsite", (UK,), "https://www.jobsite.co.uk", sources=("jobsite",), alerts="jobsite"),
    Platform("CV-Library", (UK,), "https://www.cv-library.co.uk", sources=("cvlibrary",), alerts="cvlibrary"),
    Platform("jobs.ac.uk", (UK,), "https://www.jobs.ac.uk", search=("academic",), sources=("academic", "jobs_ac_uk"),
             alerts="jobs_ac_uk", note="UK academic jobs and PhD studentships"),
    Platform("Guardian Jobs", (UK,), "https://jobs.theguardian.com", sources=("guardian_jobs",), alerts="guardian_jobs"),
    Platform("Welcome to the Jungle (Otta)", (UK, DE), "https://www.welcometothejungle.com", sources=("wttj",), alerts="wttj"),
    Platform("Find a Job (DWP)", (UK,), "https://findajob.dwp.gov.uk", note="No working feed or API; mostly local roles"),
    # ---- UAE & Saudi Arabia
    Platform("Bayt", (AE, SA), "https://www.bayt.com", sources=("bayt",), alerts="bayt",
             note="Blocks automated access (robots.txt too); job alerts work"),
    Platform("GulfTalent", (AE, SA), "https://www.gulftalent.com", sources=("gulftalent",), alerts="gulftalent"),
    Platform("NaukriGulf", (AE, SA), "https://www.naukrigulf.com", sources=("naukrigulf",), alerts="naukrigulf"),
    Platform("Gulf startup boards (Hub71, BECO, MEVP, Global Ventures)", (AE, SA), "https://jobs.hub71.com",
             search=("getro",), sources=("getro",)),
    Platform("Michael Page Middle East", (AE, SA), "https://www.michaelpage.ae", search=("generic",), sources=("career_page",),
             note="Recruitment agency (technology roles)"),
    Platform("foundit (Monster Gulf), Dubizzle, Laimoon", (AE, SA), "",
             note="Block automated access and have no usable alerts — their jobs reach you via Careerjet / Jooble"),
    Platform("Jadarat, Nafis, Dubai Careers", (AE, SA), "", note="Government platforms, mainly for nationals"),
]


def overview() -> dict[str, list[dict[str, Any]]]:
    """Per country: every platform with how it is connected and how many jobs it delivered (active jobs)."""
    from sqlalchemy import func, select

    from app.config_files import get_prefs
    from app.db import session_scope
    from app.models import Job, JobSource, TargetCompany
    from app.settings import get_settings

    settings = get_settings()
    enabled = set(get_prefs().email.enabled_parsers)
    with session_scope() as s:
        counts: dict[tuple[str, str], int] = {
            (src, country or ""): n for src, country, n in s.execute(
                select(JobSource.source, Job.country, func.count(func.distinct(JobSource.job_id)))
                .join(Job, Job.id == JobSource.job_id).group_by(JobSource.source, Job.country)).all()}
        active_types = {t for (t,) in s.execute(select(TargetCompany.ats_type).where(TargetCompany.active.is_(True))).all()}
    countries = list(dict.fromkeys([*get_prefs().locations.target_countries, *ALL]))
    out: dict[str, list[dict[str, Any]]] = {c: [] for c in countries if c in ALL}
    for p in PLATFORMS:
        searching = bool(set(p.search) & active_types) and all(getattr(settings, k, None) for k in p.keys)
        reading = bool(p.alerts) and p.alerts in enabled and settings.imap_configured
        per_country = {c: sum(counts.get((src, c), 0) for src in p.sources) for c in p.countries}
        jobs = sum(per_country.values())
        if jobs and (searching or reading):
            status, label = "ok", "jobs arriving"
        elif searching:
            status, label = "ok", "searched automatically — no matching jobs yet"
        elif p.keys and set(p.search) & active_types:
            status, label = "key", "add " + " + ".join(k.upper() for k in p.keys) + " to .env (free)"
        elif reading:
            status, label = "alert", "create a job alert on the site (to your connected mailbox)"
        elif p.alerts:
            status, label = "alert", "connect your mailbox (Setup), then create a job alert on the site"
        else:
            status, label = "none", "not connected"
        how = [w for w, on in (("search", bool(p.search)), ("API key", bool(p.keys)), ("job alerts", bool(p.alerts))) if on]
        for c in p.countries:
            if c in out:
                out[c].append({"platform": p, "status": status, "label": label, "how": how or ["—"], "jobs": per_country[c]})
    return out
