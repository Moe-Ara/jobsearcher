from datetime import date
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.core.http import FetchError
from app.services import liveness


class Http:
    def __init__(self, pages=None, json=None, errors=None):
        self.pages, self.json, self.errors = pages or {}, json or {}, errors or {}

    def _err(self, url):
        for key, code in self.errors.items():
            if key in url:
                raise FetchError(f"HTTP {code} from {url}")

    def get_text(self, url, params=None):
        self._err(url)
        return self.pages.get(url, "<html><body>Software Engineer (m/w/d) — jetzt bewerben</body></html>")

    def get_json(self, url, params=None):
        self._err(url)
        return self.json.get(url, {})

    def get_json_with_headers(self, url, params=None, headers=None):
        return self.get_json(url, params)


def job(**kw):
    base = dict(id=1, active=True, description="", requirements="", responsibilities="", sources=[], target_company=None,
                external_job_id=None, career_page_url=None, source_url=None, canonical_url=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_deadline_in_the_past_closes_the_posting():
    assert liveness.deadline_passed("Bewerbungsfrist: 15.10.2026", today=date(2026, 10, 24)) == date(2026, 10, 15)
    assert liveness.deadline_passed("Bewerbungsschluss ist der 30. November 2026", today=date(2026, 10, 24)) is None
    assert liveness.deadline_passed("Application deadline: October 1, 2026", today=date(2026, 10, 24)) == date(2026, 10, 1)
    assert liveness.check(job(description="Bitte bewerben Sie sich bis zum 01.01.2020."), Http()).closed


def test_page_checks_find_closed_and_open_postings():
    url = "https://careers.acme.example/jobs/1234"
    closed = Http(pages={url: "<p>Diese Stelle ist leider bereits besetzt.</p>"})
    assert liveness.check(job(source_url=url), closed).closed
    gone = Http(errors={"acme.example": 404})
    r = liveness.check(job(source_url=url), gone)
    assert r.closed and "404" in r.reason
    assert liveness.check(job(source_url=url), Http()).status == "open"
    blocked = Http(errors={"acme.example": 403})
    assert liveness.check(job(source_url=url), blocked).status == "unknown"  # cannot tell → the agents run
    assert liveness.check(job(source_url="https://de.indeed.com/viewjob?jk=1"), Http()).status == "unknown"
    assert liveness.check(job(active=False), Http()).closed


def test_ats_jobs_are_checked_through_their_api():
    tc = SimpleNamespace(ats_identifier="acme", ats_config={})
    gh = job(sources=[SimpleNamespace(source="greenhouse")], target_company=tc, external_job_id="77")
    assert liveness.check(gh, Http()).status == "open"
    assert liveness.check(gh, Http(errors={"boards-api.greenhouse.io": 404})).closed
    ash = job(sources=[SimpleNamespace(source="ashby")], target_company=tc, external_job_id="u1")
    board = "https://api.ashbyhq.com/posting-api/job-board/acme"
    assert liveness.check(ash, Http(json={board: {"jobs": [{"id": "u1"}]}})).status == "open"
    assert liveness.check(ash, Http(json={board: {"jobs": [{"id": "other"}]}})).closed


def test_a_closed_posting_stops_the_agents_and_expires_the_job(monkeypatch):
    from app.db import session_scope
    from app.enums import JobStatus, SourceType
    from app.main import app
    from app.models import ApplicationPackage, Job
    from app.services.ingest import ingest_posting
    from app.sources.base import JobPosting

    with session_scope() as s:
        job_id = ingest_posting(s, JobPosting(source="manual", source_type=SourceType.MANUAL, title="Closed Engineer",
                                              company="Gone GmbH", location="Ulm", external_job_id="gone-1")).job.id
    monkeypatch.setattr(liveness, "check", lambda j, http=None: liveness.Liveness("closed", "the posting page says “position has been filled”"))
    r = TestClient(app).post(f"/jobs/{job_id}/apply", data={"language": "de"}, follow_redirects=False)
    assert r.status_code == 303 and "closed" in r.headers["location"]
    with session_scope() as s:
        assert s.get(Job, job_id).status == JobStatus.EXPIRED
        assert s.query(ApplicationPackage).filter_by(job_id=job_id).count() == 0


def test_gulf_roles_reserved_for_nationals_are_rejected():
    from app.config_files import get_prefs
    from app.filters import FilterOutcome, nationals_only_check

    def run(title, text=""):
        out = FilterOutcome()
        nationals_only_check(SimpleNamespace(title=title, description=text, requirements=""), get_prefs(), out)
        return out.rejected

    assert run("DevOps Engineer (Saudi Only)") and run("AML Analyst - Builders Program - (Emirati National)")
    assert run("Associate Product Manager (Tamheer)") and run("Data Analyst", "This role is only open to UAE nationals.")
    assert not run("Backend Engineer", "Our team in Riyadh works with Saudi and international colleagues.")
    assert not run("Machine Learning Engineer")


def test_postings_reserved_for_your_own_nationality_stay_open(monkeypatch):
    from app.config_files import get_profile, get_prefs
    from app.filters import FilterOutcome, nationals_only_check

    def rejected(title):
        out = FilterOutcome()
        nationals_only_check(SimpleNamespace(title=title, description="", requirements="", country="Saudi Arabia"), get_prefs(), out)
        return out.rejected

    assert rejected("DevOps Engineer (Saudi Only)")
    monkeypatch.setattr(get_profile(), "citizenships", ["Saudi Arabia"])
    assert not rejected("DevOps Engineer (Saudi Only)") and rejected("Analyst (Emirati National)")


def test_postings_ruling_out_sponsorship_are_closed_outside_your_free_movement_area():
    from app.config_files import get_prefs
    from app.filters import FilterOutcome, work_permit_check

    def rejected(country, text):
        out = FilterOutcome()
        work_permit_check(SimpleNamespace(title="Engineer", description=text, requirements="", country=country,
                                          location=""), get_prefs(), out)
        return out.rejected

    assert rejected("United Kingdom", "You must already have the right to work in the UK.")
    assert rejected("United Kingdom", "Unfortunately we are unable to offer visa sponsorship for this role.")
    assert rejected("United Kingdom", "Applicants must be eligible to work in the UK without sponsorship.")
    assert not rejected("United Kingdom", "Visa sponsorship is available for the right candidate.")
    assert not rejected("Netherlands", "We cannot sponsor visas.")  # EU citizen: free movement
    assert not rejected("Germany", "No visa sponsorship.")
