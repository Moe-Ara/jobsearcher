from datetime import datetime

import pytest
from pydantic import SecretStr

from app.settings import get_settings
from app.sources.adzuna import AdzunaAdapter
from app.sources.arbeitnow import ArbeitnowAdapter
from app.sources.arbeitsagentur import ArbeitsagenturAdapter
from app.sources.base import CompanyFeedConfig
from app.sources.reed import ReedAdapter
from app.sources.registry import ADAPTERS
from app.sources.rss import RssFeedAdapter


class Http:
    def __init__(self, json=None, text=None, fail=None):
        self.json, self.text, self.fail, self.calls = json, text or {}, fail, []

    def get_json(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        if self.fail:
            raise self.fail
        return self.json(url, params) if callable(self.json) else self.json

    def get_json_with_headers(self, url, params=None, headers=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        return self.json(url, params) if callable(self.json) else self.json

    def get_text(self, url, params=None):
        self.calls.append((url, None))
        return self.text[url]


def cfg(**options):
    return CompanyFeedConfig(company_name="X", options=options)


def test_all_new_platforms_are_registered_as_search_sources():
    for name in ("arbeitnow", "rss", "reed", "adzuna"):
        assert name in ADAPTERS and ADAPTERS[name].partial_listing


def test_arbeitnow_pages_until_there_is_no_next_page():
    item = {"slug": "ml-engineer-berlin-1", "company_name": "Acme", "title": "ML Engineer", "description": "<p>Python</p>",
            "remote": True, "url": "https://www.arbeitnow.com/jobs/acme/ml", "job_types": ["Full-time"],
            "location": "Berlin", "created_at": 1792945207}
    dutch = {**item, "slug": "nl-1", "location": "Breda", "remote": False}
    http = Http(json=lambda url, p: {"data": [item, dutch], "links": {"next": None}} if "arbeitnow" in url
                else [{"address": {"country_code": "nl"}}])
    a = ArbeitnowAdapter(cfg(max_pages=3), http)
    import app.services.drive_time as dt

    dt.time.sleep, sleep = (lambda s: None), dt.time.sleep
    try:
        raws = a.fetch_jobs()
    finally:
        dt.time.sleep = sleep
    assert len(raws) == 2 and raws[1]["_country"] == "Netherlands"  # unknown town looked up → dropped by the country gate
    assert a.location_hint(raws[1]) == "Breda, Netherlands"
    p = a.normalize_job(item)
    assert (p.title, p.company, p.location, p.remote_status, p.description) == ("ML Engineer", "Acme", "Berlin", "remote", "Python")
    assert isinstance(p.date_posted, datetime)


FEED = """<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Senior Backend Engineer &amp;amp; Infra // Ferto AI</title><link>https://berlinstartupjobs.com/engineering/x/</link>
<pubDate>Thu, 22 Oct 2026 15:57:47 +0000</pubDate><category>Python</category><description>Teaser</description></item>
</channel></rss>"""
PAGE = """<html><script type="application/ld+json">{"@type": "JobPosting", "title": "Senior Backend Engineer &amp; Infra // Ferto AI",
"hiringOrganization": {"name": "Ferto AI"}, "description": "Build the lab platform in Python.",
"jobLocation": {"address": {"addressLocality": "Berlin", "addressCountry": "DE"}}}</script></html>"""


def test_rss_feed_splits_title_and_company_and_reads_the_full_posting():
    http = Http(text={"https://feed": FEED, "https://berlinstartupjobs.com/engineering/x/": PAGE})
    a = RssFeedAdapter(CompanyFeedConfig(company_name="BSJ", careers_url="https://feed",
                                         options={"title_separator": " // ", "fetch_page": True, "source_name": "bsj"}), http)
    raws = a.fetch_jobs()
    assert raws[0]["title"].endswith("// Ferto AI") and raws[0]["categories"] == ["Python"]
    p = a.normalize_job(a.fetch_details(raws[0]))
    assert (p.source, p.title, p.company, p.location) == ("bsj", "Senior Backend Engineer & Infra", "Ferto AI", "Berlin")
    assert "lab platform" in p.description and p.date_posted == datetime(2026, 10, 22, 15, 57, 47)


def test_reed_needs_a_key_and_uses_basic_auth(monkeypatch):
    monkeypatch.setattr(get_settings(), "reed_api_key", None)
    assert ReedAdapter(cfg(), Http(json={})).fetch_jobs() == []
    monkeypatch.setattr(get_settings(), "reed_api_key", SecretStr("rk"))
    job = {"jobId": 5123456, "jobTitle": "Data Engineer", "employerName": "Acme Ltd", "locationName": "Manchester",
           "minimumSalary": 50000.0, "maximumSalary": 60000.0, "date": "20/10/2026", "jobDescription": "Spark &amp; Python"}
    http = Http(json=lambda url, p: {"results": [job]} if url.endswith("/search") else {**job, "jobDescription": "<p>Full text</p>"})
    a = ReedAdapter(cfg(queries=["data engineer"]), http)
    raws = a.fetch_jobs()
    assert http.calls[0][2]["Authorization"] == "Basic cms6"  # base64("rk:")
    p = a.normalize_job(a.fetch_details(raws[0]))
    assert (p.company, p.location, p.country, p.description, p.salary_min) == (
        "Acme Ltd", "Manchester, United Kingdom", "United Kingdom", "Full text", 50000.0)
    assert p.date_posted == datetime(2026, 10, 20)


def test_adzuna_needs_keys_and_never_leaks_them(monkeypatch):
    monkeypatch.setattr(get_settings(), "adzuna_app_id", None)
    assert AdzunaAdapter(cfg(), Http(json={})).fetch_jobs() == []
    monkeypatch.setattr(get_settings(), "adzuna_app_id", "id1")
    monkeypatch.setattr(get_settings(), "adzuna_app_key", SecretStr("sekret"))
    with pytest.raises(RuntimeError) as err:
        AdzunaAdapter(cfg(countries=["gb"], queries=["x"]), Http(fail=RuntimeError("HTTP 401 for ...app_key=sekret"))).fetch_jobs()
    assert "sekret" not in str(err.value)
    item = {"id": "42", "title": "<strong>ML</strong> Engineer", "company": {"display_name": "Acme"},
            "location": {"display_name": "London, UK"}, "description": "Torch", "created": "2026-10-20T08:00:00Z",
            "redirect_url": "https://www.adzuna.co.uk/jobs/land/ad/42"}
    http = Http(json={"results": [item]})
    a = AdzunaAdapter(cfg(countries=["gb"], queries=["ml"]), http)
    p = a.normalize_job(a.fetch_jobs()[0])
    assert (p.title, p.company, p.location, p.country) == ("ML Engineer", "Acme", "London, UK", "United Kingdom")
    assert http.calls[0][0].endswith("/gb/search/1") and http.calls[0][1]["max_days_old"] == 7


def test_bundesagentur_searches_germany_wide_for_your_search_terms_without_temp_agencies():
    http = Http(json=lambda url, p: {"ergebnisliste": []})
    a = ArbeitsagenturAdapter(cfg(queries_from_preferences=True), http)
    a.fetch_jobs()
    from app.config_files import get_prefs

    sent = [p for _, p, _ in http.calls]
    assert [p["was"] for p in sent] == get_prefs().search.queries
    assert all(p["zeitarbeit"] == "false" and p["veroeffentlichtseit"] == 7 and "wo" not in p for p in sent)


def test_board_alert_parsers_read_job_links():
    from app.email_ingest.parsers.boards import (CwjobsAlertParser, JobsAcUkAlertParser, ReedAlertParser, XingAlertParser,
                                                 all_parsers, pick_parser)

    html = """<html><body>
      <a href="https://www.xing.com/jobs/muenchen-machine-learning-engineer-123456789?ref=mail">Machine Learning Engineer</a>
      <p>Acme GmbH</p><p>München</p>
      <a href="https://www.xing.com/jobs/berlin-data-scientist-987654321">Data Scientist</a><p>Beta AG</p><p>Berlin</p>
    </body></html>"""
    jobs = XingAlertParser().parse(html)
    assert [(j.external_job_id, j.title) for j in jobs] == [("123456789", "Machine Learning Engineer"), ("987654321", "Data Scientist")]
    assert CwjobsAlertParser().job_id("https://www.cwjobs.co.uk/job/python-developer/acme-job104567890?src=x") == "104567890"
    assert ReedAlertParser().job_id("https://www.reed.co.uk/jobs/data-engineer/51234567") == "51234567"
    assert JobsAcUkAlertParser().job_id("https://www.jobs.ac.uk/job/DSK809/phd-studentship") == "DSK809"
    parsers = all_parsers(["xing", "reed", "generic"], [])
    assert pick_parser(parsers, "Jobs <noreply@xing.com>", "Neue Jobs").name == "xing"
    assert pick_parser(parsers, "Reed <alerts@reed.co.uk>", "New jobs").name == "reed"


def test_job_sites_page_lists_every_country():
    from fastapi.testclient import TestClient

    from app.main import app

    page = TestClient(app).get("/platforms")
    assert page.status_code == 200
    for name in ("Germany", "United Kingdom", "UAE", "Saudi Arabia", "StepStone", "Reed", "Bayt", "Bundesagentur"):
        assert name in page.text
