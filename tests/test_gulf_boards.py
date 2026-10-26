from datetime import datetime, timedelta, timezone

import pytest
from pydantic import SecretStr

from app.settings import get_settings
from app.sources.base import CompanyFeedConfig
from app.sources.careerjet import CareerjetAdapter
from app.sources.jooble import JoobleAdapter
from app.sources.registry import ADAPTERS
from app.sources.workable_board import WorkableBoardAdapter


def _iso(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")


def wk_job(i, title="Senior Backend Engineer", days=3, **kw):
    return {"id": f"id-{i}", "title": title, "url": f"https://jobs.workable.com/view/{i}/x", "created": _iso(days),
            "description": "<p>Build payment APIs in Python.</p>", "requirementsSection": "<ul><li>3+ years</li></ul>",
            "employmentType": "Full-time", "department": "Engineering", "workplace": "on_site",
            "locations": ["Riyadh, Riyadh Province, Saudi Arabia"],
            "location": {"city": "Riyadh", "subregion": "Riyadh Province", "countryName": "Saudi Arabia"},
            "company": {"title": "Salla", "url": "https://jobs.workable.com/company/abc/jobs-at-salla"}, **kw}


class FakeHttp:
    def __init__(self, pages=None, response=None, fail=None):
        self.pages, self.response, self.fail, self.calls = pages or [], response, fail, []

    def get_json(self, url, params=None):
        self.calls.append(("get", url, dict(params or {})))
        token = (params or {}).get("pageToken")
        return self.pages[int(token) if token else 0]

    def get_json_with_headers(self, url, params=None, headers=None):
        self.calls.append(("get", url, dict(params or {}), dict(headers or {})))
        return self.response

    def post_json(self, url, payload):
        self.calls.append(("post", url, payload))
        if self.fail:
            raise self.fail
        return self.response


def test_new_gulf_platforms_are_registered_and_search_sources_never_expire_missing_postings():
    for name in ("workable_board", "careerjet", "jooble"):
        assert name in ADAPTERS and ADAPTERS[name].partial_listing
    assert not ADAPTERS["greenhouse"].partial_listing


def test_workable_board_searches_pages_skips_old_postings_and_dedupes():
    pages = [{"jobs": [wk_job(1), wk_job(2, days=200)], "nextPageToken": "1"},
             {"jobs": [wk_job(1), wk_job(3, title="ML Engineer", workplace="remote", locations=["TELECOMMUTE"])],
              "nextPageToken": None}]
    http = FakeHttp(pages=pages)
    cfg = CompanyFeedConfig(company_name="Workable", options={"queries": ["backend"], "locations": ["Saudi Arabia"]})
    adapter = WorkableBoardAdapter(cfg, http)
    raws = adapter.fetch_jobs()
    assert sorted(r["id"] for r in raws) == ["id-1", "id-3"]  # 200-day-old posting skipped, duplicate merged
    assert http.calls[0][2] == {"query": "backend", "location": "Saudi Arabia"} and http.calls[1][2]["pageToken"] == "1"
    p = adapter.normalize_job(raws[0])
    assert (p.company, p.location, p.country, p.remote_status) == ("Salla", "Riyadh, Saudi Arabia", "Saudi Arabia", "onsite")
    assert "payment APIs" in p.description and "3+ years" in p.requirements
    assert adapter.normalize_job(raws[1]).remote_status == "remote"


def test_careerjet_and_jooble_do_nothing_without_a_key(monkeypatch):
    monkeypatch.setattr(get_settings(), "careerjet_api_key", None)
    monkeypatch.setattr(get_settings(), "jooble_api_key", None)
    http = FakeHttp()
    assert CareerjetAdapter(CompanyFeedConfig(company_name="Careerjet"), http).fetch_jobs() == []
    assert JoobleAdapter(CompanyFeedConfig(company_name="Jooble"), http).fetch_jobs() == []
    assert http.calls == []


def test_careerjet_uses_basic_auth_and_maps_the_locale_to_a_country(monkeypatch):
    monkeypatch.setattr(get_settings(), "careerjet_api_key", SecretStr("k123"))
    job = {"title": "Data Scientist", "company": "Mozn", "locations": "Riyadh", "url": "https://jobviewtrack.com/x1",
           "description": "Build <b>fraud</b> models", "date": "Mon, 19 Oct 2026 08:00:00 GMT", "salary_min": 0}
    http = FakeHttp(response={"type": "JOBS", "hits": 1, "pages": 1, "jobs": [job]})
    adapter = CareerjetAdapter(CompanyFeedConfig(company_name="Careerjet", options={"queries": ["data"], "locales": ["en_SA"]}), http)
    raws = adapter.fetch_jobs()
    _, url, params, headers = http.calls[0]
    assert headers["Authorization"] == "Basic azEyMzo="  # base64("k123:")
    assert params["locale_code"] == "en_SA" and params["keywords"] == "data" and params["user_ip"]
    p = adapter.normalize_job(raws[0])
    assert (p.title, p.company, p.country, p.description) == ("Data Scientist", "Mozn", "Saudi Arabia", "Build fraud models")
    assert p.date_posted == datetime(2026, 10, 19, 8, 0)


def test_jooble_never_leaks_its_key_in_errors(monkeypatch):
    monkeypatch.setattr(get_settings(), "jooble_api_key", SecretStr("secret-key-42"))
    http = FakeHttp(fail=RuntimeError("HTTP 403 for https://jooble.org/api/secret-key-42"))
    with pytest.raises(RuntimeError) as err:
        JoobleAdapter(CompanyFeedConfig(company_name="Jooble", options={"queries": ["ai"]}), http).fetch_jobs()
    assert "secret-key-42" not in str(err.value) and "***" in str(err.value)


def test_a_town_wins_over_a_contradicting_jsonld_country():
    from app.sources.jsonld import jsonld_to_fields

    node = {"@type": "JobPosting", "title": "AI Engineer", "hiringOrganization": {"name": "Michael Page"},
            "jobLocation": {"address": {"addressLocality": "Dubai", "addressRegion": "United Arab Emirates", "addressCountry": "SG"}}}
    assert jsonld_to_fields(node)["country"] == "UAE"
    node["jobLocation"]["address"] = {"addressLocality": "Munich", "addressCountry": "DE"}
    assert jsonld_to_fields(node)["country"] == "DE"  # consistent data is left alone


def test_database_backup_is_a_consistent_copy_and_keeps_the_newest(tmp_path, monkeypatch):
    import sqlite3

    from app.services import backup

    db = tmp_path / "live.db"
    con = sqlite3.connect(db)
    con.execute("create table t (x)")
    con.execute("insert into t values (42)")
    con.commit()
    monkeypatch.setattr(get_settings(), "database_url", f"sqlite:///{db.as_posix()}")
    monkeypatch.setattr(get_settings(), "data_dir", tmp_path / "data")
    names = []
    for i in range(3):
        (tmp_path / "data" / "backups").mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / "backups" / f"jobsearcher-2026010{i}-000000.db").write_bytes(b"old")
    res = backup.backup_database(keep=2)
    names = [p.name for p in backup.list_backups()]
    assert len(names) == 2 and names[0] == res["file"] and res["removed_old"] == 2
    assert sqlite3.connect(tmp_path / "data" / "backups" / res["file"]).execute("select x from t").fetchone() == (42,)
    assert backup.backup_if_due()["skipped"]
    con.close()
