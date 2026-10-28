from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.config_files import get_prefs
from app.db import session_scope
from app.enums import FilterStatus, SourceType
from app.main import app
from app.models import Job
from app.services.ingest import ingest_posting
from app.services.queries import JobFilters, search_jobs
from app.services.side_jobs import side_filters, side_score
from app.sources.arbeitsagentur import ArbeitsagenturAdapter
from app.sources.base import CompanyFeedConfig, JobPosting

BA_ITEM = {
    "referenznummer": "17907-44522961-1021-S", "stellenangebotsTitel": "IT-Support Mitarbeiter (m/w/d) Teilzeit",
    "hauptberuf": "IT-Systemadministrator/in", "firma": "Bodensee IT GmbH", "entfernung": 3,
    "stellenlokationen": [{"adresse": {"plz": "88046", "ort": "Friedrichshafen", "land": "DEUTSCHLAND"}}],
    "arbeitszeitTeilzeitVormittag": True, "arbeitszeitTeilzeitNachmittag": True, "istGeringfuegigeBeschaeftigung": False,
    "quereinstiegGeeignet": True, "verguetungsangabe": "TARIF", "datumErsteVeroeffentlichung": "2026-10-20",
    "eintrittszeitraum": {"von": "2026-11-01"},
    "stellenangebotsBeschreibung": "Wir suchen Unterstützung im IT-Support, 20 Stunden pro Woche.",
}


def side_job(**kw):
    base = dict(title="Aushilfe", company="X", description="", requirements="", responsibilities="", sector=None, country="Germany", remote_status=None,
                raw_source_data={}, normalized_company="x", source_url="https://www.arbeitsagentur.de/x", location="")
    base.update(kw)
    return SimpleNamespace(**base)


def test_side_filters_hide_student_jobs_and_licence_work_but_keep_any_other_job():
    prefs = get_prefs()
    assert side_filters(side_job(title="Studentenjob Verkauf (m/w/d)"), prefs).rejected
    assert side_filters(side_job(title="Werkstudent IT"), prefs).rejected
    assert side_filters(side_job(title="Fahrer (m/w/d)", description="Führerschein Klasse CE erforderlich"), prefs).rejected
    assert not side_filters(side_job(title="Aushilfe Kasse (m/w/d) Minijob"), prefs).rejected
    assert not side_filters(side_job(title="Nachhilfelehrer Mathe"), prefs).rejected
    nationwide = side_job(title="Buchhalter", raw_source_data={"nationwide_not_remote": True})
    assert side_filters(nationwide, prefs).rejected


def test_side_score_prefers_near_fitting_flexible_jobs():
    prefs = get_prefs()
    near_it, parts = side_score(side_job(title="IT-Support Aushilfe", raw_source_data={
        "distance_km": 3, "career_changers": True, "pay_stated": True, "flexible_slots": True}), prefs)
    far_other, _ = side_score(side_job(title="Reinigungskraft", raw_source_data={"distance_km": 24, "temp_agency": True}), prefs)
    assert near_it > far_other and near_it >= 9 and "it support" in parts["fit_words"]


def test_arbeitsagentur_adapter_reads_searches_and_details():
    class Http:
        calls = []

        def get_json_with_headers(self, url, params=None, headers=None):
            self.calls.append((url, params, headers))
            if "/jobdetails/" in url:
                return {"stellenangebotsBeschreibung": BA_ITEM["stellenangebotsBeschreibung"], "istArbeitnehmerUeberlassung": False}
            return {"ergebnisliste": [{k: v for k, v in BA_ITEM.items() if k != "stellenangebotsBeschreibung"}]}

    http = Http()
    cfg = CompanyFeedConfig(company_name="BA", options={"track": "side", "searches": [
        {"label": "near", "wo": "88045 Friedrichshafen", "umkreis": 25, "arbeitszeit": "tz;mj"}]})
    adapter = ArbeitsagenturAdapter(cfg, http)
    raws = adapter.fetch_jobs()
    assert http.calls[0][2] == {"X-API-Key": "jobboerse-jobsuche"} and http.calls[0][1]["veroeffentlichtseit"] == 14
    p = adapter.normalize_job(adapter.fetch_details(raws[0]))
    assert p.company == "Bodensee IT GmbH" and p.location == "Friedrichshafen, Germany" and "Teilzeit" in p.employment_type
    assert p.raw["distance_km"] == 3 and p.raw["career_changers"] and p.raw["pay_stated"] and "IT-Support" in p.description
    assert p.url == "https://www.arbeitsagentur.de/jobsuche/jobdetail/17907-44522961-1021-S"


def test_side_jobs_stay_out_of_the_career_search_and_show_on_their_own_page():
    with session_scope() as s:
        side = ingest_posting(s, JobPosting(source="arbeitsagentur", source_type=SourceType.JOB_BOARD, title="Aushilfe IT-Support",
                                            company="Bodensee IT GmbH", location="Friedrichshafen", country="Germany",
                                            employment_type="Teilzeit", track="side", external_job_id="side-1",
                                            raw={"distance_km": 4, "drive_minutes": 9}, url="https://www.arbeitsagentur.de/jobsuche/jobdetail/side-1")).job
        side_id = side.id
        assert side.track == "side" and side.analysis_status == "SKIPPED" and side.fit_score >= 7
        main = ingest_posting(s, JobPosting(source="manual", source_type=SourceType.MANUAL, title="ML Engineer",
                                            company="Acme AI GmbH", location="Ulm", external_job_id="main-1")).job
        main_id = main.id
    with session_scope() as s:
        ids = {j.id for j in search_jobs(s, JobFilters(unreviewed=False))[0]}
        assert main_id in ids and side_id not in ids
    page = TestClient(app).get("/side-jobs").text
    assert "Aushilfe IT-Support" in page and "ML Engineer" not in page and "Side jobs" in page
    assert "Aushilfe IT-Support" in TestClient(app).get("/side-jobs?kind=teilzeit&max_min=15").text
    with session_scope() as s:
        assert s.get(Job, side_id).filter_status != FilterStatus.REJECTED


def test_regulated_professions_are_hidden_and_trade_jobs_rank_lower():
    prefs = get_prefs()
    assert side_filters(side_job(title="Rechtsanwalt (m/w/d) für IT-Recht"), prefs).rejected
    assert side_filters(side_job(title="Zahntechniker (m/w/d)"), prefs).rejected
    assert not side_filters(side_job(title="Büroaushilfe Digitalisierung"), prefs).rejected
    plain, _ = side_score(side_job(title="Aushilfe Lager", raw_source_data={"distance_km": 5}), prefs)
    trade, parts = side_score(side_job(title="Aushilfe Lager", description="Sie haben eine abgeschlossene Ausbildung als Fachkraft",
                                       raw_source_data={"distance_km": 5}), prefs)
    assert trade < plain and "needs_vocational_training" in parts["parts"]


def test_drive_times_are_routed_in_a_batch_cached_and_estimated_when_offline(tmp_path, monkeypatch):
    from app.services import drive_time

    monkeypatch.setattr(drive_time, "_cache_path", lambda name="drive_times.json": tmp_path / name)
    calls = []

    class Http:
        def get_json(self, url, params=None):
            calls.append(url)
            n = url.count(";")
            return {"durations": [[0] + [600 * i for i in range(1, n + 1)]]}

    home = (47.65, 9.48)
    places = [(47.7812, 9.6116), (47.5460, 9.7272)]
    first = drive_time.drive_minutes(home, places, Http())
    assert sorted(first.values()) == [10.0, 20.0] and len(calls) == 1
    again = drive_time.drive_minutes(home, places, Http())
    assert again == first and len(calls) == 1  # cached: no second request

    class Down:
        def get_json(self, url, params=None):
            raise OSError("router down")

    est = drive_time.drive_minutes(home, [(47.6779, 9.1753)], Down())  # Konstanz: ~21 km straight line
    assert 25 <= list(est.values())[0] <= 40


def test_side_jobs_more_than_an_hour_away_are_hidden_unless_remote():
    prefs = get_prefs()
    far = side_job(title="Aushilfe Lager", raw_source_data={"drive_minutes": 75})
    assert side_filters(far, prefs).rejected
    assert not side_filters(side_job(title="Aushilfe Lager", raw_source_data={"drive_minutes": 55}), prefs).rejected
    assert not side_filters(side_job(title="IT Support", remote_status="remote", raw_source_data={"drive_minutes": 200}), prefs).rejected
    quick, _ = side_score(side_job(title="Aushilfe", raw_source_data={"drive_minutes": 12}), prefs)
    slow, _ = side_score(side_job(title="Aushilfe", raw_source_data={"drive_minutes": 55}), prefs)
    assert quick > slow


def test_us_state_codes_mean_the_united_states():
    from app.core.normalize import infer_country

    assert infer_country("Southfield, MI") == "United States" and infer_country("King of Prussia, PA") == "United States"
    assert infer_country("Berlin, DE") == "Germany" and infer_country("Riyadh, SA") == "Saudi Arabia"
    assert infer_country("King Abdullah Economic City, 02") == "Saudi Arabia"


def test_windowed_and_capped_sources_never_expire_jobs_they_did_not_see():
    from app.sources.academic import AcademicListingAdapter
    from app.sources.workday import WorkdayAdapter

    assert ArbeitsagenturAdapter.partial_listing and AcademicListingAdapter.partial_listing

    class Http:
        def post_json(self, url, payload):
            return {"total": 999, "jobPostings": [{"externalPath": f"/job/{payload['offset'] + i}", "title": "x"} for i in range(20)]}

    a = WorkdayAdapter(CompanyFeedConfig(company_name="X", identifier="acme.wd3/External",
                                         options={"searches": ["*"], "max_jobs": 40}), Http())
    assert len(a.fetch_jobs()) == 40 and a.truncated  # stopped at its cap: the run saw only part of the board


def test_side_jobs_page_shows_what_is_new_and_when_sources_were_checked():
    with session_scope() as s:
        job = ingest_posting(s, JobPosting(source="arbeitsagentur", source_type=SourceType.JOB_BOARD, external_job_id="n1",
                                           url="https://www.arbeitsagentur.de/jobsuche/jobdetail/n1", title="IT-Support Aushilfe",
                                           company="Acme", location="Friedrichshafen", country="Germany",
                                           description="IT-Support in Teilzeit", employment_type="Teilzeit", track="side",
                                           raw={"drive_minutes": 5})).job
        assert job.track == "side"
    page = TestClient(app).get("/side-jobs")
    assert "Sources last checked" in page.text and "new in the last 24 h" in page.text and ">new</span>" in page.text
    only_new = TestClient(app).get("/side-jobs?kind=new")
    assert "IT-Support Aushilfe" in only_new.text
