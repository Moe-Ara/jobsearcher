from fastapi.testclient import TestClient

from app.db import session_scope
from app.enums import FilterStatus, Recommendation
from app.main import app
from app.models import Job
from app.services import distances, drive_time
from app.services.queries import JobFilters, search_jobs


def _job(s, title, country, rec=Recommendation.PRIORITY, score=8.5, km=None, remote=None, location="Somewhere"):
    j = Job(title=title, company="Acme", normalized_company="acme", normalized_title=title.lower(), source="greenhouse",
            country=country, location=location, remote_status=remote, track="main", active=True,
            filter_status=FilterStatus.PASSED, recommendation=rec, fit_score=score, distance_km=km)
    s.add(j)
    s.flush()
    return j


def test_the_job_list_shows_only_your_default_countries_unless_you_tick_more():
    with session_scope() as s:
        _job(s, "Berlin Engineer", "Germany")
        _job(s, "London Engineer", "United Kingdom")
        _job(s, "Remote Engineer", None, remote="remote")
    c = TestClient(app)
    page = c.get("/?applied=1&unreviewed=0&since_days=")
    assert "Berlin Engineer" in page.text and "London Engineer" not in page.text and "Remote Engineer" not in page.text
    both = c.get("/?applied=1&unreviewed=0&since_days=&country=Germany&country=United+Kingdom")
    assert "Berlin Engineer" in both.text and "London Engineer" in both.text and "Remote Engineer" not in both.text
    everything = c.get("/?applied=1&unreviewed=0&since_days=&country=*")
    assert "Remote Engineer" in everything.text and 'value="unknown"' in page.text


def test_priority_first_then_the_nearest_then_the_score():
    with session_scope() as s:
        _job(s, "Far priority", "Germany", km=300, score=9.5)
        _job(s, "Near priority", "Germany", km=12, score=8.1)
        _job(s, "Unknown distance priority", "Germany", km=None, score=9.9)
        _job(s, "Near review", "Germany", rec=Recommendation.REVIEW, km=2, score=7.0)
        jobs, _ = search_jobs(s, JobFilters(unreviewed=False, since_days=None, countries=["Germany"]))
        assert [j.title for j in jobs] == ["Near priority", "Far priority", "Unknown distance priority", "Near review"]


def test_distances_come_from_the_cache_and_remote_counts_as_zero(monkeypatch):
    assert drive_time.place_query("London, England, United Kingdom", "United Kingdom") == "London, United Kingdom"
    assert drive_time.place_query("Ulm, BW, Germany") == "Ulm, BW, Deutschland"
    cache = {"ulm, deutschland": [48.4011, 9.9876]}
    with session_scope() as s:
        near = _job(s, "Ulm job", "Germany", location="Ulm")
        remote = _job(s, "Remote job", "Germany", remote="remote")
        assert 80 < distances.set_distance(near, (47.65, 9.48), cache) < 110
        assert distances.set_distance(remote, (47.65, 9.48), cache) == 0.0


def test_new_towns_are_looked_up_in_the_background(monkeypatch):
    looked = []

    def fake_geocode(queries, http=None, limit=60, countrycodes="de"):
        cache = drive_time._load("geocode.json")
        for q in queries:
            looked.append((q, countrycodes))
            cache[q.lower()] = [51.5072, -0.1276] if countrycodes == "gb" else [48.137, 11.575]
        drive_time._save(cache, "geocode.json")
        return {q: tuple(cache[q.lower()]) for q in queries}

    monkeypatch.setattr(drive_time, "geocode", fake_geocode)
    with session_scope() as s:
        _job(s, "Munich job", "Germany", location="München")
        _job(s, "London job", "United Kingdom", location="London")
    out = distances.resolve_job_distances()
    assert out["updated"] == 2 and ("London, United Kingdom", "gb") in looked
    with session_scope() as s:
        km = {j.title: j.distance_km for j in s.query(Job).all()}
        assert km["Munich job"] < 250 < km["London job"]


def test_a_region_with_a_minimum_score_hides_weaker_jobs_there(monkeypatch, prefs):
    from app.filters import in_region
    from app.services import jobs as jobs_service
    from app.scoring import ScoreResult

    assert in_region("39104 Magdeburg", "Sachsen-Anhalt") and in_region("Halle (Saale)", "Sachsen-Anhalt")
    assert in_region("06844 Dessau-Roßlau", "Sachsen-Anhalt") and in_region("38855 Wernigerode", "Sachsen-Anhalt")
    assert not in_region("Halle (Westf.)", "Sachsen-Anhalt") and not in_region("04109 Leipzig", "Sachsen-Anhalt")
    assert not in_region("38100 Braunschweig", "Sachsen-Anhalt") and not in_region("06567 Bad Frankenhausen", "Sachsen-Anhalt")
    monkeypatch.setattr(prefs.locations, "min_score_in_regions", {"Sachsen-Anhalt": 9})
    monkeypatch.setattr(jobs_service, "compute_score", lambda **k: ScoreResult(8.6, Recommendation.PRIORITY, {}))
    with session_scope() as s:
        magdeburg = _job(s, "ML Engineer", "Germany", location="Magdeburg")
        remote = _job(s, "Remote ML Engineer", "Germany", location="Magdeburg", remote="remote")
        berlin = _job(s, "Berlin ML Engineer", "Germany", location="Berlin")
        for j in (magdeburg, remote, berlin):
            jobs_service.rescore(j)
        assert magdeburg.recommendation == Recommendation.REJECTED and magdeburg.score_breakdown["region_min_score"]["min"] == 9
        assert remote.recommendation == berlin.recommendation == Recommendation.PRIORITY
    page = TestClient(app).get("/?applied=1&unreviewed=0&since_days=")
    assert "Berlin ML Engineer" in page.text and ">ML Engineer<" not in page.text
