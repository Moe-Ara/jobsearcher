from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.config_files import get_prefs
from app.db import session_scope
from app.enums import AnalysisStatus, FilterStatus, SourceType
from app.main import app
from app.models import Job
from app.services import drive_time, internships
from app.services.ingest import ingest_posting
from app.services.internships import intern_filters, is_internship, location_points
from app.sources.base import JobPosting

GOOD = "Praktikum Softwareentwicklung KI (m/w/d)"


def intern(**kw):
    base = dict(title=GOOD, company="ifm", normalized_company="ifm", description="Wir entwickeln KI-Software.",
                requirements="", responsibilities="", sector=None, country="Germany", remote_status=None,
                location="Tettnang", raw_source_data={"drive_minutes": 18}, source_url="https://ifm.com/jobs/1",
                canonical_url=None, career_page_url=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_internship_titles_are_recognised_but_not_werkstudent_or_internal_roles():
    prefs = get_prefs()
    assert is_internship("Praktikant (m/w/d) Data Science", prefs) and is_internship("Software Engineering Intern", prefs)
    assert is_internship("Trainee Programm IT", prefs)
    assert not is_internship("Werkstudent Software", prefs) and not is_internship("Internal Audit Manager", prefs)
    assert not is_internship("International Sales Engineer", prefs)


def test_only_graduate_friendly_internships_in_your_field_pass():
    prefs = get_prefs()
    assert not intern_filters(intern(), prefs).rejected
    assert intern_filters(intern(title="Pflichtpraktikum Softwareentwicklung"), prefs).rejected
    assert intern_filters(intern(title="Schülerpraktikum IT"), prefs).rejected
    assert intern_filters(intern(title="Praktikum Online Marketing"), prefs).rejected
    assert intern_filters(intern(title="Praktikum im Fachbereich Finanzen"), prefs).rejected
    assert intern_filters(intern(title="Praktikum Einkauf Digitalisierung"), prefs).rejected  # field word, wrong function
    student_only = "Du bist an einer Hochschule eingeschrieben und suchst ein Praktikum."
    out = intern_filters(intern(description=student_only), prefs)
    assert out.rejected and out.reasons[0]["rule"] == "intern_enrollment"
    assert not intern_filters(intern(description=student_only + " Auch Absolventen sind willkommen."), prefs).rejected
    assert intern_filters(intern(description="Wir suchen dich im Studium (Informatik)."), prefs).rejected
    flagged = intern_filters(intern(description=student_only), prefs.model_copy(update={
        "internships": prefs.internships.model_copy(update={"student_only": "flag"})}))
    assert not flagged.rejected and flagged.reasons[0]["rule"] == "intern_enrollment"
    assert intern_filters(intern(description="What we need to see: Pursuing MS or PhD in Machine Learning"), prefs).rejected


def test_internships_must_be_near_home_or_fully_remote():
    prefs = get_prefs()
    far = intern_filters(intern(raw_source_data={"drive_minutes": 70}), prefs)
    assert far.rejected and far.reasons[0]["rule"] == "intern_distance"
    assert not intern_filters(intern(raw_source_data={"drive_minutes": 70}, remote_status="remote"), prefs).rejected
    assert intern_filters(intern(country="Austria", raw_source_data={}), prefs).rejected
    assert intern_filters(intern(location="Germany", raw_source_data={}), prefs).rejected  # no town: can't tell
    waiting = intern_filters(intern(location="Ravensburg", raw_source_data={}), prefs)
    assert not waiting.rejected and waiting.reasons[0]["rule"] == internships.PENDING_RULE
    assert location_points(intern(raw_source_data={"drive_minutes": 10}), prefs) > location_points(intern(), prefs)


def test_place_query_drops_work_modes_and_country_only_locations():
    assert drive_time.place_query("Friedrichshafen, Baden-Württemberg, Germany (Hybrid)") == \
        "Friedrichshafen, Baden-Württemberg, Deutschland"
    assert drive_time.place_query("Stuttgart; Friedrichshafen") == "Stuttgart, Deutschland"
    assert drive_time.place_query("Germany (Remote)") is None


def _posting(title, location="Ravensburg", **kw):
    return JobPosting(source="greenhouse", source_type=SourceType.EMPLOYER, external_job_id=title[:20], url=f"https://x.com/jobs/{abs(hash(title))}",
                      title=title, company="Bodensee Robotics", location=location, country="Germany",
                      description="Wir entwickeln Software für Robotik. Absolventen willkommen.", **kw)


def test_internship_waits_for_its_drive_time_then_is_analysed(monkeypatch):
    calls = []

    def fake_geocode(queries, http=None, limit=60):
        cache = drive_time._load("geocode.json")
        for q in queries:
            calls.append(q)
            cache[q.lower()] = [47.78, 9.61]
        drive_time._save(cache, "geocode.json")
        return {q: (47.78, 9.61) for q in queries}

    monkeypatch.setattr(drive_time, "geocode", fake_geocode)
    monkeypatch.setattr(drive_time, "drive_minutes", lambda home, places, http=None: {drive_time.key(47.78, 9.61): 24.0})
    with session_scope() as s:
        job = ingest_posting(s, _posting("Praktikum Robotik Software (m/w/d)", location="Weingarten")).job
        main = ingest_posting(s, _posting("Software Engineer Robotics (m/w/d)")).job
        job_id = job.id
        assert job.track == "intern" and main.track == "main"
        assert job.filter_status == FilterStatus.FLAGGED and job.analysis_status == AnalysisStatus.SKIPPED  # waits
    result = internships.resolve_distances()
    assert result["resolved"] == 1 and calls == ["Weingarten, Deutschland"]
    with session_scope() as s:
        job = s.get(Job, job_id)
        assert job.raw_source_data["drive_minutes"] == 24.0
        assert job.filter_status == FilterStatus.PASSED and job.analysis_status == AnalysisStatus.PENDING


def test_internships_page_lists_matches_and_explains_what_was_filtered_out():
    with session_scope() as s:
        ok = ingest_posting(s, _posting("Praktikum Data Science (m/w/d)", raw={"drive_minutes": 20})).job
        ingest_posting(s, _posting("Praktikum Marketing (m/w/d)", raw={"drive_minutes": 20}))
        assert ok.track == "intern" and ok.filter_status == FilterStatus.PASSED
    page = TestClient(app).get("/internships")
    assert page.status_code == 200
    assert "Praktikum Data Science" in page.text and "Praktikum Marketing" not in page.text
    assert "1 not your field" in page.text and "🚗 20 min" in page.text
    hidden = TestClient(app).get("/internships?hidden=1")
    assert "Praktikum Marketing" in hidden.text


def test_enrolled_students_see_student_internships(monkeypatch):
    from app.config_files import get_profile

    prefs = get_prefs()
    student_only = "Du bist an einer Hochschule eingeschrieben."
    assert intern_filters(intern(title="Pflichtpraktikum Softwareentwicklung"), prefs).rejected
    monkeypatch.setattr(get_profile(), "enrolled_student", True)
    assert not intern_filters(intern(title="Pflichtpraktikum Softwareentwicklung"), prefs).rejected
    assert not intern_filters(intern(description=student_only), prefs).rejected
    assert intern_filters(intern(title="Schülerpraktikum IT"), prefs).rejected  # never, enrolled or not


def test_home_comes_from_locations_with_the_old_side_job_setting_as_fallback():
    prefs = get_prefs()
    town, lat, lon = drive_time.home_point(prefs)
    assert town and lat and lon
    old = prefs.model_copy(update={
        "locations": prefs.locations.model_copy(update={"home": "", "home_lat": None, "home_lon": None}),
        "side_jobs": prefs.side_jobs.model_copy(update={"home": "Ulm", "home_lat": 48.4, "home_lon": 9.99})})
    assert drive_time.home_point(old) == ("Ulm", 48.4, 9.99)
    nowhere = old.model_copy(update={"side_jobs": prefs.side_jobs.model_copy(update={"home": "", "home_lat": None, "home_lon": None})})
    out = intern_filters(intern(raw_source_data={}), nowhere)
    assert not out.rejected and out.reasons[0]["rule"] == "intern_no_home"
