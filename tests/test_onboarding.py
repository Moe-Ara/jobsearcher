import io
from datetime import date

import pytest
import yaml
from fastapi.testclient import TestClient

from app.config_files import CandidateProfile
from app.llm.factory import LLMSetup
from app.main import app
from app.services import config_writer, onboarding
from app.services.cv import MasterCV, build_cv, render_text
from app.services.onboarding import CvDegree, CvExtraction, CvJob, CvPerson, CvProject
from tests.conftest import FakeProvider

EX = CvExtraction(
    cv_language="en",
    person=CvPerson(name="Lena Kraus", headline="Backend Developer", email="lena@example.org", location="Stuttgart",
                    links=["github.com/lenak", "Unity Asset Store"]),
    experience=[
        CvJob(employer="Acme GmbH", position="Software Developer", start="03/2021", end="present",
              bullets=["Built REST services in Java and Spring Boot", "Introduced Docker-based CI"], technologies=["Java", "Docker"]),
        CvJob(employer="Beta AG", position="Working student", start="01/2019", end="02/2021", bullets=["Python test tooling"]),
    ],
    education=[CvDegree(degree="M.Sc. Computer Science", institution="Uni Stuttgart", start="2019", end="2021",
                        thesis_title="Graph neural networks for logistics", focus="Machine Learning")],
    projects=[CvProject(name="Chess bot", description="Reinforcement learning agent in Rust", technologies=["Rust"])],
    skills=["Java", "Spring Boot", "Python", "Machine Learning", "Rust", "Kubernetes"],
)


def test_skills_are_grouped_by_where_the_cv_evidences_them():
    levels = onboarding.skill_levels(EX)
    assert {"Java", "Spring Boot", "Python", "Docker"} <= set(levels["direct_professional"])
    assert levels["academic"] == ["Machine Learning"] and levels["project"] == ["Rust"]
    assert levels["listed"] == ["Kubernetes"]  # named on the CV, but no job / degree / project shows it


def test_years_of_experience_counts_overlaps_once():
    jobs = [CvJob(employer="A", start="01/2020", end="12/2021"), CvJob(employer="B", start="06/2021", end="12/2022")]
    assert onboarding.years_of_experience(jobs, today=date(2026, 1, 1)) == 2.9
    assert onboarding.years_of_experience([CvJob(employer="C")]) is None


def test_conversion_produces_files_the_app_can_load_and_render():
    profile = CandidateProfile.model_validate(onboarding.to_profile(EX))
    assert profile.headline == "Backend Developer" and profile.experience[0].employer == "Acme GmbH"
    master = MasterCV.model_validate(onboarding.to_master_cv(EX))
    assert master.name == "Lena Kraus" and [x.url for x in master.contact.links] == ["https://github.com/lenak"]
    assert master.experience[0].bullets[0].keywords == ["Java", "Spring Boot"]
    for lang in ("en", "de"):  # a single-language CV still renders in both languages
        text = render_text(build_cv(master, lang, [], statement=""), with_extras=False)
        assert "Built REST services" in text and "Uni Stuttgart" in text


def test_cv_text_is_read_from_pdf_word_and_text_files():
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for i, line in enumerate(["Lena Kraus", "Software Developer at Acme GmbH since 2021"] * 4):
        c.drawString(72, 750 - 16 * i, line)
    c.save()
    assert "Acme GmbH" in onboarding.extract_text("cv.pdf", buf.getvalue())
    import docx

    d = docx.Document()
    d.add_paragraph("Lena Kraus — Backend Developer")
    out = io.BytesIO()
    d.save(out)
    assert "Backend Developer" in onboarding.extract_text("cv.docx", out.getvalue())
    assert onboarding.extract_text("cv.txt", "Hällo".encode()) == "Hällo"
    with pytest.raises(ValueError):
        onboarding.extract_text("cv.png", b"\x89PNG")


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """Config writes go to a temporary folder — never the real config/."""
    from app.config_files import config_store

    for name in ("profile.yaml", "cv.yaml", "preferences.yaml"):
        src = config_writer.config_path(name)
        (tmp_path / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(config_writer, "config_path", lambda name: tmp_path / name)
    monkeypatch.setattr(config_store, "reload", lambda: None)
    monkeypatch.setattr(onboarding, "setup_dir", lambda: tmp_path)
    return tmp_path


def test_cv_import_writes_drafts_and_saving_validates_first(tmp_config, monkeypatch):
    fake = FakeProvider()
    fake.responses["CvExtraction"] = EX
    setup = LLMSetup.single(fake, "fake-strong", "fake-cheap")
    monkeypatch.setattr("app.llm.factory.get_llm", lambda: setup)
    res = onboarding.run_cv_import("Lena Kraus\nSoftware Developer at Acme GmbH ..." * 10)
    assert res["jobs"] == 2 and "Lena Kraus" in fake.last_user["CvExtraction"]
    drafts = onboarding.drafts()
    assert set(drafts) == {"profile.yaml", "cv.yaml"} and "Acme GmbH" in drafts["cv.yaml"]
    before = (tmp_config / "profile.yaml").read_text(encoding="utf-8")
    with pytest.raises(config_writer.ConfigError):
        onboarding.save_drafts({"profile.yaml": drafts["profile.yaml"], "cv.yaml": "name: [unclosed"})
    assert (tmp_config / "profile.yaml").read_text(encoding="utf-8") == before  # nothing written when one is invalid
    onboarding.save_drafts(drafts)
    assert "Acme GmbH" in (tmp_config / "profile.yaml").read_text(encoding="utf-8") and not onboarding.drafts()
    assert list((tmp_config / ".backups").glob("profile-*.yaml"))  # previous version kept


def test_settings_updates_keep_the_comments_in_the_yaml(tmp_config):
    config_writer.update("preferences.yaml", {"internships.max_drive_minutes": 30, "search.queries": ["data engineer"]})
    text = (tmp_config / "preferences.yaml").read_text(encoding="utf-8")
    assert "# Search preferences" in text or text.lstrip().startswith("#")
    data = yaml.safe_load(text)
    assert data["internships"]["max_drive_minutes"] == 30 and data["search"]["queries"] == ["data engineer"]
    with pytest.raises(config_writer.ConfigError):
        config_writer.update("preferences.yaml", {"internships.max_drive_minutes": "far"})


def test_setup_page_and_search_form(tmp_config, monkeypatch):
    calls = {}
    monkeypatch.setattr(config_writer, "update", lambda name, changes: calls.setdefault(name, changes))
    monkeypatch.setattr("app.services.drive_time.geocode", lambda queries, http=None, limit=60: {q: (48.78, 9.18) for q in queries})
    c = TestClient(app)
    page = c.get("/setup")
    assert page.status_code == 200 and "Checklist" in page.text and "Import CV" in page.text
    r = c.post("/setup/search", data={"countries": ["Germany", "UAE"], "other_countries": "Denmark",
                                      "queries": "data engineer\nML engineer", "home": "Stuttgart", "internships": "1"},
               follow_redirects=False)
    assert r.status_code == 303 and "saved" in r.headers["location"]
    ch = calls["preferences.yaml"]
    assert ch["locations.target_countries"] == ["Germany", "UAE", "Denmark"] and ch["search.queries"] == ["data engineer", "ML engineer"]
    assert ch["locations.home"] == "Stuttgart" and ch["locations.home_lat"] == 48.78 and ch["side_jobs.enabled"] is False
    assert c.get("/setup/edit/profile.yaml").status_code == 200
    assert "Unknown" in c.get("/setup/edit/secrets.env", follow_redirects=True).text


def test_no_model_schema_has_a_property_named_title():
    """Local structured-output engines drop a property called "title" (a JSON-Schema keyword): the field then comes
    back empty or the whole answer fails validation. Every schema sent to a model must avoid that name."""
    import inspect

    from pydantic import BaseModel

    from app.llm import schemas
    from app.services import cv, onboarding as ob, research

    def props(schema: dict) -> set[str]:
        names = set(schema.get("properties", {}))
        for sub in schema.get("$defs", {}).values():
            names |= set(sub.get("properties", {}))
        return names

    models = [m for mod in (schemas, ob, research) for _, m in inspect.getmembers(mod, inspect.isclass)
              if issubclass(m, BaseModel) and m.__module__ == mod.__name__] + [cv.CvProfileStatement]
    assert len(models) > 10
    offenders = [m.__name__ for m in models if "title" in props(m.model_json_schema())]
    assert offenders == []


def test_pasted_posting_fields_map_job_title_back_to_title(monkeypatch):
    from app.db import session_scope
    from app.llm.schemas import ExtractedPosting
    from app.services.analysis import extract_fields

    fake = FakeProvider()
    fake.responses["ExtractedPosting"] = ExtractedPosting(job_title="Data Engineer", company="Acme", location="Stuttgart",
                                                         country="Germany", employment_type="")
    monkeypatch.setattr("app.services.analysis.get_llm", lambda: LLMSetup.single(fake, "fake-strong", "fake-cheap"))
    with session_scope() as s:
        assert extract_fields(s, "Data Engineer at Acme") == {"title": "Data Engineer", "company": "Acme",
                                                              "location": "Stuttgart", "country": "Germany"}


def test_first_start_creates_your_config_from_the_examples_without_overwriting(tmp_path, monkeypatch):
    from app.config_files import ensure_config
    from app.settings import get_settings

    monkeypatch.setattr(get_settings(), "config_dir", tmp_path / "config")
    copied = ensure_config()
    assert {"preferences.yaml", "profile.yaml", "cv.yaml", "companies.yaml"} <= set(copied)
    assert (tmp_path / "config" / "letter_templates" / "README.txt").exists()
    (tmp_path / "config" / "profile.yaml").write_text("headline: Mine\n", encoding="utf-8")
    assert ensure_config() == [] and (tmp_path / "config" / "profile.yaml").read_text(encoding="utf-8") == "headline: Mine\n"
