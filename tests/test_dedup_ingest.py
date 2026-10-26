from app.core.normalize import normalize_company, normalize_title
from app.db import session_scope
from app.dedup import Candidate, same_vacancy
from app.enums import SourceType
from app.models import Job, JobSource, TargetCompany
from app.services.ingest import ingest_posting
from app.sources.base import JobPosting

DESC = (
    "Bosch is looking for an AI Engineer to design and deploy machine learning models for industrial "
    "manufacturing. You will build data pipelines in Python, containerise services with Docker, set up CI/CD "
    "and work with plant engineers on predictive maintenance and visual inspection. Requirements: degree in "
    "computer science or similar, 3+ years experience, PyTorch, MLOps, cloud. "
)


def cand(company, title, location, description=""):
    return Candidate(normalize_company(company), normalize_title(title), title, location, description)


def test_bosch_example_is_same_vacancy():
    a = cand("Bosch", "AI Engineer (m/w/d)", "Stuttgart")
    b = cand("Robert Bosch GmbH", "Artificial Intelligence Engineer", "Stuttgart, Germany")
    assert same_vacancy(a, b).same


def test_guards_against_false_merges():
    assert not same_vacancy(cand("Bosch", "Senior AI Engineer", "Stuttgart"), cand("Bosch", "Junior AI Engineer", "Stuttgart")).same
    assert not same_vacancy(cand("Bosch", "AI Engineer", "Stuttgart"), cand("Bosch", "AI Engineer", "Dresden")).same
    assert not same_vacancy(cand("Bosch", "AI Engineer", "Stuttgart"), cand("Siemens", "AI Engineer", "Stuttgart")).same
    assert not same_vacancy(cand("Bosch", "AI Engineer", "Stuttgart"), cand("Bosch", "Backend Engineer", "Stuttgart")).same


def test_description_similarity_used_when_available():
    a = cand("Bosch", "AI Engineer", "Stuttgart", DESC)
    b = cand("Bosch", "Engineer Artificial Intelligence – Manufacturing", "Stuttgart", DESC + " Apply now!")
    assert same_vacancy(a, b).same
    c = cand("Bosch", "AI Engineer", "Stuttgart", "Completely different role about HR payroll systems. " * 10)
    assert not same_vacancy(a, c).same


def posting(source, source_type, ext, url, title, company="Bosch", location="Stuttgart", description=""):
    return JobPosting(source=source, source_type=source_type, external_job_id=ext, url=url, title=title,
                      company=company, location=location, description=description)


def test_board_then_employer_merges_and_employer_becomes_canonical():
    with session_scope() as s:
        tc = TargetCompany(company_name="Bosch", normalized_name="bosch", aliases=["Robert Bosch GmbH"], priority="A", ats_config={})
        s.add(tc)
    with session_scope() as s:
        r1 = ingest_posting(s, posting("linkedin", SourceType.JOB_BOARD, "3912345678",
                                       "https://www.linkedin.com/comm/jobs/view/3912345678/?trk=x", "AI Engineer (m/w/d)",
                                       company="Robert Bosch GmbH", description="Short snippet"))
        assert r1.outcome == "new"
        assert r1.job.target_company_id is not None  # alias resolved
    with session_scope() as s:
        tc = s.query(TargetCompany).one()
        r2 = ingest_posting(s, posting("smartrecruiters", SourceType.EMPLOYER, "744000", "https://jobs.smartrecruiters.com/BoschGroup/744000",
                                       "Artificial Intelligence Engineer", location="Stuttgart, Germany", description=DESC), target_company=tc)
        assert r2.outcome == "merged"
    with session_scope() as s:
        jobs = s.query(Job).all()
        assert len(jobs) == 1
        job = jobs[0]
        assert job.best_source_type == "employer" and job.source == "smartrecruiters"
        assert job.canonical_url == "https://jobs.smartrecruiters.com/BoschGroup/744000"
        assert job.description.startswith("Bosch is looking")
        assert {src.source for src in job.sources} == {"linkedin", "smartrecruiters"}  # provenance kept


def test_reingest_same_source_is_unchanged_and_updates_on_change():
    with session_scope() as s:
        r = ingest_posting(s, posting("greenhouse", SourceType.EMPLOYER, "1", "https://boards.greenhouse.io/x/jobs/1", "Robotics Engineer", description=DESC))
        assert r.outcome == "new"
    with session_scope() as s:
        assert ingest_posting(s, posting("greenhouse", SourceType.EMPLOYER, "1", "https://boards.greenhouse.io/x/jobs/1", "Robotics Engineer", description=DESC)).outcome == "unchanged"
    with session_scope() as s:
        r = ingest_posting(s, posting("greenhouse", SourceType.EMPLOYER, "1", "https://boards.greenhouse.io/x/jobs/1", "Robotics Engineer", description=DESC + " New paragraph."))
        assert r.outcome == "updated"
        assert s.query(JobSource).count() == 1


def test_two_ids_from_same_source_never_merge():
    with session_scope() as s:
        ingest_posting(s, posting("greenhouse", SourceType.EMPLOYER, "1", "https://boards.greenhouse.io/x/jobs/1", "Software Engineer"))
        r = ingest_posting(s, posting("greenhouse", SourceType.EMPLOYER, "2", "https://boards.greenhouse.io/x/jobs/2", "Software Engineer"))
        assert r.outcome == "new"
