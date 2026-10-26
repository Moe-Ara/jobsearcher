"""Untrusted sources: gig / AI-training marketplaces (micro1, Alignerr, Outlier, Mercor …) are skipped before any LLM
call — and precise matching keeps real jobs that merely mention similar words."""

from types import SimpleNamespace

import pytest

from app.config_files import get_prefs
from app.core.normalize import normalize_company
from app.db import session_scope
from app.enums import AnalysisStatus, FilterStatus, SourceType
from app.filters import apply_hard_filters
from app.services.ingest import ingest_posting
from app.sources.base import JobPosting

REAL_ML_JOB = ("We are looking for an ML Engineer in Munich. You will train large language models, build data pipelines "
               "in Python and deploy them with Docker. Full-time, permanent position.")


def job(**kw):
    base = dict(title="Machine Learning Engineer", company="Acme GmbH", location="Munich", country="Germany",
                remote_status="hybrid", description=REAL_ML_JOB, requirements="", responsibilities="", sector=None,
                years_experience_required=None, canonical_url="https://jobs.acme.de/ml-engineer", source_url=None,
                career_page_url=None, sources=[])
    base.update(kw)
    base["normalized_company"] = normalize_company(base["company"])
    return SimpleNamespace(**base)


def rule_hits(j):
    out = apply_hard_filters(j, get_prefs())
    return out, [r for r in out.reasons if r["rule"] == "untrusted_source"]


@pytest.mark.parametrize("company", ["micro1", "micro1 AI", "Alignerr", "Mercor", "Mercor Inc.", "Outlier AI", "DataAnnotation",
                                     "Remotasks"])
def test_marketplace_companies_are_skipped(company):
    out, hits = rule_hits(job(company=company, title="Senior Software Engineer (Python)"))
    assert out.status == FilterStatus.REJECTED and hits, company


@pytest.mark.parametrize("url", ["https://jobs.micro1.ai/apply/123", "https://app.alignerr.com/jobs/9", "https://work.mercor.com/x"])
def test_marketplace_domains_in_job_urls_are_skipped(url):
    _, hits = rule_hits(job(company="Some Recruiter", canonical_url=url))
    assert hits and "untrusted-sources" in hits[0]["detail"]


def test_marketplace_links_inside_the_posting_are_skipped():
    _, hits = rule_hits(job(company="Talent Partner", description=REAL_ML_JOB + " Apply via outlier.ai/experts today."))
    assert hits


def test_source_urls_count_too():
    _, hits = rule_hits(job(company="LinkedIn poster", sources=[SimpleNamespace(url="https://dataannotation.tech/jobs")]))
    assert hits


@pytest.mark.parametrize("title", ["AI Trainer (German)", "LLM Tutor – Python", "RLHF Specialist", "AI Training Expert (Coding)",
                                   "Prompt Evaluator", "Data Annotator"])
def test_gig_titles_are_skipped(title):
    _, hits = rule_hits(job(title=title))
    assert hits, title


def test_platform_wording_needs_a_gig_signal():
    gig = ("Help train the next generation of AI models with your coding expertise. Freelance, $40/hr, flexible hours.")
    _, hits = rule_hits(job(title="Python Developer", description=gig))
    assert hits
    one_phrase_no_gig = "Join our talent network to hear about new roles. " + REAL_ML_JOB
    _, hits = rule_hits(job(description=one_phrase_no_gig))
    assert not hits


@pytest.mark.parametrize("kw", [
    dict(description=REAL_ML_JOB),                                                    # trains LLMs, permanent role
    dict(description="Whatever happens, we ship. Kotlin backend role."),              # "appen" inside "happens"
    dict(description="Tools: Braintrust for LLM evals, LangSmith, W&B."),              # a tool, not the marketplace
    dict(description="You work with our AI Trainer module for course content."),       # phrase in description only
    dict(company="Outlier Ventures Research"),                                         # different company, same first word
    dict(company="Turing Labs GmbH"),                                                  # Turing only matched by domain
    dict(company="Crossover Health Systems"),
])
def test_real_jobs_are_not_caught(kw):
    _, hits = rule_hits(job(**kw))
    assert not hits, (kw, hits)


def test_ingested_marketplace_job_is_hidden_and_never_analysed():
    with session_scope() as s:
        result = ingest_posting(s, JobPosting(source="manual", source_type=SourceType.MANUAL, title="Senior Python Engineer",
                                              company="micro1", location="Remote, Germany", description=REAL_ML_JOB))
        j = result.job
        assert j.filter_status == FilterStatus.REJECTED
        assert j.analysis_status == AnalysisStatus.SKIPPED
        assert any(r["rule"] == "untrusted_source" for r in j.filter_reasons)


def test_list_is_editable_in_preferences():
    u = get_prefs().untrusted_sources
    assert "micro1" in u.companies and "Alignerr" in u.companies and "alignerr.com" in u.domains
