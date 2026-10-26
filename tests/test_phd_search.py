"""Funded PhD search: academic listing adapter, funding / field hard filters, aggregator employers, longer retention."""

from datetime import timedelta
from types import SimpleNamespace

import pytest

from app.config_files import get_prefs
from app.core.normalize import normalize_company
from app.db import session_scope
from app.enums import FilterStatus, SourceType
from app.filters import apply_hard_filters
from app.models import TargetCompany, utcnow
from app.services.ingest import ingest_posting
from app.services.retention import posted_too_long_ago
from app.sources.academic import AcademicListingAdapter, guess_country
from app.sources.base import CompanyFeedConfig, JobPosting
from app.sources.registry import ADAPTERS

ML_PHD = ("The Chair of Robotics invites applications for a doctoral researcher in machine learning for autonomous "
          "robots. You will work on deep learning and computer vision for perception, implement methods in Python "
          "and C++ and publish at top venues. Requirements: Master's degree in computer science or a related field.")


def job(**kw):
    base = dict(title="PhD Position in Machine Learning for Robotics (m/w/d)", company="Universität Stuttgart",
                location="Stuttgart", country="Germany", remote_status=None,
                description=ML_PHD + " Salary according to TV-L E13 (100 %).", requirements="", responsibilities="",
                sector=None, years_experience_required=None, canonical_url="https://uni.example/phd", source_url=None,
                career_page_url=None, sources=[])
    base.update(kw)
    base["normalized_company"] = normalize_company(base["company"])
    return SimpleNamespace(**base)


def rules(j):
    out = apply_hard_filters(j, get_prefs())
    return out, {r["rule"]: r for r in out.reasons}


# ------------------------------------------------------------------ hard filters


@pytest.mark.parametrize("title", [
    "PhD Position in Machine Learning for Robotics (m/w/d)",
    "Doktorand*in Künstliche Intelligenz (m/w/d)",
    "Wissenschaftliche*r Mitarbeiter*in (Doktorand*in) Informatik",
    "Doctoral Researcher – Autonomous Systems",
])
def test_funded_phd_in_field_passes(title):
    out, hits = rules(job(title=title))
    assert out.status == FilterStatus.PASSED, out.reasons


@pytest.mark.parametrize("title", [
    "Postdoc in Machine Learning", "Postdoctoral Researcher – Robotics", "Professor (W2) für Künstliche Intelligenz",
    "Masterarbeit: Deep Learning für Bildverarbeitung", "Studentische Hilfskraft (HiWi) Robotik",
    "Postdoktorand (w/m/d) für KI und Maschinelles Lernen", "Wissenschaftlerin / Wissenschaftler mit Promotion (m/w/d) KI",
    "W2-Professur für Robotik", "Professur (W3) für Maschinelles Lernen",
])
def test_positions_needing_a_doctorate_or_for_students_are_rejected(title):
    out, hits = rules(job(title=title))
    assert out.status == FilterStatus.REJECTED and "seniority_or_type" in hits, title


def test_self_funded_phd_is_rejected():
    out, hits = rules(job(description=ML_PHD + " This is a self-funded PhD; applicants must secure their own funding."))
    assert out.status == FilterStatus.REJECTED and "phd_funding" in hits


def test_phd_without_funding_information_is_flagged():
    out, hits = rules(job(description=ML_PHD))
    assert out.status == FilterStatus.FLAGGED and hits["phd_funding"]["severity"] == "flag"


def test_euraxess_eu_programme_field_is_not_read_as_unfunded():
    out, hits = rules(job(description="Is the job funded through the EU Research Framework Programme? Not funded by a "
                                      "EU programme. " + ML_PHD))
    assert "phd_funding" not in hits or hits["phd_funding"]["severity"] == "flag"
    assert out.status != FilterStatus.REJECTED


def test_wissenschaftliche_mitarbeiter_counts_as_paid_position():
    out, hits = rules(job(title="Wissenschaftliche*r Mitarbeiter*in Robotik", description=ML_PHD))
    assert out.status == FilterStatus.PASSED, out.reasons


def test_phd_outside_the_field_is_rejected():
    out, hits = rules(job(title="PhD student in Recycling of Solid Electrolytes for Batteries",
                          description="Doctoral position in electrochemistry and materials science. Salary per TV-L E13."))
    assert out.status == FilterStatus.REJECTED and "phd_field" in hits


def test_field_is_recognised_from_the_text_when_the_title_is_generic():
    out, hits = rules(job(title="Doktorand*in (m/w/d) am Lehrstuhl Prof. Müller", description=ML_PHD + " Vergütung E13."))
    assert out.status == FilterStatus.PASSED, out.reasons


def test_ordinary_jobs_are_not_touched_by_the_phd_rules():
    out, hits = rules(job(title="Machine Learning Engineer", description="Build ML systems in Python. Permanent role."))
    assert "phd_funding" not in hits and "phd_field" not in hits


def test_phd_abroad_is_still_subject_to_target_countries():
    out, hits = rules(job(country="Spain", location="Madrid"))
    assert out.status == FilterStatus.REJECTED and "country" in hits


# ------------------------------------------------------------------ adapter

LISTING = """<html><body>
<a href="/jobs/wissenschaftliche-r-mitarbeiter-in-doktorand-in-informatik-uni-ulm-1111">Wissenschaftliche*r Mitarbeiter*in
  (Doktorand*in) Informatik Universität Ulm</a>
<a href="/jobs/sachbearbeiter-in-personal-uni-ulm-2222">Sachbearbeiter*in Personal Universität Ulm</a>
<a href="/jobs/phd-position-robotics-tu-darmstadt-3333">PhD Position in Robotics at TU Darmstadt</a>
<a href="/jobs/phd-position-robotics-tu-darmstadt-3333">PhD Position in Robotics at TU Darmstadt</a>
<a href="/ueber-uns">Über uns</a>
</body></html>"""

JSONLD_DETAIL = """<html><head><script type="application/ld+json">{"@context": "https://schema.org", "@type": "JobPosting",
 "title": "Wissenschaftliche*r Mitarbeiter*in (Doktorand*in) Informatik", "identifier": "1111",
 "hiringOrganization": {"@type": "Organization", "name": "Universität Ulm"},
 "jobLocation": {"@type": "Place", "address": {"addressLocality": "Ulm", "addressCountry": "DE"}},
 "description": "<p>Promotion im Bereich maschinelles Lernen. E 13 TV-L.</p>", "datePosted": "2026-09-20"}</script>
</head><body><main>Promotion im Bereich maschinelles Lernen. E 13 TV-L.</main></body></html>"""

EURAXESS_DETAIL = """<html><head><meta property="og:title" content="PhD Position in Robotics"></head><body><main>
16 Oct 2026 Job Information Organisation/Company Technische Universität Darmstadt Research Field Computer science
Researcher Profile First Stage Researcher (R1) Application Deadline 15 Nov 2026 Country Germany Type of Contract
Temporary Job Status Full-time Offer Description We offer a fully funded PhD position in robot learning.
</main></body></html>"""


class FakeResponse:
    def __init__(self, text, ctype="text/html"):
        self.text, self.content, self.headers = text, text.encode(), {"content-type": ctype}


class FakeHttp:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def get_text(self, url, params=None):
        self.calls.append(url)
        return self.pages[url]

    def get(self, url, params=None, accept=None):
        self.calls.append(url)
        return FakeResponse(self.pages[url])


def board_cfg(**options):
    return CompanyFeedConfig(company_name="academics.de (PhD search)", careers_url="https://www.academics.de/search",
                             options={"job_link_pattern": r"academics\.de/jobs/[a-z0-9-]+-\d+$", "board": True,
                                      "employer_from_posting": True, **options})


def test_adapter_is_registered():
    assert ADAPTERS["academic"] is AcademicListingAdapter


def test_listing_keeps_only_research_position_links_once():
    http = FakeHttp({"https://www.academics.de/search": LISTING, "https://www.academics.de/q2": LISTING})
    raws = AcademicListingAdapter(board_cfg(listing_urls=["https://www.academics.de/q2"]), http).fetch_jobs()
    assert [r["url"].rsplit("-", 1)[1] for r in raws] == ["1111", "3333"]


def test_listing_fails_only_when_every_page_fails():
    class Broken(FakeHttp):
        def get_text(self, url, params=None):
            if "broken" in url:
                raise RuntimeError("HTTP 500")
            return super().get_text(url)

    ok = Broken({"https://www.academics.de/search": LISTING})
    assert AcademicListingAdapter(board_cfg(listing_urls=["https://broken.example/x"]), ok).fetch_jobs()
    with pytest.raises(RuntimeError):
        AcademicListingAdapter(CompanyFeedConfig(company_name="x", careers_url="https://broken.example/a",
                                                 options={"job_link_pattern": "jobs"}), Broken({})).fetch_jobs()


def test_jsonld_posting_uses_the_real_employer_not_the_board():
    url = "https://www.academics.de/jobs/wissenschaftliche-r-mitarbeiter-in-doktorand-in-informatik-uni-ulm-1111"
    adapter = AcademicListingAdapter(board_cfg(), FakeHttp({url: JSONLD_DETAIL}))
    p = adapter.normalize_job(adapter.fetch_details({"kind": "link", "url": url, "anchor": "Doktorand*in"}))
    assert p.company == "Universität Ulm" and p.source == "academic" and p.source_type == SourceType.JOB_BOARD
    assert p.url == url and "E 13" in p.description


def test_page_posting_reads_labels_for_employer_and_country():
    url = "https://euraxess.ec.europa.eu/jobs/473154"
    cfg = CompanyFeedConfig(company_name="EURAXESS (PhD search)", careers_url="https://euraxess.ec.europa.eu/jobs/search",
                            options={"job_link_pattern": r"jobs/\d+$", "board": True, "employer_from_posting": True})
    adapter = AcademicListingAdapter(cfg, FakeHttp({url: EURAXESS_DETAIL}))
    p = adapter.normalize_job(adapter.fetch_details({"kind": "link", "url": url, "anchor": "PhD Position in Robotics"}))
    assert p.company == "Technische Universität Darmstadt" and p.country == "Germany"


def test_employer_from_title_when_page_has_no_labels():
    url = "https://ellis.eu/research/jobs/2026-10-22-phd"
    page = "<html><body><main>PhD Position in Deep Learning at TU Darmstadt. Apply by 30/11/26. Fully funded.</main></body></html>"
    cfg = CompanyFeedConfig(company_name="ELLIS (PhD positions)", careers_url="https://ellis.eu/research/jobs",
                            options={"job_link_pattern": "jobs/", "board": True, "employer_from_posting": True})
    adapter = AcademicListingAdapter(cfg, FakeHttp({url: page}))
    p = adapter.normalize_job(adapter.fetch_details({"kind": "link", "url": url, "anchor": "PhD Position in Deep Learning at TU Darmstadt"}))
    assert p.company == "TU Darmstadt" and p.country == "Germany"


def test_employer_portal_keeps_its_configured_location():
    url = "https://portal.mytum.de/jobs/wissenschaftler/NewsArticle_1"
    cfg = CompanyFeedConfig(company_name="Technische Universität München", careers_url="https://portal.mytum.de/jobs",
                            options={"job_link_pattern": "NewsArticle_"}, country="Germany", city="Munich")
    page = "<html><body><main>Doctoral Research Associate in robotics, joint project with Singapore.</main></body></html>"
    adapter = AcademicListingAdapter(cfg, FakeHttp({url: page}))
    p = adapter.normalize_job(adapter.fetch_details({"kind": "link", "url": url, "anchor": "Doctoral Research Associate"}))
    assert p.company == "Technische Universität München" and p.location == "Munich, Germany"
    assert p.source_type == SourceType.EMPLOYER


def test_guess_country_from_title_and_text():
    assert guess_country("PhD position at Universidad Autónoma de Madrid", "") == "Spain"
    assert guess_country("Doktorand*in Informatik", "Doktorand*in Informatik Universität Duisburg-Essen Essen") == "Germany"


# ------------------------------------------------------------------ ingest + retention


def test_aggregator_feed_keeps_provenance_but_not_the_company():
    with session_scope() as s:
        feed = TargetCompany(company_name="academics.de (PhD search)", normalized_name="academicsde phd search",
                             ats_type="academic", ats_config={}, aliases=[], priority="B")
        s.add(feed)
        s.flush()
        posting = JobPosting(source="academic", source_type=SourceType.JOB_BOARD, external_job_id="https://a.de/jobs/x-9",
                             url="https://a.de/jobs/x-9", title="Doktorand*in Künstliche Intelligenz",
                             company="Hochschule Ravensburg-Weingarten", location="Weingarten", country="Germany",
                             description=ML_PHD + " Vergütung nach TV-L E13.")
        result = ingest_posting(s, posting, target_company=feed, employer_from_posting=True)
        assert result.outcome == "new"
        assert result.job.company == "Hochschule Ravensburg-Weingarten" and result.job.target_company_id is None
        assert result.source.target_company_id == feed.id
        assert result.job.filter_status == FilterStatus.PASSED, result.job.filter_reasons


def test_academic_postings_get_a_longer_retention_window():
    five_weeks = utcnow() - timedelta(days=35)
    assert posted_too_long_ago(five_weeks) is True
    assert posted_too_long_ago(five_weeks, academic=True) is False
    assert posted_too_long_ago(utcnow() - timedelta(days=90), academic=True) is True


def test_import_button_reaches_the_import_route(monkeypatch):
    from fastapi.testclient import TestClient

    import app.services.companies as companies
    from app.main import app

    monkeypatch.setattr(companies, "import_companies_yaml", lambda: {"created": 5})
    r = TestClient(app).post("/companies/import", follow_redirects=False)
    assert r.status_code == 303 and "Imported+5" in r.headers["location"].replace("%20", "+")


def test_research_position_at_a_chair_is_not_a_professorship():
    out, hits = rules(job(title="Wissenschaftliche/r Mitarbeiter/in an der Professur für Maschinelles Lernen"))
    assert out.status == FilterStatus.PASSED, out.reasons


def test_academic_source_postings_are_field_checked_whatever_the_title():
    out, hits = rules(job(title="Promotions-Stipendien im Promotionskolleg Arbeit und Machtressourcen", source="academic",
                          description="Stipendium für Promotionen in Soziologie und Politikwissenschaft."))
    assert out.status == FilterStatus.REJECTED and "phd_field" in hits


def test_weak_title_words_need_field_evidence():
    out, hits = rules(job(title="Wissenschaftliche*r Mitarbeiter*in thermochemisches Recycling – Prozesssimulation",
                          description="Prozesssimulation von Vergasungsanlagen und Energiesystemmodellierung. E 13 TV-L."))
    assert out.status == FilterStatus.REJECTED and "phd_field" in hits


def test_board_location_abroad_becomes_a_non_target_country():
    url = "https://www.academics.de/jobs/phd-positions-aarhus-9"
    detail = JSONLD_DETAIL.replace('"addressLocality": "Ulm", "addressCountry": "DE"', '"addressLocality": "Aarhus, Andere Länder"')
    adapter = AcademicListingAdapter(board_cfg(), FakeHttp({url: detail}))
    p = adapter.normalize_job(adapter.fetch_details({"kind": "link", "url": url, "anchor": "PhD Positions"}))
    assert p.country == "Outside Germany"
