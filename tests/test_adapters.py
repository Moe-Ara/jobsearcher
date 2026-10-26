import json

from app.sources.base import CompanyFeedConfig
from app.sources.generic import GenericCareerPageAdapter
from app.sources.greenhouse import GreenhouseAdapter
from app.sources.jsonld import extract_jsonld_postings, jsonld_to_fields
from app.sources.lever import LeverAdapter
from app.sources.personio import PersonioAdapter, parse_personio_xml
from app.sources.registry import detect_ats
from app.sources.smartrecruiters import SmartRecruitersAdapter

CFG = CompanyFeedConfig(company_name="Acme", identifier="acme", careers_url="https://acme.example/careers")


def test_greenhouse_normalize():
    raw = {
        "id": 4012, "title": "ML Engineer", "absolute_url": "https://boards.greenhouse.io/acme/jobs/4012",
        "location": {"name": "Berlin, Germany"}, "updated_at": "2026-09-01T10:00:00-04:00",
        "content": "&lt;p&gt;Build &lt;strong&gt;ML&lt;/strong&gt; systems.&lt;/p&gt;",
        "departments": [{"name": "AI"}], "offices": [],
    }
    p = GreenhouseAdapter(CFG, None).normalize_job(raw)
    assert p.external_job_id == "4012" and p.location == "Berlin, Germany"
    assert p.description == "Build ML systems." and p.department == "AI" and p.date_posted is not None
    assert "content" not in p.raw


def test_lever_normalize_splits_lists():
    raw = {
        "id": "abc-1", "text": "Robotics Software Engineer", "hostedUrl": "https://jobs.lever.co/acme/abc-1",
        "categories": {"location": "Munich", "commitment": "Full-time", "team": "Robotics"},
        "descriptionPlain": "Intro text", "createdAt": 1756000000000, "workplaceType": "hybrid",
        "lists": [{"text": "What you'll do", "content": "<li>Build robots</li>"}, {"text": "Requirements", "content": "<li>C++</li>"}],
        "salaryRange": {"min": 60000, "max": 80000, "currency": "EUR"},
    }
    p = LeverAdapter(CFG, None).normalize_job(raw)
    assert p.remote_status == "hybrid" and "C++" in p.requirements and "Build robots" in p.responsibilities
    assert p.salary_min == 60000 and p.employment_type == "Full-time"


PERSONIO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<workzag-jobs><position><id>123</id><subcompany>Acme GmbH</subcompany><office>Dresden</office>
<department>Engineering</department><name>Simulation Engineer (m/w/d)</name>
<jobDescriptions><jobDescription><name>Your tasks</name><value><![CDATA[<ul><li>Digital twins</li></ul>]]></value></jobDescription>
<jobDescription><name>Your profile</name><value><![CDATA[<p>Unity, C#</p>]]></value></jobDescription></jobDescriptions>
<employmentType>permanent</employmentType><seniority>experienced</seniority><schedule>full-time</schedule>
<createdAt>2026-09-10T08:00:00+00:00</createdAt></position></workzag-jobs>"""


def test_personio_parse_and_normalize():
    raws = parse_personio_xml(PERSONIO_XML)
    assert len(raws) == 1
    p = PersonioAdapter(CFG, None).normalize_job(raws[0])
    assert p.url == "https://acme.jobs.personio.de/job/123"
    assert "Digital twins" in p.responsibilities and "Unity" in p.requirements and p.location == "Dresden"
    json.dumps(p.raw)  # raw must be JSON-serialisable


def test_smartrecruiters_normalize():
    raw = {
        "id": "744", "name": "Computer Vision Engineer", "releasedDate": "2026-09-01T00:00:00.000Z",
        "location": {"city": "Renningen", "country": "de", "fullLocation": "Renningen, BW, Germany", "hybrid": True},
        "experienceLevel": {"label": "Mid-Senior Level"}, "typeOfEmployment": {"label": "Full-time"},
        "jobAd": {"sections": {"jobDescription": {"text": "<p>Perception for robots</p>"}, "qualifications": {"text": "<p>OpenCV</p>"}}},
    }
    p = SmartRecruitersAdapter(CFG, None).normalize_job(raw)
    assert p.remote_status == "hybrid" and p.country == "DE" and p.requirements == "OpenCV"
    assert p.url == "https://jobs.smartrecruiters.com/acme/744"


JSONLD_PAGE = """<html><head><script type="application/ld+json">{"@context":"https://schema.org","@graph":[{"@type":"Organization","name":"X"},
{"@type":"JobPosting","title":"Platform Engineer","datePosted":"2026-09-20","hiringOrganization":{"@type":"Organization","name":"Acme AG"},
"jobLocation":{"@type":"Place","address":{"addressLocality":"Riyadh","addressCountry":"SA"}},"jobLocationType":"TELECOMMUTE",
"description":"<p>Kubernetes &amp; Terraform</p>","identifier":{"@type":"PropertyValue","value":"PE-7"},
"baseSalary":{"currency":"SAR","value":{"minValue":20000,"maxValue":30000}}}]}</script></head><body></body></html>"""


def test_jsonld_extraction():
    nodes = extract_jsonld_postings(JSONLD_PAGE)
    assert len(nodes) == 1
    f = jsonld_to_fields(nodes[0])
    assert f["title"] == "Platform Engineer" and f["company"] == "Acme AG" and f["country"] == "SA"
    assert f["remote_status"] == "remote" and f["description"] == "Kubernetes & Terraform"
    assert f["salary_min"] == 20000 and f["external_job_id"] == "PE-7"


def test_generic_adapter_uses_jsonld():
    class FakeHttp:
        def get_text(self, url):
            return JSONLD_PAGE

    adapter = GenericCareerPageAdapter(CFG, FakeHttp())
    raws = adapter.fetch_jobs()
    p = adapter.normalize_job(adapter.fetch_details(raws[0]))
    assert p.title == "Platform Engineer" and p.company == "Acme" and p.external_job_id == "PE-7"


def test_detect_ats():
    assert detect_ats("https://job-boards.greenhouse.io/celonis/jobs/123").ats_type == "greenhouse"
    m = detect_ats("https://job-boards.eu.greenhouse.io/acme/jobs/55")
    assert (m.identifier, m.job_id, m.options) == ("acme", "55", {"region": "eu"})
    assert detect_ats("https://boards.greenhouse.io/embed/job_board?for=acme").identifier == "acme"
    m = detect_ats("https://jobs.lever.co/acme/abc-def")
    assert (m.ats_type, m.identifier, m.job_id) == ("lever", "acme", "abc-def")
    m = detect_ats("https://wandelbots.jobs.personio.de/job/1234567?display=en")
    assert (m.ats_type, m.identifier, m.job_id) == ("personio", "wandelbots", "1234567")
    m = detect_ats("https://jobs.smartrecruiters.com/BoschGroup/744000012345-ai-engineer")
    assert (m.ats_type, m.identifier, m.job_id) == ("smartrecruiters", "BoschGroup", "744000012345")
    assert detect_ats("https://careers.example.com/jobs").ats_type == "generic"


class _Http:
    """Answers by URL substring; records calls."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def _answer(self, url, payload=None):
        self.calls.append((url, payload))
        for key, value in self.routes.items():
            if key in url:
                return value(payload) if callable(value) else value
        raise AssertionError(f"unexpected {url}")

    def get_json(self, url, params=None):
        return self._answer(url, params)

    def post_json(self, url, payload):
        return self._answer(url, payload)


def test_ashby_workable_recruitee_normalize():
    from app.sources.ashby import AshbyAdapter
    from app.sources.recruitee import RecruiteeAdapter
    from app.sources.workable import WorkableAdapter

    ashby = AshbyAdapter(CFG, _Http({"ashbyhq": {"jobs": [
        {"id": "u1", "title": "ML Engineer", "location": "Berlin", "isListed": True, "descriptionPlain": "Build models",
         "address": {"postalAddress": {"addressCountry": "Germany"}}, "jobUrl": "https://jobs.ashbyhq.com/acme/u1",
         "publishedAt": "2026-10-01T10:00:00Z"},
        {"id": "u2", "title": "Hidden", "isListed": False}]}}))
    raws = ashby.fetch_jobs()
    p = ashby.normalize_job(raws[0])
    assert len(raws) == 1 and p.location == "Berlin, Germany" and p.description == "Build models" and p.date_posted
    work = WorkableAdapter(CFG, _Http({}))
    p = work.normalize_job({"shortcode": "AB12", "title": "Backend Engineer", "url": "https://apply.workable.com/acme/j/AB12",
                            "locations": [{"city": "Riyadh", "country": "Saudi Arabia"}], "description": "<p>Go</p>",
                            "published_on": "2026-10-02"})
    assert p.external_job_id == "AB12" and p.location == "Riyadh, Saudi Arabia" and p.description == "Go"
    rec = RecruiteeAdapter(CompanyFeedConfig(company_name="Acme", options={"host": "careers.acme.example"}), _Http({}))
    assert rec._base() == "https://careers.acme.example"
    p = rec.normalize_job({"id": 7, "title": "Data Engineer", "careers_url": "https://x/o/data", "locations": [
        {"city": "Luxembourg", "country": "Luxembourg"}], "description": "<p>Data</p>", "requirements": "<p>SQL</p>"})
    assert p.location == "Luxembourg, Luxembourg" and p.requirements == "SQL"


def test_workday_searches_target_countries_and_reads_details():
    from app.sources.workday import WorkdayAdapter, parse_identifier

    assert parse_identifier("nvidia.wd5/NVIDIAExternalCareerSite") == ("nvidia.wd5.myworkdayjobs.com", "nvidia", "NVIDIAExternalCareerSite")
    http = _Http({
        "/jobs": lambda payload: {"total": 1, "jobPostings": [
            {"title": f"Engineer {payload['searchText']}", "externalPath": f"/job/{payload['searchText']}-City/Engineer_JR{len(payload['searchText'])}"}]},
        "/job/": {"jobPostingInfo": {"title": "Engineer", "jobDescription": "<p>CUDA</p>", "location": "Germany, Munich",
                                     "country": {"descriptor": "Germany"}, "startDate": "2026-10-01",
                                     "externalUrl": "https://nvidia.wd5.myworkdayjobs.com/x"}},
    })
    adapter = WorkdayAdapter(CompanyFeedConfig(company_name="NVIDIA", identifier="nvidia.wd5/Site",
                                               options={"search": ["Germany", "Luxembourg"]}), http)
    raws = adapter.fetch_jobs()
    assert [r["externalPath"].split("/")[2] for r in raws] == ["Germany-City", "Luxembourg-City"]
    assert adapter.location_hint(raws[0]) == "Germany City"
    p = adapter.normalize_job(adapter.fetch_details(raws[0]))
    assert p.description == "CUDA" and "Germany" in p.location and p.date_posted is not None


def test_amazon_adapter_filters_by_country_and_category():
    from app.sources.amazon import AmazonJobsAdapter

    http = _Http({"amazon.jobs": {"hits": 1, "jobs": [{"id_icims": "1058", "title": "SDE", "job_path": "/en/jobs/1058/sde",
                                                         "normalized_location": "Luxembourg, LUX", "posted_date": "October 19, 2026",
                                                         "description": "Build", "basic_qualifications": "Java"}]}})
    adapter = AmazonJobsAdapter(CompanyFeedConfig(company_name="Amazon", options={"countries": ["LUX"]}), http)
    raws = adapter.fetch_jobs()
    params = dict((k, v) for k, v in http.calls[0][1] if k != "category[]")
    assert params["normalized_country_code[]"] == "LUX" and any(k == "category[]" for k, _ in http.calls[0][1])
    p = adapter.normalize_job(raws[0])
    assert p.url == "https://www.amazon.jobs/en/jobs/1058/sde" and p.requirements == "Java" and p.date_posted.day == 19


def test_detect_new_ats_urls():
    assert detect_ats("https://jobs.ashbyhq.com/deepl/5d5e").ats_type == "ashby"
    assert detect_ats("https://apply.workable.com/foodics/j/AB12/").identifier == "foodics"
    assert detect_ats("https://instagrid.recruitee.com/o/x").ats_type == "recruitee"
    wd = detect_ats("https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/x")
    assert (wd.ats_type, wd.identifier) == ("workday", "nvidia.wd5/NVIDIAExternalCareerSite")


def test_country_gate_keeps_targets_and_unknown_drops_foreign():
    from app.services.companies import country_gate

    keep = country_gate(CompanyFeedConfig(company_name="X"))
    assert keep("Munich, Germany") and keep("Remote") and keep("") and keep("London; Dubai")
    assert keep("Esch-sur-Alzette") and not keep("San Francisco, United States") and not keep("Paris")
    assert country_gate(CompanyFeedConfig(company_name="X", options={"all_countries": True}))("Paris")


def test_getro_board_pages_and_uses_the_portfolio_company():
    from app.sources.getro import GetroAdapter

    pages = {0: [{"id": i, "title": f"Engineer {i}", "url": f"https://acme.example/{i}", "locations": ["Dubai - United Arab Emirates"],
                  "organization": {"name": "Tabby"}, "skills": ["Python"], "seniority": "mid", "work_mode": "hybrid",
                  "created_at": 1792499935} for i in range(20)], 1: [{"id": 99, "title": "Data Scientist", "locations": ["Riyadh"],
                                                                      "organization": {"name": "Tamara"}}]}
    http = _Http({"api.getro.com": lambda payload: {"results": {"jobs": pages.get(payload["page"], [])}}})
    adapter = GetroAdapter(CompanyFeedConfig(company_name="Hub71 jobs", identifier="9266"), http)
    raws = adapter.fetch_jobs()
    assert len(raws) == 21 and len(http.calls) == 3
    p = adapter.normalize_job(raws[0])
    assert p.company == "Tabby" and p.remote_status == "hybrid" and "Python" in p.description and p.date_posted
    assert adapter.location_hint(raws[-1]) == "Riyadh"
