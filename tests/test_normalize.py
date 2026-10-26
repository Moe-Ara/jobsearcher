from app.core.normalize import (
    canonical_seniority,
    canonicalize_url,
    extract_language_requirements,
    extract_years_experience,
    infer_country,
    infer_remote_status,
    normalize_company,
    normalize_title,
)
from app.core.text import clean_text, html_to_text


def test_title_normalisation_handles_gender_tags_and_synonyms():
    assert normalize_title("AI Engineer (m/w/d)") == normalize_title("Artificial Intelligence Engineer (f/m/x)")
    assert normalize_title("Sr. ML Engineer – Remote") == "senior machine learning engineer"
    assert "cplusplus" in normalize_title("C++ Developer (all genders)")
    assert normalize_title("Softwareentwickler (gn)") == "software developer"


def test_company_normalisation_strips_legal_forms():
    assert normalize_company("Robert Bosch GmbH") == "robert bosch"
    assert normalize_company("Siemens AG") == "siemens"
    assert normalize_company("Example GmbH & Co. KG") == "example"
    assert normalize_company("Müller Technologies SE") == "muller technologies"


def test_canonical_urls():
    assert canonicalize_url("https://www.linkedin.com/comm/jobs/view/3912345678/?trackingId=abc&refId=x") == "https://www.linkedin.com/jobs/view/3912345678"
    assert canonicalize_url("https://de.indeed.com/rc/clk?jk=abc123&from=ja") == "https://de.indeed.com/viewjob?jk=abc123"
    assert canonicalize_url("https://Jobs.Lever.co/acme/123/?lever-source=LinkedIn&utm_source=x#apply") == "https://jobs.lever.co/acme/123"
    assert canonicalize_url("javascript:alert(1)") is None


def test_country_inference():
    assert infer_country("Stuttgart, Baden-Württemberg") == "Germany"
    assert infer_country("München") == "Germany"
    assert infer_country("Riyadh") == "Saudi Arabia"
    assert infer_country("Dubai, United Arab Emirates") == "UAE"
    assert infer_country("Berlin, DE") == "Germany"
    assert infer_country("Seattle, WA, United States") == "United States"
    assert infer_country("", "de") == "Germany"
    assert infer_country("Remote") is None


def test_remote_inference():
    assert infer_remote_status("Engineer", "Munich Hybrid") == "hybrid"
    assert infer_remote_status("Engineer (Remote)", "") == "remote"
    assert infer_remote_status("Engineer", "Berlin", "We offer 2 days home office per week") == "hybrid"
    assert infer_remote_status("Engineer", "Berlin", "remote sensing data") is None


def test_seniority_and_experience():
    assert canonical_seniority("Senior Software Engineer") == "Senior"
    assert canonical_seniority("Software Engineer", "Mid-Senior Level") == "Senior"
    assert canonical_seniority("Software Engineer", "Entry level") == "Junior"
    assert extract_years_experience("You have 3+ years of experience in Python and 5 years of professional experience in C++") == 3
    assert extract_years_experience("mindestens 2 Jahre Berufserfahrung") == 2
    assert extract_years_experience("Founded 25 years ago") is None


def test_language_requirements():
    langs = extract_language_requirements("Fluent English and German (C1) required")
    assert "German (c1)" in langs and any(l.startswith("English") for l in langs)


def test_text_sanitising_removes_invisible_and_scripts():
    assert clean_text("a​b‮c\x00") == "abc"
    text = html_to_text("<p>Hello<script>alert(1)</script></p><ul><li>One</li><li>Two</li></ul>")
    assert "alert" not in text and "• One" in text and "• Two" in text
    assert html_to_text("&lt;p&gt;Escaped &amp;amp; fine&lt;/p&gt;") == "Escaped & fine"


def test_tidy_location():
    from app.services.ingest import tidy_location

    assert tidy_location("Berlin, , Germany") == "Berlin, Germany"
    assert tidy_location(", Munich ,") == "Munich"
