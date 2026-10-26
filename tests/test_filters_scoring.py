from types import SimpleNamespace

from app.config_files import get_prefs
from app.core.normalize import normalize_company
from app.enums import FilterStatus, Recommendation
from app.filters import apply_hard_filters
from app.scoring import compute_score


def job(**kw):
    base = dict(
        title="Software Engineer", company="Acme", location="Berlin", country="Germany", remote_status=None,
        description="Build backend services in Python.", requirements="", responsibilities="", sector=None,
        years_experience_required=None,
    )
    base.update(kw)
    base["normalized_company"] = normalize_company(base["company"])
    return SimpleNamespace(**base)


def test_clean_job_passes(prefs):
    assert apply_hard_filters(job(), prefs).status == FilterStatus.PASSED


def test_pure_defence_company_rejected(prefs):
    out = apply_hard_filters(job(company="Hensoldt Sensors GmbH"), prefs)
    assert out.status == FilterStatus.REJECTED and out.reasons[0]["rule"] == "defence"


def test_defence_wording_rejected(prefs):
    out = apply_hard_filters(job(description="You will develop software for military vehicles and weapon systems."), prefs)
    assert out.rejected


def test_mixed_company_is_flagged_not_rejected(prefs):
    out = apply_hard_filters(job(company="Airbus Operations GmbH", description="Commercial aircraft cabin software."), prefs)
    assert out.status == FilterStatus.FLAGGED


def test_cyber_defence_is_not_military(prefs):
    out = apply_hard_filters(job(title="Security Automation Engineer", description="Join our cyber defense team; defense-in-depth, SOAR, SIEM."), prefs)
    assert out.status == FilterStatus.PASSED


def test_single_defence_mention_only_flags(prefs):
    out = apply_hard_filters(job(description="Our sensors serve automotive, medical and defence customers worldwide."), prefs)
    assert out.status in (FilterStatus.FLAGGED, FilterStatus.REJECTED)
    # "defence customers" is a very-strong phrase → reject; plain single keyword → flag
    out2 = apply_hard_filters(job(description="We collaborate with partners incl. NATO research labs on open standards."), prefs)
    assert out2.status == FilterStatus.FLAGGED


def test_wrong_country_rejected_but_remote_flagged(prefs):
    assert apply_hard_filters(job(location="Paris", country="France"), prefs).rejected
    assert apply_hard_filters(job(location="Remote", country="Netherlands", remote_status="remote"), prefs).rejected  # remote abroad: rejected
    unknown = apply_hard_filters(job(location="Talent Pool", country=None), prefs)
    assert unknown.status == FilterStatus.FLAGGED and unknown.reasons[0]["rule"] == "country_unknown"


def test_seniority_profession_experience(prefs):
    assert apply_hard_filters(job(title="Werkstudent Software (m/w/d)"), prefs).rejected
    assert apply_hard_filters(job(title="Head of Engineering"), prefs).rejected
    assert apply_hard_filters(job(title="Sales Manager DACH"), prefs).rejected
    assert not apply_hard_filters(job(title="Sales Engineer – AI Platform"), prefs).rejected  # relevant word rescues
    assert apply_hard_filters(job(years_experience_required=12), prefs).rejected


def test_missing_one_technology_is_not_a_hard_filter(prefs):
    assert not apply_hard_filters(job(requirements="Must have Azure, Python, Kafka"), prefs).rejected


ANALYSIS = {
    "technical_match": 9, "experience_match": 8, "career_match": 9, "location_match": 9,
    "role_family": "Industrial AI", "sector": "Industrial manufacturing", "seniority": "Mid",
    "critical_gaps": [], "overall_score": 9.5,
}


def score(**kw):
    args = dict(filter_status="PASSED", analysis=ANALYSIS, country="Germany", location="Stuttgart",
                remote_status="hybrid", seniority="Mid", best_source_type="employer", target_priority=None)
    args.update(kw)
    return compute_score(prefs=get_prefs(), **args)


def test_scoring_thresholds_and_weights():
    high = score()
    assert high.recommendation == Recommendation.PRIORITY and high.score >= 8
    assert set(high.breakdown["components"]) == {"technical", "experience", "role_family", "sector", "location", "career", "source"}
    weak = score(analysis={**ANALYSIS, "technical_match": 3, "experience_match": 4, "career_match": 4, "role_family": "Other", "sector": "Retail"})
    assert weak.recommendation == Recommendation.LOW_PRIORITY


def test_scoring_does_not_trust_llm_overall():
    a = score(analysis={**ANALYSIS, "overall_score": 1.0})
    b = score(analysis={**ANALYSIS, "overall_score": 10.0})
    assert a.score == b.score


def test_scoring_gaps_bonus_and_rejection():
    base = score().score
    assert score(analysis={**ANALYSIS, "critical_gaps": ["PhD", "10y C#"]}).score < base
    assert score(target_priority="A").score >= base
    assert score(filter_status="REJECTED").recommendation == Recommendation.REJECTED
    assert score(analysis=None).recommendation == Recommendation.PENDING


def test_german_compound_titles(prefs):
    for title in ["Pflichtpraktikum im Bereich Fertigungsplanung", "Werkstudententätigkeit KI", "Abschlussarbeit Robotik",
                  "Flächenmanager im Facility Management (w/m/div.)", "Duales Studium Informatik"]:
        assert apply_hard_filters(job(title=title), prefs).rejected, title
    assert not apply_hard_filters(job(title="Enterprise Architect (m/w/d)"), prefs).rejected


def test_german_compound_professions(prefs):
    for title in ["Fachkraft für Lagerlogistik / stellv. Lagerleitung (w,m, div.)", "Rechtsreferendar (w/m/div.) - Wahlstation",
                  "Projektmanager technischer Systemvertrieb Region Stuttgart", "Mechatroniker Instandhaltung"]:
        assert apply_hard_filters(job(title=title), prefs).rejected, title
    for title in ["Softwareentwickler Embedded (m/w/d)", "Fachinformatiker Anwendungsentwicklung", "Data Engineer Vertriebsanalytik"]:
        assert not apply_hard_filters(job(title=title), prefs).rejected, title


def test_real_world_location_strings():
    from app.core.normalize import infer_country, infer_country_from_title

    cases = {
        "Redwood City, US, California": "United States", "Remotely in the USA": "United States", "Dallas": "United States",
        "Remote, US, Texas": "United States", "Gurugram": "India", "Chennai": "India", "Bangalore, IN": "India",
        "Kuala Lumpur": "Malaysia", "Nantes": "France", "Sofia": "Bulgaria", "Belgrade Office": "Serbia",
        "66386 Sankt Ingbert": "Germany", "Konstanz": "Germany", "Riyadh, SA": "Saudi Arabia", "Dubai": "UAE",
    }
    for loc, country in cases.items():
        assert infer_country(loc) == country, loc
    assert infer_country("Talent Pool") is None
    assert infer_country_from_title("Data Scientist - Remote Malaysia (m/f/d)") == "Malaysia"
    assert infer_country_from_title("German speaking Support Engineer") is None


def test_unrealistic_seniority_and_years(prefs):
    assert apply_hard_filters(job(title="Principal Software Engineer"), prefs).rejected
    assert apply_hard_filters(job(title="Staff Machine Learning Engineer"), prefs).rejected
    assert not apply_hard_filters(job(title="Senior Software Engineer"), prefs).rejected  # kept, but penalised
    assert apply_hard_filters(job(years_experience_required=8), prefs).rejected
    assert not apply_hard_filters(job(years_experience_required=5), prefs).rejected


def test_senior_and_years_penalties_keep_strong_senior_matches_out_of_priority():
    mid = score(seniority="Mid", candidate_years=2.9)
    senior = score(seniority="Senior", candidate_years=2.9)
    senior_5y = score(seniority="Senior", candidate_years=2.9, years_required=5)
    assert mid.recommendation == "PRIORITY"
    assert senior.score <= mid.score - 1.5 and senior.recommendation != "PRIORITY"
    assert senior_5y.score < senior.score and senior_5y.breakdown["years_penalty"] > 0
    assert score(seniority="Mid", candidate_years=2.9, years_required=3).breakdown["years_penalty"] == 0  # within tolerance
    # the job's own (title) seniority wins over the model's reading
    assert score(seniority="Senior", analysis={**ANALYSIS, "seniority": "Mid"}).breakdown["seniority"] == "Senior"
