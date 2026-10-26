"""Opening jobs: never send the user through expiring email tracking redirects when there's a better link."""

from types import SimpleNamespace

from app.services.links import is_tracking_url, job_links

CTS = "https://cts.indeed.com/v3/H4sIAAAAAAAA_abc"
STEP = "https://click.stepstone.de/f/a/AbC~~/AAAmIhA~/xyz"


def src(source, url, source_type="job_board"):
    return SimpleNamespace(source=source, url=url, source_type=source_type)


def job(sources, title="Data Engineer (m/w/d)", company="Beispiel GmbH", location="88045 Friedrichshafen"):
    return SimpleNamespace(sources=sources, title=title, company=company, location=location,
                           canonical_url=sources[0].url if sources else None)


def test_tracking_detection():
    assert is_tracking_url(CTS) and is_tracking_url(STEP)
    assert is_tracking_url("https://de.indeed.com/pagead/clk?ad=xyz")
    assert not is_tracking_url("https://jobs.smartrecruiters.com/BoschGroup/744000153751119-x")
    assert not is_tracking_url("https://www.linkedin.com/jobs/view/4476183578")


def test_indeed_email_job_opens_board_search_not_tracking_link():
    links = job_links(job([src("indeed", CTS)]))
    p = links["primary"]
    assert p.label == "Find on Indeed ↗" and p.url.startswith("https://de.indeed.com/jobs?q=")
    assert "Data+Engineer+Beispiel+GmbH" in p.url and "l=Friedrichshafen" in p.url and "m%2Fw%2Fd" not in p.url
    labels = [x.label for x in links["extras"]]
    assert labels == ["Web search", "Email link"] and links["extras"][1].url == CTS


def test_stepstone_truncated_title_search():
    p = job_links(job([src("stepstone", STEP)], title="Softwareentwickler Schwerpunkt Anwend...", location="Konstanz"))["primary"]
    assert p.url == "https://www.stepstone.de/work/?ke=Softwareentwickler+Schwerpunkt+Anwend+Beispiel+GmbH&ws=Konstanz"


def test_direct_employer_link_wins_over_email_link():
    employer = "https://jobs.smartrecruiters.com/BoschGroup/744000153751119-x"
    links = job_links(job([src("indeed", CTS), src("smartrecruiters", employer, "employer")]))
    assert links["primary"].url == employer and links["primary"].label == "Open job ↗"
    assert any(x.label == "Email link" for x in links["extras"])


def test_feed_and_linkedin_jobs_open_directly():
    li = "https://www.linkedin.com/jobs/view/4476183578"
    assert job_links(job([src("linkedin", li)]))["primary"].url == li


def test_button_text_company_is_ignored_in_searches():
    p = job_links(job([src("indeed", CTS)], company="Job anzeigen", location=""))["primary"]
    assert "anzeigen" not in p.url.lower() and p.url.startswith("https://de.indeed.com/jobs?q=Data+Engineer")
