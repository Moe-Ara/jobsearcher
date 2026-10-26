from email.message import EmailMessage as MimeMessage

from app.db import session_scope
from app.email_ingest.parsers.boards import ArbeitsagenturAlertParser, LinkedInAlertParser, StepStoneAlertParser
from app.email_ingest.service import process_eml_bytes
from app.models import EmailMessage, Job

LINKEDIN_HTML = """
<html><body><table>
<tr><td>Your job alert for <b>AI Engineer</b> in Germany</td></tr>
<tr><td>
  <table><tr><td><a href="https://www.linkedin.com/comm/jobs/view/4011111111/?trackingId=a%3D&refId=1"><img alt="Bosch"></a></td>
  <td><a href="https://www.linkedin.com/comm/jobs/view/4011111111/?trackingId=b">AI Engineer (m/w/d)</a>
      <p>Bosch · Stuttgart, Baden-Württemberg, Germany</p><p>Actively recruiting</p><p>3 connections</p></td></tr></table>
</td></tr>
<tr><td>
  <table><tr><td><a href="https://www.linkedin.com/comm/jobs/view/4022222222/?trackingId=c">Robotics Software Engineer</a>
      <p>NEURA Robotics · Metzingen, Germany (Hybrid)</p><p>Easy Apply</p></td></tr></table>
</td></tr>
<tr><td><a href="https://www.linkedin.com/comm/jobs/search/?keywords=AI">See all jobs</a></td></tr>
</table></body></html>
"""

STEPSTONE_HTML = """
<div><a href="https://www.stepstone.de/stellenangebote--MLOps-Engineer-m-w-d-Muenchen-Acme-GmbH--12345678-inline.html?rltr=x">MLOps Engineer (m/w/d)</a>
<div>Acme GmbH</div><div>München</div></div>
<div><a href="https://www.stepstone.de/stellenangebote--Data-Engineer-Berlin-Foo-AG--87654321-inline.html">Data Engineer</a>
<div>Foo AG</div><div>Berlin</div></div>
"""

BA_HTML = """
<p><a href="https://www.arbeitsagentur.de/jobsuche/jobdetail/10000-1199999999-S?utm=x">Softwareentwickler Python (m/w/d)</a><br>
Example Systems GmbH<br>Leipzig</p>
"""


def test_linkedin_parser_extracts_multiple_jobs():
    jobs = LinkedInAlertParser().parse(LINKEDIN_HTML)
    assert [j.external_job_id for j in jobs] == ["4011111111", "4022222222"]
    first = jobs[0]
    assert first.title == "AI Engineer (m/w/d)" and first.company == "Bosch"
    assert first.location.startswith("Stuttgart") and first.url == "https://www.linkedin.com/jobs/view/4011111111"
    assert jobs[1].company == "NEURA Robotics"


def test_stepstone_and_ba_parsers():
    ss = StepStoneAlertParser().parse(STEPSTONE_HTML)
    assert [(j.external_job_id, j.title, j.company, j.location) for j in ss] == [
        ("12345678", "MLOps Engineer (m/w/d)", "Acme GmbH", "München"),
        ("87654321", "Data Engineer", "Foo AG", "Berlin"),
    ]
    ba = ArbeitsagenturAlertParser().parse(BA_HTML)
    assert len(ba) == 1 and ba[0].external_job_id == "10000-1199999999-S" and ba[0].company == "Example Systems GmbH"


def _eml(html: str, sender: str, mid: str) -> bytes:
    msg = MimeMessage()
    msg["From"] = sender
    msg["To"] = "me@example.com"
    msg["Subject"] = "New jobs for you"
    msg["Message-ID"] = mid
    msg.set_content("plain fallback")
    msg.add_alternative(html, subtype="html")
    return msg.as_bytes()


def test_eml_processing_ingests_and_is_idempotent():
    data = _eml(LINKEDIN_HTML, "LinkedIn Job Alerts <jobalerts-noreply@linkedin.com>", "<abc@linkedin.com>")
    preview = process_eml_bytes(data, dry_run=True)
    assert preview["parser"] == "linkedin" and len(preview["jobs"]) == 2
    res = process_eml_bytes(data)
    assert res["found"] == 2 and res["new"] == 2
    assert process_eml_bytes(data)["skipped"] == "already processed"
    with session_scope() as s:
        assert s.query(Job).count() == 2 and s.query(EmailMessage).count() == 1


def test_unknown_sender_is_ignored():
    res = process_eml_bytes(_eml(LINKEDIN_HTML, "newsletter@shop.example", "<n@x>"), dry_run=True)
    assert res["parser"] is None and res["jobs"] == []


# Structure mirrors real 2026 alert emails (opaque tracking links, button texts); content is invented.
def cts(job_key: str | None, target: str = "") -> str:
    """A real-shaped Indeed tracking link: /v3/<base64url(gzip(JSON {"u": destination}))>."""
    import base64
    import gzip
    import json

    url = target or (f"https://de.indeed.com/pagead/clk?from=jobi2a_jobmatch-de-DE_email&jrtk=5-cmh1-1-1k59-{job_key}&mo=r"
                     if job_key else "https://profile.indeed.com/preferences")
    token = base64.urlsafe_b64encode(gzip.compress(json.dumps({"u": url}).encode())).decode().rstrip("=")
    return f"https://cts.indeed.com/v3/{token}"


JK = "e8db2d86423da865"
INDEED_MATCH_HTML = """
<html><head><style>.x{}</style></head><body><!--[if mso]><i>&emsp;</i><![endif]-->
<p>Data Engineer (m/w/d) bei Beispiel Analytics GmbH</p><p>Guten Tag,</p>
<a href="CTS_JOB">Job anzeigen</a> <a href="CTS_OTHER">Passt nicht</a>
<table><tr><td><a href="CTS_JOB">Data Engineer (m/w/d)</a></td></tr>
<tr><td>Beispiel Analytics GmbH</td></tr><tr><td>88045 Friedrichshafen</td></tr><tr><td>Aktiver Arbeitgeber</td></tr>
<tr><td>Gehalt</td></tr><tr><td>55.000 € – 65.000 € pro Jahr</td></tr><tr><td>Vollzeit</td></tr>
<tr><td>Vorschau der Stellenbeschreibung</td></tr><tr><td>Du baust Datenpipelines mit Python.</td></tr></table>
<a href="CTS_JOB">Mehr erfahren</a>
<table><tr><td><a href="CTS_OTHER">Mindestgrundgehalt</a></td></tr><tr><td>50.000 € pro Jahr</td></tr><tr><td>Wurde seit 90 Tagen nicht aktualisiert</td></tr></table>
<p>Möchten Sie mehr Job-Vorschläge wie diesen erhalten?</p><a href="https://cts.indeed.com/v3/H4sIAAAA_n">Nein</a><a href="https://cts.indeed.com/v3/H4sIAAAA_j">Ja</a>
<p>Halten Sie Ihr Indeed-Profil aktuell</p><p>MAX MUSTER</p><a href="https://cts.indeed.com/v3/H4sIAAAA_p">Profil bearbeiten</a>
<a href="https://cts.indeed.com/v3/H4sIAAAA_u">Abbestellen</a></body></html>
""".replace("CTS_JOB", cts(JK)).replace("CTS_OTHER", cts(None))

STEPSTONE_AGENT_HTML = """
<html><body><p>Hallo Max,</p><p>wir haben neue Jobs gefunden, die zu deiner Suche passen</p>
<div><span>Passt hervorragend</span><a href="https://click.stepstone.de/f/a/AAA~~/x1">Softwareentwickler Schwerpunkt Anwend...</a>
<div>Beispiel Systems GmbH</div><div>Konstanz</div><div>51-250</div><div>Mitarbeiter</div><div>Feste Anstellung</div><div>Vollzeit</div></div>
<div><a href="https://click.stepstone.de/f/a/BBB~~/x2">DevOps-Engineer (m/w/d)</a><div>Muster Cloud AG</div><div>bundesweit</div>
<div>1000+</div><div>Mitarbeiter</div><div>Homeoffice möglich, Vollzeit</div><div>55.000 - 75.000 €/Jahr (geschätzt für Vollzeit)</div></div>
<a href="https://click.stepstone.de/f/a/CCC~~/x3">Weitere passende Jobs anzeigen</a>
<a href="https://click.stepstone.de/f/a/DDD~~/x4">Diese E-Mail abbestellen</a><a href="https://click.stepstone.de/f/a/EEE~~/x5">Kontakt</a>
</body></html>
"""


def test_indeed_match_email_real_structure():
    from app.email_ingest.parsers.boards import IndeedAlertParser

    jobs = IndeedAlertParser().parse(INDEED_MATCH_HTML)
    assert [(j.title, j.company, j.location) for j in jobs] == [("Data Engineer (m/w/d)", "Beispiel Analytics GmbH", "88045 Friedrichshafen")]
    assert "Datenpipelines" in jobs[0].description and "Gehalt" not in jobs[0].description
    # the job key from the tracking payload: stable id + canonical URL; the "Mindestgrundgehalt" UI card is no job
    assert jobs[0].external_job_id == JK and jobs[0].url == f"https://de.indeed.com/viewjob?jk={JK}"
    assert (jobs[0].salary_min, jobs[0].salary_max) == (55000, 65000)
    p = IndeedAlertParser()
    assert p.matches("Indeed <donotreply@match.indeed.com>", "Data Engineer (m/w/d) bei Beispiel Analytics GmbH")
    assert not p.matches("indeedapply@indeed.com", "Bewerbung über Indeed: Data Engineer bei X")
    assert not p.matches("login@indeed.com", "Sign in to Indeed with code: 123456")


def test_stepstone_agent_email_real_structure():
    from app.email_ingest.parsers.boards import StepStoneAlertParser

    jobs = StepStoneAlertParser().parse(STEPSTONE_AGENT_HTML)
    assert [(j.title, j.company, j.location) for j in jobs] == [
        ("Softwareentwickler Schwerpunkt Anwend", "Beispiel Systems GmbH", "Konstanz"),
        ("DevOps-Engineer (m/w/d)", "Muster Cloud AG", "bundesweit"),
    ]
    assert not StepStoneAlertParser().matches("info@jobagent.stepstone.de", "Ihre Bewerbung bei X wurde angesehen")


def test_gulf_board_alerts_bayt_gulftalent_naukrigulf():
    from app.email_ingest.parsers.boards import BaytAlertParser, GulfTalentAlertParser, NaukriGulfAlertParser

    bayt = """<table><tr><td><a href="https://www.bayt.com/en/uae/jobs/software-engineer-automated-trading-dubai-uae-74414826/?utm_source=alert">
      Software Engineer Automated Trading</a><div>Cobblestone Energy</div><div>Dubai, UAE</div></td></tr>
      <tr><td><a href="https://www.bayt.com/en/saudi-arabia/jobs/backend-developer-riyadh-74500001/">Backend Developer</a>
      <div>Salla</div><div>Riyadh, Saudi Arabia</div></td></tr></table>"""
    jobs = BaytAlertParser().parse(bayt)
    assert [(j.external_job_id, j.title, j.company) for j in jobs] == [
        ("74414826", "Software Engineer Automated Trading", "Cobblestone Energy"), ("74500001", "Backend Developer", "Salla")]
    assert jobs[0].url == "https://www.bayt.com/en/uae/jobs/software-engineer-automated-trading-dubai-uae-74414826/"
    gt = """<div><a href="https://www.gulftalent.com/uae/jobs/title/software-engineer">see all</a>
      <p><a href="https://www.gulftalent.com/uae/jobs/senior-software-engineer-java-full-stack-577125?src=alert">Senior Software Engineer Java Full Stack</a></p>
      <p>Emirates Group</p><p>Dubai</p></div>"""
    jobs = GulfTalentAlertParser().parse(gt)
    assert [(j.external_job_id, j.company) for j in jobs] == [("577125", "Emirates Group")]
    assert jobs[0].url == "https://www.gulftalent.com/uae/jobs/senior-software-engineer-java-full-stack-577125"
    ng = """<div><a href="https://www.naukrigulf.com/ai-engineer-jobs-in-abu-dhabi-uae-in-g42-3-to-6-years-n-cd-1000-jid-140825500123">AI Engineer</a>
      <span>G42</span><span>Abu Dhabi - UAE</span></div>"""
    jobs = NaukriGulfAlertParser().parse(ng)
    assert jobs[0].external_job_id == "140825500123" and jobs[0].title == "AI Engineer"
    assert BaytAlertParser().matches("Bayt.com <alerts@bayt.com>", "New jobs for you")
