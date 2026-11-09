"""Replies from employers: rejections / invitations / confirmations are recognised, matched to the application and
move the job along — never backwards and never over a status you set later. Plus the alert parsers added with it."""

from datetime import datetime, timedelta
from email.message import EmailMessage as Mime

from sqlalchemy import select

from app.db import session_scope
from app.email_ingest import replies as R
from app.email_ingest.parsers import boards as B
from app.models import ApplicationPackage, EmailMessage, Job, StatusEvent, utcnow


# --------------------------------------------------------------------------- classification

def test_rejections_in_german_and_english_are_recognised():
    assert R.classify("Ihre Bewerbung", "Sehr geehrter Herr Beispiel,\nleider müssen wir Ihnen mitteilen, dass wir Sie im "
                      "weiteren Auswahlverfahren nicht berücksichtigen können.") == "rejection"
    assert R.classify("Deine Bewerbung", "nach Durchsicht aller Unterlagen haben wir uns leider für jemanden entschieden, "
                      "der noch besser passt.") == "rejection"
    assert R.classify("AW: Bewerbung", "Nach sorgfältiger Abwägung haben wir uns entschieden, den Auswahlprozess mit "
                      "anderen Kandidat*innen fortzusetzen.") == "rejection"
    assert R.classify("Your application", "Unfortunately, we have decided to move forward with other candidates.") == "rejection"
    assert R.classify("Application Update", "We will not be moving forward with your candidacy.") == "rejection"
    assert R.classify("Absage auf deine Bewerbung als Entwickler", "Hallo,") == "rejection"
    # a line break in the middle of the sentence (HTML mail) changes nothing
    assert R.classify("Ihre Bewerbung", "Gesamtprofil haben wir uns, sodass wir Ihnen hiermit leider\nabsagen müssen.") == "rejection"


def test_a_receipt_that_only_describes_what_might_happen_is_no_rejection_or_invitation():
    receipt = ("Vielen Dank für Ihre Bewerbung. Sollten Sie innerhalb von vier Wochen nichts von uns hören, haben wir uns "
               "für andere Bewerber entschieden. Solange Sie keine Absage erhalten, bleibt Ihre Bewerbung im Verfahren.")
    assert R.classify("Eingangsbestätigung", receipt) == "received"
    assert R.classify("Application received", "Thank you for applying. If we would like to invite you to an interview, "
                      "we will contact you promptly.") == "received"
    privacy = "Ihre Daten werden nach Ablauf des Verfahrens und einer erteilten Absage gelöscht."
    assert R.classify("Eingang Ihrer Bewerbung", "Wir haben Ihre Bewerbung erhalten. " + privacy) == "received"


def test_invitations_next_steps_and_board_marketing():
    assert R.classify("Ihre Bewerbung als IT-Spezialist", "wir möchten Sie gerne persönlich kennenlernen und laden Sie zu "
                      "einem Vorstellungsgespräch am 02.11. um 15:30 Uhr ein.") == "interview"
    assert R.classify("Interview for Software Engineer at Beispiel GmbH", "When: Tuesday 1:15 PM") == "interview"
    assert R.classify("Ihre Bewerbung", "Wir möchten den Bewerbungsprozess mit Ihnen gerne fortführen und benötigen die "
                      "nachfolgenden Unterlagen von Ihnen: Zeugnisse.") == "next_step"
    assert R.classify("Du hast gute Chancen auf ein Interview für diesen Job", "Bewirb dich jetzt") == "other"
    assert R.classify("Your application was sent: Embedded Developer", "We hope that Acme invites you to interview soon.") == "received"
    # your own application text quoted below a reply decides nothing
    quoted = ("Wir sichten derzeit die Bewerbungen und melden uns.\nKontaktformular schrieb am 05.01.2026 15:45:\n"
              "Über eine Einladung zu einem persönlichen Gespräch freue ich mich sehr.")
    assert R.classify("Re: Bewerbung als Aushilfe", quoted) == "other"


# --------------------------------------------------------------------------- matching and effects

def _job(title, company, status="APPLIED", **kw):
    with session_scope() as s:
        job = Job(title=title, company=company, source="manual", status=status, status_changed_at=utcnow() - timedelta(days=10),
                  normalized_company=company.lower(), normalized_title=title.lower(), **kw)
        s.add(job)
        s.flush()
        return job.id


def _mail(subject, body, sender="Recruiting <jobs@nordlicht-mobility.example>", **headers):
    m = Mime()
    m["From"], m["Subject"], m["Message-ID"] = sender, subject, headers.pop("mid", f"<{abs(hash(subject + body))}@x>")
    m["Date"] = headers.pop("date", "Mon, 02 Nov 2026 10:00:00 +0100")
    for k, v in headers.items():
        m[k.replace("_", "-")] = v
    m.set_content(body)
    return m


def _process(msg):
    with session_scope() as s:
        matcher = R.ReplyMatcher(s)
    return R.process_reply(msg, matcher)


def test_a_rejection_replying_to_the_email_you_sent_rejects_that_job():
    jid = _job("Software Engineer (m/w/d)", "Nordlicht Mobility GmbH")
    with session_scope() as s:
        s.add(ApplicationPackage(job_id=jid, status="done", language="de",
                                 meta={"submission": {"channel": "email", "to": "jobs@nordlicht-mobility.example",
                                                      "message_id": "<sent-1@web.de>"}}))
    res = _process(_mail("AW: Bewerbung um die Stelle als Software Engineer",
                         "leider müssen wir Ihnen mitteilen, dass wir uns für einen anderen Bewerber entschieden haben.",
                         sender="Personal <hr@other-domain.example>", In_Reply_To="<sent-1@web.de>"))
    assert res["kind"] == "rejection" and res["job_id"] == jid and res["effect"] == "APPLIED → REJECTED"
    with session_scope() as s:
        job = s.get(Job, jid)
        assert job.status == "REJECTED"
        ev = s.scalars(select(StatusEvent).where(StatusEvent.job_id == jid)).all()[-1]
        assert "rejection" in ev.note and "AW: Bewerbung" in ev.note
        row = s.scalars(select(EmailMessage)).one()
        assert row.reply_kind == "rejection" and row.job_id == jid and row.parser is None


def test_an_ats_reply_is_matched_by_the_company_in_the_sender_and_the_title():
    a = _job("Backend Developer (m/w/d)", "Beispiel Robotics AG")
    b = _job("Data Engineer (m/w/d)", "Beispiel Robotics AG")
    other = _job("Backend Developer (m/w/d)", "Nordlicht Mobility GmbH")
    res = _process(_mail("Deine Bewerbung als Data Engineer (m/w/d)",
                         "Hallo, leider können wir dich nicht weiter berücksichtigen.",
                         sender="Beispiel Robotics AG <beispiel-robotics-jobs@m.personio.de>"))
    assert res["job_id"] == b and res["effect"] == "APPLIED → REJECTED"
    with session_scope() as s:
        assert s.get(Job, a).status == "APPLIED" and s.get(Job, other).status == "APPLIED"


def test_two_applications_at_one_company_and_no_title_stays_unlinked_with_both_offered():
    a = _job("Backend Developer (m/w/d)", "Beispiel Robotics AG")
    b = _job("Data Engineer (m/w/d)", "Beispiel Robotics AG")
    res = _process(_mail("Ihre Bewerbung", "leider müssen wir Ihnen absagen.",
                         sender="Beispiel Robotics AG <noreply@beispiel-robotics.example>"))
    assert res["job_id"] is None and set(res["candidates"]) == {a, b}
    with session_scope() as s:
        assert {s.get(Job, a).status, s.get(Job, b).status} == {"APPLIED"}
        assert s.scalars(select(EmailMessage)).one().match_note.startswith("possible jobs:")


def test_your_later_status_wins_over_an_older_email():
    jid = _job("Software Engineer (m/w/d)", "Nordlicht Mobility GmbH", status="INTERVIEW")
    with session_scope() as s:
        s.get(Job, jid).status_changed_at = datetime(2026, 11, 5)  # you set Interview after the e-mail arrived
    res = _process(_mail("Ihre Bewerbung als Software Engineer", "leider müssen wir Ihnen absagen.",
                         date="Mon, 02 Nov 2026 10:00:00 +0100"))
    assert res["kind"] == "rejection" and res["job_id"] == jid and res["effect"] is None
    with session_scope() as s:
        assert s.get(Job, jid).status == "INTERVIEW"


def test_confirmation_turns_applying_into_applied_and_an_invitation_into_interview():
    jid = _job("Software Engineer (m/w/d)", "Nordlicht Mobility GmbH", status="APPLYING")
    assert _process(_mail("Eingangsbestätigung Ihrer Bewerbung als Software Engineer",
                          "wir haben Ihre Bewerbung erhalten."))["effect"] == "APPLYING → APPLIED"
    assert _process(_mail("Einladung zum Vorstellungsgespräch – Software Engineer", "Wir laden Sie herzlich zu einem "
                          "Gespräch ein.", date="Tue, 03 Nov 2026 10:00:00 +0100"))["effect"] == "APPLIED → INTERVIEW"
    with session_scope() as s:
        assert s.get(Job, jid).status == "INTERVIEW"


def test_a_later_email_moves_a_job_the_app_moved_itself_even_when_read_in_one_batch():
    jid = _job("Software Engineer (m/w/d)", "Nordlicht Mobility GmbH")
    with session_scope() as s:
        matcher = R.ReplyMatcher(s)
    R.process_reply(_mail("Einladung zum Online Assessment", "Wir laden Sie zu einem Online-Gespräch ein.",
                          date="Mon, 02 Nov 2026 10:00:00 +0100"), matcher)
    res = R.process_reply(_mail("Ihre Bewerbung als Software Engineer", "leider müssen wir Ihnen absagen.",
                                date="Thu, 05 Nov 2026 10:00:00 +0100"), matcher)
    assert res["effect"] == "INTERVIEW → REJECTED"
    with session_scope() as s:
        assert s.get(Job, jid).status == "REJECTED"


def test_a_job_you_never_applied_to_needs_company_and_title_and_is_never_rejected():
    jid = _job("Data Engineer (m/w/d)", "Nordlicht Mobility GmbH", status="PRIORITY")
    res = _process(_mail("Ihre Bewerbung", "leider müssen wir Ihnen absagen."))
    assert res["job_id"] is None
    res = _process(_mail("Deine Bewerbung als Data Engineer ist eingegangen", "Vielen Dank für deine Bewerbung als Data "
                         "Engineer bei Nordlicht Mobility.", sender="Nordlicht Mobility <jobs@nordlicht-mobility.example>"))
    assert res["kind"] == "received" and res["job_id"] == jid and res["effect"] == "PRIORITY → APPLIED"


def test_the_position_named_in_the_email_decides_between_and_against_applications():
    a = _job("Software Developer (m/w/d) Fullstack Angular & Go", "Nordstadt Digital GmbH")
    b = _job("Fullstack Software Developer (m/w/d) Typescript & Angular", "Nordstadt Digital GmbH")
    res = _process(_mail("Deine Bewerbung - Fullstack Software Developer (m/w/d) Typescript & Angular",
                         "leider sehen wir derzeit keine Möglichkeit.", sender="hr@nordstadt-digital.example"))
    assert res["job_id"] == b
    # one application at a big employer, but the rejection names another position: offered, never applied by itself
    c = _job("Software-Ingenieur Full-Stack (m/w/d)", "Forschungsinstitut Nordstadt", canonical_url="https://jobs.fi-nordstadt.example/1")
    res = _process(_mail("Ihre Bewerbung als Wissenschaftliche Mitarbeiterin für intelligente Sensorik",
                         "Leider müssen wir Ihnen mitteilen, dass wir Ihre Bewerbung nicht in die engere Auswahl einbezogen haben.",
                         sender="Forschungsinstitut <recruiting@fi-nordstadt.example>"))
    assert res["job_id"] is None and c in res["candidates"]
    with session_scope() as s:
        assert s.get(Job, a).status == "APPLIED" and s.get(Job, c).status == "APPLIED"


def test_common_words_in_a_company_name_never_match_on_their_own():
    jid = _job("Softwareentwickler Java (m/w/d)", "QRT Information Products AG")
    res = _process(_mail("Aktuelle Information zu Ihrer Bewerbung", "leider müssen wir Ihnen absagen.",
                         sender="bewerbung@other-employer.example"))
    assert res["job_id"] is None
    with session_scope() as s:
        assert s.get(Job, jid).status == "APPLIED"


def test_a_rejection_for_a_job_you_applied_to_outside_the_app_needs_company_and_title():
    jid = _job("Data Engineer (m/w/d)", "Nordlicht Mobility GmbH", status="REVIEW")
    res = _process(_mail("Ihre Bewerbung als Data Engineer (m/w/d)", "leider können wir Sie nicht berücksichtigen.",
                         sender="Nordlicht Mobility <jobs@nordlicht-mobility.example>"))
    assert res["job_id"] == jid and res["effect"] == "REVIEW → REJECTED"


def test_poll_reads_alerts_and_replies_in_one_pass_and_each_once(monkeypatch):
    from app.email_ingest import service
    from app.settings import Settings
    from tests.imap_fake import fake_imap

    jid = _job("Software Engineer (m/w/d)", "Nordlicht Mobility GmbH")
    alert = Mime()
    alert["From"], alert["Subject"], alert["Message-ID"] = "jobalerts-noreply@linkedin.com", "Network Security Engineer at Acme", "<al@x>"
    alert.set_content("x")
    alert.add_alternative('<a href="https://www.linkedin.com/comm/jobs/view/4000000009/">Network Security Engineer</a>'
                          '<p>Acme · Berlin</p>', subtype="html")
    reply = _mail("Ihre Bewerbung als Software Engineer", "leider müssen wir Ihnen absagen.", mid="<rej@x>")
    own = _mail("Bewerbung um die Stelle als Tester", "anbei meine Bewerbung", sender="Me <me@web.de>", mid="<own@x>")
    news = _mail("Unser Newsletter", "Angebote der Woche", sender="shop@example.com", mid="<n@x>")
    mail = {"INBOX": {b"1": alert.as_bytes(), b"2": reply.as_bytes(), b"3": own.as_bytes(), b"4": news.as_bytes()},
            "JobHunting": {b"9": reply.as_bytes()}}
    FakeIMAP = fake_imap(mail)
    monkeypatch.setattr(service.imaplib, "IMAP4_SSL", FakeIMAP)
    cfg = Settings(_env_file=None, imap_host="imap.web.de", imap_user="me@web.de", imap_password="pw", imap_folder="*")
    monkeypatch.setattr(service, "get_settings", lambda: cfg)
    monkeypatch.setattr("app.settings.get_settings", lambda: cfg)
    res = service.poll_mailbox()
    assert res["messages"] == 1 and res["new"] == 1 and res["replies"] == 1 and res["rejections"] == 1
    assert res["status_changes"] == 1 and res["errors"] == 0
    assert sorted(FakeIMAP.fetched) == [b"1", b"2"]  # never your own mail, a newsletter or the copy in another folder
    with session_scope() as s:
        assert s.get(Job, jid).status == "REJECTED"
    FakeIMAP.fetched.clear()
    assert service.poll_mailbox()["replies"] == 0 and FakeIMAP.fetched == []  # read once


def test_replies_page_and_assigning_a_reply(client=None):
    from fastapi.testclient import TestClient

    from app.main import app

    jid = _job("Data Engineer (m/w/d)", "Beispiel Robotics AG")
    _job("Backend Developer (m/w/d)", "Beispiel Robotics AG")
    _process(_mail("Ihre Bewerbung", "leider müssen wir Ihnen absagen.",
                   sender="Beispiel Robotics AG <noreply@beispiel-robotics.example>"))
    c = TestClient(app)
    page = c.get("/replies?show=unmatched")
    assert page.status_code == 200 and "Which application is it?" in page.text and "Data Engineer" in page.text
    with session_scope() as s:
        rid = s.scalars(select(EmailMessage.id)).one()
    r = c.post(f"/replies/{rid}/assign", data={"job_id": str(jid), "show": "all"}, follow_redirects=False)
    assert r.status_code == 303
    with session_scope() as s:
        assert s.get(Job, jid).status == "REJECTED" and s.get(EmailMessage, rid).job_id == jid
    assert "E-mails from the employer" in c.get(f"/jobs/{jid}").text


# --------------------------------------------------------------------------- alert parsers

def test_alert_subjects_with_security_application_or_interview_in_a_title_are_still_alerts():
    parsers = B.all_parsers(["indeed", "stepstone", "linkedin"], [])
    assert B.pick_parser(parsers, "donotreply@match.indeed.com", "Network Security Engineer (m/w/d) bei Acme").name == "indeed"
    assert B.pick_parser(parsers, "donotreply@match.indeed.com", "Web-Applications Developer bei Acme").name == "indeed"
    assert B.pick_parser(parsers, "info@jobagent.stepstone.de", "Du hast gute Chancen auf ein Interview für diesen Job").name == "stepstone"
    assert B.pick_parser(parsers, "jobs-noreply@linkedin.com", "Max, your application was sent to Acme") is None
    assert B.pick_parser(parsers, "info@email.stepstone.de", "Ihre Bewerbung bei Acme wurde angesehen") is None


def test_new_alert_parsers_read_their_cards():
    jobvector = ('<a href="https://www.jobvector.de/jobs/entwickler/?sort=dateStart&jobId=ad01e1ebc2243491&utm_source=x">'
                 '<b>Embedded Softwareentwickler (m/w/d)</b><span>Beispiel Maschinenbau GmbH</span>'
                 '<span>53.603 €/Jahr (geschätzt)</span><span>Damme</span></a>'
                 '<a href="https://www.jobvector.de/myjobvector/recommendations?jobId=b85ffe812089f623&token=SECRET">'
                 '<b>Softwareentwickler - KI (m/w/d)</b><span>Nordlicht Mobility</span><span>Köln</span></a>')
    jobs = B.JobvectorAlertParser().parse(jobvector)
    assert [(j.title, j.company, j.location) for j in jobs] == [
        ("Embedded Softwareentwickler (m/w/d)", "Beispiel Maschinenbau GmbH", "Damme"),
        ("Softwareentwickler - KI (m/w/d)", "Nordlicht Mobility", "Köln")]
    assert all("token" not in j.url and "utm" not in j.url for j in jobs)

    haystack = ('<a href="https://haystackapp.io/go?j=232efaf5-6720-4720-81a6-036a83a99d35&sub=me">Senior Python Engineer</a>'
                '<a href="https://haystackapp.io/go?j=232efaf5-6720-4720-81a6-036a83a99d35&sub=me">Acme · Hybrid · Berlin, Germany</a>')
    (h,) = B.HaystackAlertParser().parse(haystack)
    assert (h.title, h.company, h.location, h.url) == ("Senior Python Engineer", "Acme", "Berlin, Germany",
                                                       "https://haystackapp.io/go?j=232efaf5-6720-4720-81a6-036a83a99d35")

    workwise = ('<a href="https://www.workwise.io/jobsuche?eTT=PERSONAL.TOKEN"><b>React Entwickler (m/w/d)</b><span>Acme GmbH</span>'
                '<span>Mannheim +1</span><span>40 Stunden</span></a><a href="https://www.workwise.io/jobsuche?eTT=X">Alle passenden Jobs anschauen</a>')
    (w,) = B.WorkwiseAlertParser().parse(workwise)
    assert (w.title, w.company, w.location, w.url) == ("React Entwickler (m/w/d)", "Acme GmbH", "Mannheim", None)

    company = B.CompanyAlertParser()
    company.message_meta = {"subject": "Neue Stellenausschreibung(en) an der Universität Nordstadt – Ihre Suche", "from_name": ""}
    (c,) = company.parse('<a href="http://jobs.uni-nordstadt.example/job/Wissenschaftliche-Mitarbeiterin-KI/1454149833/?from=email&refid=1">'
                         'Wissenschaftliche/r Mitarbeiter/in - KI</a>')
    assert (c.company, c.url) == ("Universität Nordstadt", "https://jobs.uni-nordstadt.example/job/Wissenschaftliche-Mitarbeiterin-KI/1454149833/")


def test_stepstone_recommendation_and_indeed_single_job_emails():
    stepstone = ("<p>Hallo Max,</p><p>das passt perfekt! Basierend auf deinem Profil und deiner letzten Suche könnte das der "
                 "richtige Job für dich sein.</p><p>Softwareentwickler (m/w/d)</p><p>Beispiel Automobil Treuhand ...</p>"
                 "<p>Ostfildern</p><p>Feste Anstellung</p><p>vor 15 Tagen</p>"
                 '<a href="https://click.stepstone.de/f/a/xyz">Ich bin interessiert</a>')
    (p,) = B.StepStoneAlertParser().parse(stepstone)
    assert (p.title, p.company, p.location, p.url) == ("Softwareentwickler (m/w/d)", "Beispiel Automobil Treuhand", "Ostfildern", None)

    indeed = ("<p>Global AI Lead (m/w/d) bei Beispiel GmbH</p><p>aufgrund Ihres Profils könnte diese Stelle passen.</p>"
              '<a href="https://cts.indeed.com/v3/notajobkey">Global AI Lead (m/w/d)</a><p>Beispiel GmbH</p><p>Ravensburg</p>'
              "<p>Anstellungsarten</p>")
    (i,) = B.IndeedAlertParser().parse(indeed)
    assert (i.title, i.company, i.location, i.url) == ("Global AI Lead (m/w/d)", "Beispiel GmbH", "Ravensburg", None)
    assert i.external_job_id.startswith("in-")
