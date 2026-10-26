"""Alert parsing fixes (Indeed / StepStone), full-posting enrichment, mailbox clean-up, sending and bulk runs."""

from types import SimpleNamespace

import pytest

from app.config_files import get_prefs
from app.db import session_scope
from app.enums import JobStatus, SourceType
from app.models import ApplicationPackage, EmailMessage, Job, utcnow
from app.services import submit
from app.services.ingest import ingest_posting
from app.sources.base import JobPosting
from tests.test_email import cts


# ------------------------------------------------------------------ parsers


def test_indeed_job_key_from_all_link_shapes():
    from app.email_ingest.parsers.boards import indeed_job_key

    assert indeed_job_key("https://de.indeed.com/viewjob?jk=0123456789abcdef&from=x") == "0123456789abcdef"
    assert indeed_job_key("https://de.indeed.com/pagead/clk/dl?jrtk=5-cmh1-1-1k5-0123456789abcdef&amp;x=1") == "0123456789abcdef"
    assert indeed_job_key(cts("fedcba9876543210")) == "fedcba9876543210"
    assert indeed_job_key(cts(None)) is None
    assert indeed_job_key("https://cts.indeed.com/v3/H4sIAAAA_broken") is None


def test_stepstone_ui_cards_are_dropped_and_magic_links_never_stored():
    from app.email_ingest.parsers.boards import StepStoneAlertParser

    html = """<div><a href="https://click.stepstone.de/f/a/A~~/1">Weil du dir diese Fähigkeiten</a><div>Top-Fähigkeiten</div><div>Bei diesem Job kannst du</div></div>
    <div><a href="https://www.stepstone.de/v2/magiclink/exchange?magicLink=eyJ">Data Engineer (m/w/d)</a><div>Acme GmbH</div><div>Ulm</div></div>
    <div><a href="https://click.stepstone.de/f/a/C~~/3">Diesen Job melden</a><div>Erhöhe deine Chancen</div><div>Du passt</div></div>"""
    jobs = StepStoneAlertParser().parse(html)
    assert [(j.title, j.company) for j in jobs] == [("Data Engineer (m/w/d)", "Acme GmbH")]
    assert jobs[0].url is None


def test_alert_junk_is_purged_but_used_jobs_are_kept():
    from app.email_ingest.junk import purge_alert_junk

    with session_scope() as s:
        for title, company in (("Anmelden", "Unsere Top-Empfehlung"), ("Mindestgrundgehalt", "50.000 € pro Jahr"),
                               ("Data Engineer", "Acme GmbH")):
            ingest_posting(s, JobPosting(source="indeed", source_type=SourceType.JOB_BOARD, title=title, company=company,
                                         external_job_id=title, location="Ulm"))
        kept = ingest_posting(s, JobPosting(source="stepstone", source_type=SourceType.JOB_BOARD, title="Weil du dir",
                                            company="Top-Fähigkeiten", external_job_id="w")).job
        kept.notes = "my note"
    assert purge_alert_junk() == {"deleted": 2, "kept_because_you_used_them": 1}
    with session_scope() as s:
        assert sorted(j.title for j in s.query(Job)) == ["Data Engineer", "Weil du dir"]


# ------------------------------------------------------------------ enrichment


def thin_job(s, **kw):
    base = dict(source="stepstone", source_type=SourceType.JOB_BOARD, title="Senior Software Engineer – (.NET / Az",
                company="UDG Rhein-Main GmbH", location="Mainz", external_job_id="t1", description="Feste Anstellung",
                raw={"truncated_title": True})
    base.update(kw)
    return ingest_posting(s, JobPosting(**base)).job


FULL = ("<html><head><title>Senior Software Engineer – (.NET / Azure) (m/w/d) | UDG Rhein-Main GmbH</title></head>"
        "<body><main><h1>Senior Software Engineer – (.NET / Azure) (m/w/d)</h1><p>" + "Bei der UDG Rhein-Main GmbH "
        "entwickelst du Cloud-Lösungen mit .NET und Azure. " * 20 + "</p></main></body></html>")


class FakeWeb:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def get_text(self, url, params=None):
        self.calls.append(url)
        if url in self.pages:
            return self.pages[url]
        raise RuntimeError("404")


@pytest.fixture
def web(monkeypatch):
    from app.services import enrich, research

    fake = FakeWeb({"https://karriere.udg.de/jobs/senior-dotnet": FULL,
                    "https://other.example/jobs/x": "<html><body><main>" + "Ein ganz anderer Job als Koch. " * 40 + "</main></body></html>"})
    monkeypatch.setattr(enrich, "get_http", lambda: fake)
    monkeypatch.setattr(enrich, "assert_public_http_url", lambda u: None)
    monkeypatch.setattr(research, "web_search", lambda q, limit=8: [
        "https://www.stepstone.de/stellenangebote--x", "https://www.stepstone.de/v2/magiclink/exchange?magicLink=e",
        "https://other.example/jobs/x", "https://karriere.udg.de/jobs/senior-dotnet"])
    return fake


def test_full_posting_is_found_verified_and_applied(web):
    from app.services.enrich import apply_enrichment, find_full_posting

    with session_scope() as s:
        job = thin_job(s)
        found = find_full_posting(job)
        assert found["url"] == "https://karriere.udg.de/jobs/senior-dotnet"
        assert not any("stepstone" in c or "magiclink" in c for c in web.calls)  # never the board or a login link
        assert apply_enrichment(job, found) == "found"
        assert job.title.startswith("Senior Software Engineer – (.NET / Azure)") and len(job.description) > 600
        assert job.career_page_url == found["url"] and job.raw_source_data["enrich"]["result"] == "found"


def test_unrelated_pages_are_not_taken(web):
    from app.services.enrich import find_full_posting

    web.pages.pop("https://karriere.udg.de/jobs/senior-dotnet")
    with session_scope() as s:
        assert find_full_posting(thin_job(s)) is None


def test_enrichment_sweep_skips_recent_attempts(web):
    from app.services.enrich import enrich_thin_jobs

    with session_scope() as s:
        thin_job(s)
    assert enrich_thin_jobs()["found"] == 1
    assert enrich_thin_jobs()["checked"] == 0  # the job has its full text now


# ------------------------------------------------------------------ mailbox clean-up (fake IMAP)


class FakeImap:
    capabilities = (b"IMAP4REV1", b"MOVE")

    def __init__(self):
        self.moved, self.selected = [], []
        self.mail = {"INBOX": {b"1": "<old@indeed.com>", b"2": "<new@indeed.com>", b"3": "<friend@web.de>"},
                     "JobHunting": {b"7": "<old2@stepstone.de>"}}
        self.folder = None

    def list(self):
        return "OK", [b'(\\HasNoChildren) "/" "INBOX"', b'(\\HasNoChildren) "/" "JobHunting"', b'(\\Trash) "/" "Papierkorb"']

    def select(self, name, readonly=False):
        self.folder = name.strip('"')
        self.selected.append((self.folder, readonly))
        return "OK", [b"1"]

    def uid(self, cmd, *args):
        box = self.mail.get(self.folder, {})
        if cmd == "search":
            return "OK", [b" ".join(box)]
        if cmd == "fetch":
            out = []
            for uid in args[0].split(","):
                out.append((f"{uid} (UID {uid} BODY[HEADER.FIELDS (MESSAGE-ID)]".encode(),
                            f"Message-ID: {box[uid.encode()]}\r\n\r\n".encode()))
            return "OK", out
        if cmd == "MOVE":
            self.moved.append((self.folder, args[0], args[1]))
            return "OK", []
        raise AssertionError(cmd)

    def logout(self):
        pass


def test_mailbox_cleanup_moves_only_processed_alerts(monkeypatch):
    from app.email_ingest import cleanup

    fake = FakeImap()
    monkeypatch.setattr(cleanup, "_connect", lambda s: fake)
    monkeypatch.setattr(cleanup, "get_settings", lambda: SimpleNamespace(imap_configured=True, imap_folder="*"))
    old = utcnow().replace(year=utcnow().year - 1)
    with session_scope() as s:
        s.add(EmailMessage(message_id="<old@indeed.com>", parser="indeed", processed_at=old))
        s.add(EmailMessage(message_id="<old2@stepstone.de>", parser="stepstone", processed_at=old))
        s.add(EmailMessage(message_id="<new@indeed.com>", parser="indeed"))  # processed just now: kept for now
        s.add(EmailMessage(message_id="<friend@web.de>", parser=None, processed_at=old))  # not an alert
    preview = cleanup.clean_seen_alerts(dry_run=True)
    assert preview["matched"] == 2 and preview["moved"] == 0 and preview["target"] == "Papierkorb"
    assert fake.moved == [] and all(ro for _, ro in fake.selected)
    run = cleanup.clean_seen_alerts(dry_run=False)
    assert run["moved"] == 2
    assert sorted((f, uids) for f, uids, _ in fake.moved) == [("INBOX", "1"), ("JobHunting", "7")]


# ------------------------------------------------------------------ sending (only on command)


AD = ("Wir suchen eine Data Engineer. Bitte sende deine Bewerbung (Anschreiben, Lebenslauf) an bewerbung@acme.example. "
      "Datenschutz: datenschutz@acme.example · Fragen: noreply@acme.example")


def test_application_address_extraction():
    assert submit.application_address(SimpleNamespace(description=AD, requirements="", responsibilities="")) == "bewerbung@acme.example"
    assert submit.application_address(SimpleNamespace(description="Kontakt: info@acme.example", requirements="",
                                                      responsibilities="")) is None
    assert submit.application_address(SimpleNamespace(description="Apply via our portal. noreply@x.example",
                                                      requirements="", responsibilities="")) is None


@pytest.fixture
def ready_package():
    from app.services.cv import build_cv, get_master_cv

    with session_scope() as s:
        job = ingest_posting(s, JobPosting(source="manual", source_type=SourceType.MANUAL, title="Data Engineer (m/w/d)",
                                           company="Acme GmbH", location="Ulm", description=AD)).job
        pkg = ApplicationPackage(job_id=job.id, status="done", language="de", verdict="SEND", reviews=[], rounds=1,
                                 cv=build_cv(get_master_cv(), "de", [], "Profil."),
                                 email={"subject": "Bewerbung", "body": "Sehr geehrte Damen und Herren, …"})
        s.add(pkg)
        s.flush()
        return job.id, pkg.id


def test_send_application_attaches_the_pdf_and_marks_the_job_applied(ready_package, monkeypatch):
    sent = []
    monkeypatch.setattr(submit, "_smtp_send", lambda msg: sent.append(msg))
    monkeypatch.setattr("app.services.application_agent.package_file",
                        lambda s, pkg, kind: (b"%PDF-1.7 fake", "x.pdf", "application/pdf"))
    job_id, pkg_id = ready_package
    result = submit.send_application(pkg_id)
    msg = sent[0]
    assert msg["To"] == "bewerbung@acme.example" and "alex.beispiel@example.org" in msg["Bcc"]
    assert [p.get_filename() for p in msg.iter_attachments()] == ["Bewerbung_Alex_Beispiel_Acme_GmbH.pdf"]
    with session_scope() as s:
        assert s.get(Job, job_id).status == JobStatus.APPLIED
        assert s.get(ApplicationPackage, pkg_id).meta["submission"]["to"] == result["to"]
    with pytest.raises(submit.SubmitError, match="already sent"):
        submit.send_application(pkg_id)  # never twice


def test_send_rails(ready_package, monkeypatch):
    monkeypatch.setattr(submit, "_smtp_send", lambda msg: pytest.fail("must not send"))
    job_id, pkg_id = ready_package
    with session_scope() as s:
        s.get(ApplicationPackage, pkg_id).verdict = "REVISE"
    with pytest.raises(submit.SubmitError, match="REVISE"):
        submit.send_application(pkg_id)
    with session_scope() as s:
        s.get(Job, job_id).description = "Apply on our careers portal."
    with pytest.raises(submit.SubmitError, match="no application e-mail"):
        submit.send_application(pkg_id, allow_revise=True)
    monkeypatch.setattr(get_prefs().application, "max_sends_per_day", 1)  # an optional cap (0 = no limit)
    monkeypatch.setattr(submit, "_sent_today", lambda: 1)
    with pytest.raises(submit.SubmitError, match="Daily limit"):
        submit.send_application(pkg_id)


def test_mark_applied_for_portal_jobs(ready_package):
    job_id, pkg_id = ready_package
    submit.mark_applied(pkg_id)
    with session_scope() as s:
        assert s.get(Job, job_id).status == JobStatus.APPLIED
        assert s.get(ApplicationPackage, pkg_id).meta["submission"]["channel"] == "manual"


# ------------------------------------------------------------------ bulk


def _jobs(*titles_companies):
    ids = []
    with session_scope() as s:
        for title, company, source in titles_companies:
            job = ingest_posting(s, JobPosting(source=source, source_type=SourceType.MANUAL, title=title, company=company,
                                               location=("Berlin", "München", "Kronberg", "Köln")[len(ids)], description=AD,
                                               url=f"https://{source}.example/{len(ids)}-{title.replace(' ', '-')}")).job
            ids.append(job.id)
    return ids


def test_bulk_prepares_and_only_sends_in_send_mode(monkeypatch):
    from app.services import application_agent, bulk

    j1, j2, j3 = _jobs(("Backend Dev", "One GmbH", "manual"), ("ML Engineer", "Two GmbH", "manual"),
                       ("QA Engineer", "Three GmbH", "manual"))
    monkeypatch.setattr(application_agent, "run_application",
                        lambda job_id, language=None: {"package": job_id * 10, "verdict": "SEND" if job_id != j2 else "REVISE",
                                                       "ats": 80, "recruiter": 8})
    sent = []

    def fake_send(pkg_id, allow_revise=False):
        if pkg_id == j2 * 10:
            raise submit.SubmitError("The reviewer said REVISE")
        sent.append(pkg_id)
        return {"to": "jobs@x.example"}

    monkeypatch.setattr(submit, "send_application", fake_send)
    monkeypatch.setattr(submit, "pause_between_sends", lambda: None)
    prepared = bulk.run_bulk([j1, j2, j3])
    assert prepared["prepared"] == 3 and prepared["sent"] == 0 and sent == []
    result = bulk.run_bulk([j1, j2, j3], send=True)
    assert result["sent"] == 2 and sent == [j1 * 10, j3 * 10]
    assert [r.get("not_sent") for r in result["results"]] == [None, "The reviewer said REVISE", None]


def test_bulk_never_applies_twice_for_one_vacancy(monkeypatch):
    """Already applied (this row or the same posting from another board), or the same vacancy twice in one run."""
    from app.services import application_agent, bulk
    from app.services.jobs import set_status

    a, b, c, d = _jobs(("Software Engineer (m/w/d)", "Usercentrics GmbH", "stepstone"),
                       ("Software Engineer", "Usercentrics", "linkedin"),
                       ("AI Software Engineering", "Avanade", "linkedin"), ("AI Software Engineering", "Avanade", "indeed"))
    with session_scope() as s:
        set_status(s, s.get(Job, a), JobStatus.APPLIED)
    ran = []
    monkeypatch.setattr(application_agent, "run_application",
                        lambda job_id, language=None: ran.append(job_id) or {"package": 1, "verdict": "SEND", "ats": 80,
                                                                              "recruiter": 8})
    result = bulk.run_bulk([a, b, c, d])
    assert ran == [c]
    skipped = {r["job_id"]: r["skipped"] for r in result["results"] if r.get("skipped")}
    assert "already applied" in skipped[a] and f"same vacancy as job {a}" in skipped[b]
    assert skipped[d] == f"same vacancy as job {c} in this queue run" and result["skipped"] == 3


def test_sending_marks_copies_of_the_vacancy_applied_and_blocks_them(ready_package, monkeypatch):
    monkeypatch.setattr(submit, "_smtp_send", lambda msg: None)
    monkeypatch.setattr("app.services.application_agent.package_file",
                        lambda s, pkg, kind: (b"%PDF-1.7 fake", "x.pdf", "application/pdf"))
    job_id, pkg_id = ready_package
    (twin,) = _jobs(("Data Engineer", "Acme", "linkedin"))
    result = submit.send_application(pkg_id)
    assert result["also_marked"] == [twin]
    with session_scope() as s:
        assert s.get(Job, twin).status == JobStatus.APPLIED
        twin_pkg = ApplicationPackage(job_id=twin, status="done", language="de", verdict="SEND", reviews=[], rounds=1,
                                      cv=s.get(ApplicationPackage, pkg_id).cv, email={})
        s.add(twin_pkg)
        s.flush()
        twin_pkg_id = twin_pkg.id
    with pytest.raises(submit.SubmitError, match="already applied"):
        submit.send_application(twin_pkg_id)


def test_an_unconfirmed_send_blocks_a_second_send(ready_package, monkeypatch):
    """The app died between handing the e-mail to SMTP and saving: the marker must stop a duplicate."""
    monkeypatch.setattr("app.services.application_agent.package_file",
                        lambda s, pkg, kind: (b"%PDF-1.7 fake", "x.pdf", "application/pdf"))
    job_id, pkg_id = ready_package

    def crash(msg):
        raise KeyboardInterrupt  # not an SMTP error: simulates the process dying mid-send

    monkeypatch.setattr(submit, "_smtp_send", crash)
    with pytest.raises(KeyboardInterrupt):
        submit.send_application(pkg_id)
    with session_scope() as s:
        assert "sending" in s.get(ApplicationPackage, pkg_id).meta
    monkeypatch.setattr(submit, "_smtp_send", lambda msg: pytest.fail("must not send twice"))
    with pytest.raises(submit.SubmitError, match="never confirmed"):
        submit.send_application(pkg_id)
    submit.mark_applied(pkg_id)  # you checked your Sent folder
    with session_scope() as s:
        assert "sending" not in s.get(ApplicationPackage, pkg_id).meta


def test_a_failed_smtp_send_can_be_retried(ready_package, monkeypatch):
    monkeypatch.setattr("app.services.application_agent.package_file",
                        lambda s, pkg, kind: (b"%PDF-1.7 fake", "x.pdf", "application/pdf"))
    job_id, pkg_id = ready_package

    def down(msg):
        raise OSError("SMTP down")

    monkeypatch.setattr(submit, "_smtp_send", down)
    with pytest.raises(OSError):
        submit.send_application(pkg_id)
    monkeypatch.setattr(submit, "_smtp_send", lambda msg: None)
    assert submit.send_application(pkg_id)["to"] == "bewerbung@acme.example"


def test_bulk_send_needs_the_confirmation_tick():
    from fastapi.testclient import TestClient

    from app.main import app

    r = TestClient(app).post("/applications/bulk", data={"job_ids": ["1"], "mode": "send"}, follow_redirects=False)
    assert r.status_code == 303 and "confirmation" in r.headers["location"]


def test_generic_short_titles_need_the_company_in_page_title_or_url():
    from app.services.enrich import clean_page_title, same_posting

    job = SimpleNamespace(title="Software Engineer", company="Hyra GmbH")
    text = "Software Engineer jobs remote. " * 40
    assert not same_posting(job, "Remote Software Engineer Jobs", text + " hyra", "https://jobs.example/remote")
    assert same_posting(job, "Software Engineer – Hyra", text + " hyra", "https://jobs.example/x")
    assert clean_page_title("Agentic Software Engineer (m/w/d) | YoungCapital.de") == "Agentic Software Engineer (m/w/d)"


def test_bulk_progress_is_reported_live(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app.main import app
    from app.services import application_agent, bulk

    monkeypatch.setattr(bulk, "_state_path", lambda: tmp_path / "bulk_state.json")
    (j1, j2) = _jobs(("Platform Engineer", "Live GmbH", "manual"), ("Data Scientist", "Live Two GmbH", "manual"))
    seen = []

    def fake_run(job_id, language=None):
        seen.append(bulk.status()["current"])  # while the agents work on a job, it is the current one
        return {"package": job_id, "verdict": "SEND", "ats": 90, "recruiter": 8}

    monkeypatch.setattr(application_agent, "run_application", fake_run)
    bulk.run_bulk([j1, j2])
    assert [c["title"] for c in seen] == ["Platform Engineer", "Data Scientist"]
    data = TestClient(app).get("/api/bulk").json()
    assert data["total"] == 0 and not data["running"] and [r["verdict"] for r in data["results"]] == ["SEND", "SEND"]
    assert [r["title"] for r in data["results"]] == ["Data Scientist", "Platform Engineer"] and data["finished"]


def test_bulk_stops_on_repeated_identical_errors_and_can_resume(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app.main import app
    from app.services import application_agent, bulk, tasks

    monkeypatch.setattr(bulk, "_state_path", lambda: tmp_path / "bulk_state.json")
    ids = _jobs(("Platform Engineer", "Resume One GmbH", "manual"), ("Data Engineer", "Resume Two GmbH", "manual"),
                ("ML Engineer", "Resume Three GmbH", "manual"), ("QA Engineer", "Resume Four GmbH", "manual"))
    ids += _jobs(("Cloud Engineer", "Resume Five GmbH", "manual"))
    down = {"on": True}

    def flaky(job_id, language=None):
        if down["on"] and job_id != ids[0]:
            raise RuntimeError("Local inference server unreachable at http://ollama:11434")
        return {"package": job_id, "verdict": "SEND", "ats": 90, "recruiter": 8}

    monkeypatch.setattr(application_agent, "run_application", flaky)
    result = bulk.run_bulk(ids)
    assert result["prepared"] == 1 and result["failed"] == 3 and "same error" in result["stopped"]
    info = bulk.interrupted()
    assert info and info["remaining"] == ids[1:] and "same error" in info["reason"]
    assert "Resume the queue" in TestClient(app).get("/applications").text
    down["on"] = False
    monkeypatch.setattr(tasks, "run_in_background", lambda name, fn: fn() or True)
    r = TestClient(app).post("/applications/bulk/resume", follow_redirects=False)
    assert r.status_code == 303 and "continues" in r.headers["location"]
    assert bulk.interrupted() is None and bulk.saved_state()["finished"] and not bulk.saved_state()["items"]
    assert {x["job_id"] for x in bulk.saved_state()["results"] if x.get("verdict") == "SEND"} == set(ids)


def test_same_vacancy_found_again_after_applying_is_applied_not_new():
    from app.enums import JobStatus
    from app.services import submit
    from app.services.jobs import set_status

    with session_scope() as s:
        first = Job(title="ML Engineer", company="Acme", normalized_company="acme", normalized_title="ml engineer",
                    source="linkedin", status=JobStatus.NEW)
        again = Job(title="ML Engineer", company="Acme", normalized_company="acme", normalized_title="ml engineer",
                    source="smartrecruiters", status=JobStatus.NEW)
        s.add_all([first, again])
        s.flush()
        assert submit.inherit_applied(s, again) is None  # nothing applied yet
        set_status(s, first, JobStatus.APPLIED)
        assert submit.inherit_applied(s, again) == first.id
        assert again.status == JobStatus.APPLIED and submit.applied_reason(s, again)


def test_jobs_sharing_only_a_careers_page_are_not_the_same_vacancy():
    from app.enums import JobStatus
    from app.services import submit

    board = "https://jobs.smartrecruiters.com/BoschGroup"
    with session_scope() as s:
        applied = Job(title="Promotion SoH", company="Bosch", normalized_company="bosch", normalized_title="promotion soh",
                      source="smartrecruiters", status=JobStatus.APPLIED, career_page_url=board,
                      canonical_url="https://jobs.smartrecruiters.com/BoschGroup/744000155670559-promotion-soh")
        other = Job(title="Maintenance Engineer", company="Bosch", normalized_company="bosch",
                    normalized_title="maintenance engineer", source="smartrecruiters", status=JobStatus.NEW,
                    career_page_url=board, canonical_url="https://jobs.smartrecruiters.com/BoschGroup/7440001-maintenance")
        repost = Job(title="Promotion SoH (LinkedIn)", company="Bosch", normalized_company="bosch",
                     normalized_title="promotion soh linkedin", source="linkedin", status=JobStatus.NEW,
                     career_page_url="https://jobs.smartrecruiters.com/BoschGroup/744000155670559-promotion-soh")
        tum = "https://portal.mytum.de/jobs/wissenschaftler"  # a listing page that looks like a posting URL
        phd_a = Job(title="PhD A", company="TUM", normalized_company="tum", normalized_title="phd a", source="academic",
                    status=JobStatus.APPLIED, career_page_url=tum, canonical_url=tum + "/NewsArticle_20261019_165238")
        phd_b = Job(title="PhD B", company="TUM", normalized_company="tum", normalized_title="phd b", source="academic",
                    status=JobStatus.NEW, career_page_url=tum, canonical_url=tum + "/NewsArticle_20240929_181057")
        s.add_all([applied, other, repost, phd_a, phd_b])
        s.flush()
        assert submit.applied_reason(s, other) is None and submit.inherit_applied(s, other) is None
        assert submit.applied_reason(s, phd_b) is None and submit.inherit_applied(s, phd_b) is None
        assert submit.inherit_applied(s, repost) == applied.id  # same posting URL → same vacancy


def test_prepared_packages_leave_the_to_do_tabs_once_the_job_is_no_longer_new():
    from app.web.applications import package_state

    pkg = SimpleNamespace(status="done", verdict="SEND", meta={})
    job = lambda status: SimpleNamespace(status=status, applied_at=None, application_email=None, apply_url=None,  # noqa: E731
                                         career_page_url=None, canonical_url=None, source_url=None, description="",
                                         requirements="", responsibilities="", sources=[])
    assert package_state(pkg, job(JobStatus.NEW))["state"] == "ready"
    assert package_state(SimpleNamespace(status="done", verdict="REVISE", meta={}), job(JobStatus.NEW))["state"] == "review"
    for status in (JobStatus.SKIPPED, JobStatus.EXPIRED, JobStatus.PRIORITY, JobStatus.APPLYING):
        assert package_state(pkg, job(status))["state"] == "closed"
    assert package_state(pkg, job(JobStatus.APPLIED))["state"] == "applied"
