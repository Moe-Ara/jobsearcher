"""Final-pass hardening: evidence levels, relevance labels & metrics, calibration runs, defence, provenance, IMAP."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config_files import CandidateProfile, get_prefs, get_profile
from app.core.normalize import normalize_company
from app.db import session_scope
from app.enums import AnalysisStatus, FilterStatus, JobStatus, SourceType
from app.filters import apply_hard_filters
from app.llm.roles import build_setup
from app.llm.schemas import ClaimCheck, CoverLetterDraft, FactCheckResult
from app.main import app
from app.models import Job, JobAnalysis, JobSource, SourceRun, TargetCompany
from app.services import analysis, companies
from app.services.analysis import verify_and_clean
from app.services.calibration import precision_at_k, ranking_metrics, set_relevance
from app.services.evidence import find_level_violations, reconcile_claim
from app.services.ingest import ingest_posting
from app.services.letters import generate_letter
from app.settings import Settings
from app.sources.base import JobPosting, JobSourceAdapter
from tests.conftest import FakeProvider, fit_result, screening_result

Q8 = "hf.co/Qwen/Qwen3-8B-GGUF:Q5_K_M"
Q14 = "hf.co/Qwen/Qwen3-14B-GGUF:Q5_K_M"
MISTRAL = "mistral-small-3.1-text:q4_k_s"


def local_setup(local=None, **kw):
    local = local or FakeProvider(name="local", is_paid=False)
    setup = build_setup(Settings(_env_file=None, local_llm_enabled=True, **kw), get_prefs(), local=local, cloud=None, cloud_models=None)
    setup.provider = local
    return setup


def new_job(s, title="Applied AI Engineer", company="Acme", description=None, **kw):
    description = description or f"{title}: Python, machine learning and Docker for industrial AI products. " * 5
    return ingest_posting(s, JobPosting(source="manual", source_type=SourceType.MANUAL, title=title, company=company,
                                        location=kw.pop("location", "Stuttgart"), description=description, **kw)).job


# --------------------------------------------------------------------------- 1. profile


def test_profile_extra_fields_and_skill_groups_preserved():
    p = CandidateProfile.model_validate({
        "headline": "X", "hobbies": ["chess"],
        "skills": {"direct_professional": ["Java"], "academic": ["C++"], "project": ["C++"], "certifications": ["AWS SAA"]},
        "experience": [{"employer": "A", "role": "Dev", "highlights": ["x"], "team_size": 5}],
        "education": [{"degree": "M.Sc.", "institution": "Uni"}],
    })
    text = p.as_prompt_text()
    assert "hobbies" in text and "team_size" in text and "AWS SAA" in text and "institution: Uni" in text
    assert p.skills.levels()["c++"] == {"academic", "project"}
    assert ("AWS SAA", "certifications") in p.skills.all()
    assert "- C++: academic, project" in p.evidence_map_text()
    warnings = p.completeness_warnings()
    assert "experience 'A': period missing" in warnings and "education 'M.Sc.': period missing" in warnings
    assert "years_professional_experience is not set" in warnings


def test_full_profile_reaches_all_prompts(fake_llm):
    from app.llm.letter_prompts import factcheck_user, writer_user
    from app.llm.prompts import fit_system_prompt, screen_system_prompt

    profile, prefs = get_profile(), get_prefs()
    job = SimpleNamespace(title="T", company="C", location="L", employment_type=None, description="d", responsibilities="", requirements="")
    for text in (screen_system_prompt(profile, prefs), fit_system_prompt(profile, prefs),
                 writer_user(profile, job, prefs.writing, "de"), factcheck_user(profile, job, "Brief")):
        assert "Hochschule Nordstadt" in text and "location_preferences" in text and "SKILL_EVIDENCE" in text


# --------------------------------------------------------------------------- 2. evidence levels


@pytest.mark.parametrize("sentence,flagged", [
    ("Durch meine mehrjährige berufliche Tätigkeit als C++-Entwickler bringe ich Erfahrung mit.", "C++"),
    ("I have several years of professional C++ development experience.", "C++"),
    ("Ich bringe berufliche Erfahrung in KI und Machine Learning mit.", "Machine learning"),
    ("Beruflich habe ich bereits TypeScript- und Next.js-Anwendungen entwickelt.", "TypeScript"),
    ("Bei Nordlicht Mobility Software habe ich mit ROS2 Roboter gesteuert.", "ROS2"),
])
def test_exaggerated_evidence_level_is_flagged(profile, sentence, flagged):
    skills = [v.skill for v in find_level_violations(sentence, profile)]
    assert flagged in skills


@pytest.mark.parametrize("sentence", [
    "Im Studium und in eigenen Projekten habe ich intensiv mit C++ gearbeitet.",
    "In meiner Masterarbeit habe ich LLM-basierte Planung und Klassifikationsmodelle eingesetzt.",
    "Bei Nordlicht Mobility Software habe ich ein Java/Spring-basiertes Validierungsframework entwickelt.",
    "Beruflich habe ich mit Docker, Jenkins und Python gearbeitet.",
    "Die technische Ausrichtung der Stelle finde ich besonders interessant.",
])
def test_correct_evidence_levels_pass(profile, sentence):
    assert find_level_violations(sentence, profile) == []


def test_reconcile_professional_claim_with_academic_support():
    c = reconcile_claim({"claim": "C++ developer", "claimed_level": "professional", "status": "SUPPORTED_ACADEMIC", "note": ""})
    assert c["status"] == "UNSUPPORTED" and "exaggeration" in c["note"]
    ok = reconcile_claim({"claim": "Java", "claimed_level": "professional", "status": "SUPPORTED_PROFESSIONAL"})
    assert ok["status"] == "SUPPORTED_PROFESSIONAL"


def test_validator_removes_level_exaggeration_even_if_llm_misses_it():
    bad = "Durch meine mehrjährige berufliche Tätigkeit als C++-Entwickler kenne ich performante Systeme."
    good = "Im Studium und in eigenen Projekten habe ich intensiv mit C++ gearbeitet."
    letter = f"Sehr geehrte Damen und Herren,\n\n{bad} {good}\n\nMit freundlichen Grüßen\n[Name]"
    setup = local_setup()
    setup.provider.responses["CoverLetterDraft"] = CoverLetterDraft(angle="a", selected_evidence=[], subject="Bewerbung", letter=letter)
    # the (simulated) LLM wrongly accepts the professional claim
    setup.provider.responses["FactCheckResult"] = FactCheckResult(claims=[
        ClaimCheck(sentence=bad, claim="professional C++", claimed_level="professional", status="SUPPORTED_PROFESSIONAL", profile_evidence="C++", note=""),
        ClaimCheck(sentence=good, claim="C++ in study/projects", claimed_level="academic", status="SUPPORTED_ACADEMIC", profile_evidence="C++", note=""),
    ])
    setup.provider.texts = [letter.replace("Sehr geehrte", "Sehr  geehrte")]  # "correction" that changes nothing
    with session_scope() as s:
        job_id = new_job(s, title="C++ Softwareentwickler").id
    generate_letter(job_id, language="de", use_editor=False, setup=setup)
    with session_scope() as s:
        from app.models import ApplicationDocument

        final = s.query(ApplicationDocument).filter_by(stage="final").one()
        assert "mehrjährige berufliche Tätigkeit als C++" not in final.content
        assert good in final.content
        guard = [c for c in final.validation["claims"] if c.get("source") == "evidence-guard"]
        assert guard and guard[0]["status"] == "UNSUPPORTED" and "C++" in guard[0]["note"]
        assert final.validation["counts"]["SUPPORTED_ACADEMIC"] >= 1


def test_deep_analysis_cannot_label_academic_skill_professional(profile):
    result = fit_result(strong_matches=[
        {"skill": "C++", "evidence": "direct_professional", "profile_evidence": "C++"},
        {"skill": "Java", "evidence": "direct_professional", "profile_evidence": "Java"},
        {"skill": "Machine learning", "evidence": "direct_professional", "profile_evidence": "Machine learning"},
    ]).model_dump()
    cleaned = verify_and_clean(result, profile)
    levels = {m["skill"]: m["evidence"] for m in cleaned["strong_matches"]}
    assert levels == {"C++": "academic", "Java": "direct_professional", "Machine learning": "academic"}
    assert len(cleaned["evidence_level_corrections"]) == 2


# --------------------------------------------------------------------------- 4-6. labels & metrics


def test_relevance_labels_are_independent_of_status():
    client = TestClient(app)
    with session_scope() as s:
        job = new_job(s)
        job.fit_score = 7.3
        job_id = job.id
    r = client.post(f"/jobs/{job_id}/relevance", data={"label": "GREAT", "reasons": ["too generic", "not-a-reason"]},
                    headers={"HX-Request": "true"})
    assert r.status_code == 200 and "rel-great active" in r.text
    with session_scope() as s:
        job = s.get(Job, job_id)
        assert job.relevance_label == "GREAT" and job.relevance_reasons == ["too generic"] and job.relevance_labeled_at
        assert job.status == JobStatus.NEW and job.fit_score is not None
        score = job.fit_score
    client.post(f"/jobs/{job_id}/relevance", data={"label": "BAD"})
    with session_scope() as s:
        job = s.get(Job, job_id)
        assert job.relevance_label == "BAD" and job.fit_score == score  # labels never change scores
    assert client.post(f"/jobs/{job_id}/relevance", data={"label": "AWESOME"}).status_code == 400
    assert "GREAT" in client.get("/?applied=1&unreviewed=0&relevance=BAD&since_days=0").text


def test_precision_at_k():
    ranking = ["GREAT", "BAD", "GREAT", "MAYBE", "BAD", "GREAT", "BAD", "BAD", "MAYBE", "GREAT", "BAD"]
    assert precision_at_k(ranking, 5, {"GREAT"})["value"] == 0.4
    assert precision_at_k(ranking, 10, {"GREAT"})["value"] == 0.4
    assert precision_at_k(ranking, 5, {"GREAT", "MAYBE"})["value"] == 0.6
    short = precision_at_k(ranking, 20, {"GREAT"})
    assert short["n"] == 11 and short["value"] == round(4 / 11, 3)
    assert precision_at_k([], 5, {"GREAT"})["value"] is None


def test_ranking_metrics_against_labels():
    with session_scope() as s:
        spec = [(9.0, "GREAT"), (8.5, "GREAT"), (8.0, "BAD"), (7.0, "MAYBE"), (6.0, "GREAT"), (4.0, "BAD"), (None, "GREAT"), (5.0, "UNRATED")]
        for i, (score, label) in enumerate(spec):
            job = new_job(s, title=f"Engineer {i}", company=f"Firm {i}")
            job.fit_score = score
            set_relevance(job, label)
    with session_scope() as s:
        m = ranking_metrics(s)
    assert m["counts"] == {"UNRATED": 1, "GREAT": 4, "MAYBE": 1, "BAD": 2}
    assert m["ranked"] == 6 and m["unscored_labeled"]["GREAT"] == 1
    p5 = next(p for p in m["primary"] if p["k"] == 5)
    assert p5["value"] == 0.6  # GREAT, GREAT, BAD, MAYBE, GREAT
    assert next(p for p in m["secondary"] if p["k"] == 5)["value"] == 0.8
    assert m["avg_score"] == {"GREAT": round((9 + 8.5 + 6) / 3, 2), "MAYBE": 7.0, "BAD": 6.0}
    assert m["base_rate"]["primary"] == 0.5


def test_calibration_page_labels_and_moves_on():
    client = TestClient(app)
    with session_scope() as s:
        a, b = new_job(s, title="Robotics Engineer", company="R1").id, new_job(s, title="Vision Engineer", company="V1").id
    page = client.get("/calibrate").text
    assert "Precision@k" in page and "GREAT" in page
    first = a if f"/jobs/{a}" in page else b
    r = client.post("/calibrate/label", data={"job_id": first, "label": "MAYBE", "reasons": ["wrong sector"],
                                              "unrated": "1", "rejected": "0", "order": "shuffled", "offset": "0"},
                    follow_redirects=True)
    other = b if first == a else a
    assert f"/jobs/{other}" in r.text
    with session_scope() as s:
        assert s.get(Job, first).relevance_label == "MAYBE"


# --------------------------------------------------------------------------- 7-8. calibration subset & routing


def test_analyze_selected_processes_only_the_subset_with_batching_and_cache(monkeypatch):
    local = FakeProvider(name="local", is_paid=False)
    local.responses["JobScreening"] = lambda user: screening_result(relevance_score=2, likely_relevant=False) \
        if "Clerk" in user else screening_result()
    setup = local_setup(local=local)
    monkeypatch.setattr(analysis, "get_llm", lambda: setup)
    with session_scope() as s:
        selected = [new_job(s, title=t, company=f"Co {i}").id for i, t in enumerate(["AI Engineer", "Clerk Data Entry", "MLOps Engineer"])]
        rejected = new_job(s, title="Werkstudent KI", company="Co X").id
        for i in range(3):
            new_job(s, title=f"Backlog Engineer {i}", company=f"Backlog {i}")  # must not be touched
    result = analysis.analyze_selected(selected + [rejected])
    assert result["queued"] == 4 and result["screened_promising"] == 2
    assert [k for _, k in local.calls] == ["JobScreening"] * 3 + ["JobFitAnalysis"] * 2
    assert {m for m, k in local.calls if k == "JobFitAnalysis"} == {Q14} and all(m != MISTRAL for m, _ in local.calls)
    with session_scope() as s:
        assert s.query(Job).filter(Job.title.like("Backlog%"), Job.analysis_status != AnalysisStatus.PENDING).count() == 0
        clerk = s.query(Job).filter(Job.title.like("Clerk%")).one()
        assert clerk.analysis_status == AnalysisStatus.SCREENED  # 14B never called for it
        assert s.query(JobAnalysis).filter(JobAnalysis.provider == "local").all() and all(
            a.cost_usd == 0 for a in s.query(JobAnalysis))
    calls_before = len(local.calls)
    again = analysis.analyze_selected(selected)
    assert len(local.calls) == calls_before  # unchanged jobs → served from cache, no model call
    assert "analysed (cached)" in again["results"]


# --------------------------------------------------------------------------- 12. defence


def jobns(**kw):
    base = dict(title="Software Engineer", company="Acme", location="Munich", country="Germany", remote_status=None,
                description="Build civilian software in Python.", requirements="", responsibilities="", sector=None,
                years_experience_required=None)
    base.update(kw)
    base["normalized_company"] = normalize_company(base["company"])
    return SimpleNamespace(**base)


def test_defence_rules(prefs):
    assert apply_hard_filters(jobns(company="Airbus Defence and Space GmbH"), prefs).rejected
    assert apply_hard_filters(jobns(company="Rheinmetall AG", description="Develop fire control systems for armoured vehicles."), prefs).rejected
    assert apply_hard_filters(jobns(title="Engineer Military Autonomous Systems"), prefs).rejected
    civil = apply_hard_filters(jobns(company="Airbus Operations GmbH", description="Cabin software for commercial aircraft."), prefs)
    assert civil.status == FilterStatus.FLAGGED  # mixed company, civilian role → human/LLM review, not rejection
    assert apply_hard_filters(jobns(company="SecureCo Cyber Defense GmbH", title="SOC Analyst Automation"), prefs).status == FilterStatus.PASSED


def test_uncertain_llm_defence_verdict_flags_instead_of_rejecting(fake_llm):
    fake_llm.provider.triage = screening_result(defence_related=True, defence_reason="customer list mentions armed forces", confidence=0.4)
    with session_scope() as s:
        job = new_job(s, company="Airbus Operations GmbH")
        outcome = analysis.analyze_job(s, job, fake_llm)
        assert outcome == "analysed" and job.filter_status == FilterStatus.FLAGGED
        assert any(r["rule"] == "llm_defence_check" and r["severity"] == "flag" for r in job.filter_reasons)
    fake_llm.provider.triage = screening_result(defence_related=True, defence_reason="weapons targeting", confidence=0.9)
    with session_scope() as s:
        job = new_job(s, title="Targeting Engineer", company="Other GmbH")
        assert analysis.analyze_job(s, job, fake_llm) == "rejected (defence)" and job.filter_status == FilterStatus.REJECTED


# --------------------------------------------------------------------------- 11. sources, provenance, inactive


class FakeAdapter(JobSourceAdapter):
    name = "fakeats"
    feeds: dict = {}

    def fetch_jobs(self):
        feed = self.feeds[self.company.identifier]
        if isinstance(feed, Exception):
            raise feed
        return feed

    def normalize_job(self, raw):
        return JobPosting(source=self.name, source_type=SourceType.EMPLOYER, external_job_id=raw["id"],
                          url=f"https://jobs.example/{raw['id']}", title=raw["title"], company=self.company.company_name,
                          location="Berlin", description=raw["title"] + " — Python backend services and CI/CD. " * 8)


def test_feed_isolation_inactive_marking_and_provenance(monkeypatch):
    monkeypatch.setitem(companies.ADAPTERS, "fakeats", FakeAdapter)
    FakeAdapter.feeds = {
        "good": [{"id": "1", "title": "Backend Engineer"}, {"id": "2", "title": "Platform Engineer"}],
        "broken": RuntimeError("HTTP 503 from feed"),
    }
    with session_scope() as s:
        good = TargetCompany(company_name="GoodCo", normalized_name="goodco", ats_type="fakeats", ats_identifier="good", ats_config={}, aliases=[])
        broken = TargetCompany(company_name="BrokenCo", normalized_name="brokenco", ats_type="fakeats", ats_identifier="broken", ats_config={}, aliases=[])
        s.add_all([good, broken])
        s.flush()
        good_id, broken_id = good.id, broken.id
        # the same vacancy was seen earlier on a job board
        board = ingest_posting(s, JobPosting(source="linkedin", source_type=SourceType.JOB_BOARD, external_job_id="999",
                                             url="https://www.linkedin.com/jobs/view/999", title="Backend Engineer",
                                             company="GoodCo", location="Berlin", description="short snippet")).job
        board_id = board.id
    result = companies.check_companies([broken_id, good_id])
    assert result["errors"] == 1 and result["new"] == 1 and result["merged"] == 1  # broken feed didn't stop GoodCo
    with session_scope() as s:
        run = s.query(SourceRun).filter(SourceRun.target_company_id == broken_id).one()
        assert run.success is False and "503" in run.errors[0]
        merged = s.get(Job, board_id)
        assert {src.source for src in merged.sources} == {"linkedin", "fakeats"}  # provenance kept
        assert merged.best_source_type == "employer" and merged.canonical_url == "https://jobs.example/1"
        assert s.query(Job).count() == 2  # duplicate not shown twice

    FakeAdapter.feeds["good"] = [{"id": "1", "title": "Backend Engineer"}]  # "2" disappeared from the feed
    companies.check_company(good_id)
    with session_scope() as s:
        gone = s.query(JobSource).filter_by(external_job_id="2").one().job
        assert gone.active is False and gone.status == JobStatus.EXPIRED
        assert s.get(Job, board_id).active is True
        page = TestClient(app).get("/?applied=1&unreviewed=0&show_inactive=1&status=EXPIRED&since_days=0").text
        assert "Platform Engineer" in page and "inactive" in page


# --------------------------------------------------------------------------- IMAP: all folders


def test_imap_folder_selection():
    from app.email_ingest.service import decode_mailbox_name, mailbox_folders, parse_list_response

    lines = [
        b'(\\HasNoChildren) "/" "INBOX"', b'(\\HasNoChildren) "/" "Entw&APw-rfe"', b'(\\HasNoChildren \\Sent) "/" "Gesendet"',
        b'(\\HasNoChildren \\Trash) "/" "Papierkorb"', b'(\\HasNoChildren) "/" "Spam"', b'(\\HasNoChildren) "/" "Jobs/LinkedIn"',
        b'(\\Noselect \\HasChildren) "/" "Jobs"', b'(\\HasNoChildren) "/" "Newsletter"',
    ]
    assert decode_mailbox_name("Entw&APw-rfe") == "Entwürfe"
    assert parse_list_response(lines)[5] == ({"\\hasnochildren"}, "/", "Jobs/LinkedIn")

    class FakeImap:
        def list(self):
            return "OK", lines

    assert mailbox_folders(FakeImap(), "*") == ["INBOX", "Spam", "Jobs/LinkedIn", "Newsletter"]
    assert mailbox_folders(FakeImap(), "INBOX, Jobs/LinkedIn") == ["INBOX", "Jobs/LinkedIn"]

    class Gmail:
        def list(self):
            return "OK", [b'(\\HasNoChildren) "/" "INBOX"', b'(\\All \\HasNoChildren) "/" "[Gmail]/All Mail"',
                          b'(\\HasNoChildren \\Junk) "/" "[Gmail]/Spam"', b'(\\HasNoChildren) "/" "Jobs"']

    assert mailbox_folders(Gmail(), "*") == ["[Gmail]/All Mail", "[Gmail]/Spam"]


def test_poll_mailbox_scans_all_folders_and_dedups(monkeypatch):
    from email.message import EmailMessage as Mime

    from app.email_ingest import service

    def eml(mid, job_id):
        m = Mime()
        m["From"], m["Subject"], m["Message-ID"] = "jobalerts-noreply@linkedin.com", "Jobs", mid
        m.set_content("x")
        m.add_alternative(f'<a href="https://www.linkedin.com/comm/jobs/view/{job_id}/">AI Engineer {job_id}</a><p>Acme · Berlin</p>', subtype="html")
        return m.as_bytes()

    mail = {"INBOX": {b"1": eml("<a@x>", "4000000001")},
            "Newsletter": {b"7": eml("<b@x>", "4000000002"), b"8": eml("<a@x>", "4000000001")}}  # <a@x> duplicated

    class FakeIMAP:
        def __init__(self, *a, **k):
            self.folder = None

        def login(self, u, p):
            return "OK", []

        def list(self):
            return "OK", [b'(\\HasNoChildren) "/" "INBOX"', b'(\\HasNoChildren) "/" "Newsletter"', b'(\\Trash) "/" "Papierkorb"']

        def select(self, name, readonly=False):
            self.folder = name.strip('"')
            assert readonly
            return "OK", [b"1"]

        def uid(self, cmd, *args):
            box = mail[self.folder]
            if cmd == "search":
                return "OK", [b" ".join(box)] if "linkedin.com" in args[-1] else [b""]
            raw = box[args[0]]
            if "HEADER.FIELDS" in args[1]:
                mid = [ln for ln in raw.split(b"\n") if ln.lower().startswith(b"message-id")][0]
                return "OK", [(b"hdr", mid + b"\r\n\r\n")]
            return "OK", [(b"body", raw)]

        def logout(self):
            return "OK", []

    monkeypatch.setattr(service.imaplib, "IMAP4_SSL", FakeIMAP)
    monkeypatch.setattr(service, "get_settings", lambda: Settings(_env_file=None, imap_host="imap.web.de", imap_user="me@web.de",
                                                                  imap_password="pw", imap_folder="*"))
    result = service.poll_mailbox()
    assert result["folders"] == 2 and result["messages"] == 2 and result["new"] == 2 and result["errors"] == 0


def test_degree_gaps_already_satisfied_are_removed(profile):
    from app.services.analysis import degree_gap_satisfied

    assert degree_gap_satisfied("Studium der Informatik oder vergleichbar", profile)
    assert degree_gap_satisfied("Bachelor's degree in Computer Science or related field", profile)
    assert degree_gap_satisfied("Abgeschlossenes Studium im MINT-Bereich", profile)
    assert not degree_gap_satisfied("Promotion in Physik", profile)
    assert not degree_gap_satisfied("PhD in Machine Learning", profile)
    assert not degree_gap_satisfied("Kubernetes", profile)
    cleaned = verify_and_clean(fit_result(critical_gaps=["Studium der Informatik oder vergleichbar", "Kubernetes"]).model_dump(), profile)
    assert cleaned["critical_gaps"] == ["Kubernetes"] and cleaned["resolved_gaps"] == ["Studium der Informatik oder vergleichbar"]


def test_unknown_technology_from_vacancy_is_unsupported(profile):
    from app.services.evidence import find_unknown_technologies

    hit = find_unknown_technologies("Dabei habe ich über 200 automatisierte Tests implementiert und wende Tools wie Mockito an.", profile)
    assert hit and "Mockito" in hit[0]["claim"] and hit[0]["status"] == "UNSUPPORTED"
    assert find_unknown_technologies("Kubernetes würde ich mir gern schnell erarbeiten.", profile) == []  # willingness to learn
    assert find_unknown_technologies("Ihr Team setzt Kubernetes und Azure ein.", profile) == []  # about the company
    assert find_unknown_technologies("Ich habe mit Docker und Jenkins gearbeitet.", profile) == []  # in profile


def test_short_draft_gets_one_rewrite(monkeypatch):
    short = "Sehr geehrte Damen und Herren,\n\nKurz.\n\nMit freundlichen Grüßen\n[Name]"
    long = "Sehr geehrte Damen und Herren,\n\n" + "Ich habe Java und Spring beruflich eingesetzt. " * 40 + "\n\nMit freundlichen Grüßen\n[Name]"
    setup = local_setup(local_german_editor_enabled=False)
    drafts = iter([short, long])
    setup.provider.responses["CoverLetterDraft"] = lambda user: CoverLetterDraft(angle="a", selected_evidence=[], subject="B", letter=next(drafts))
    setup.provider.responses["FactCheckResult"] = FactCheckResult(claims=[])
    with session_scope() as s:
        job_id = new_job(s, title="Java Entwickler").id
    generate_letter(job_id, language="de", setup=setup)
    assert [k for _, k in setup.provider.calls][:2] == ["CoverLetterDraft", "CoverLetterDraft"]
    assert "Wörter" in setup.provider.last_user["CoverLetterDraft"]
    with session_scope() as s:
        from app.models import ApplicationDocument

        draft = s.query(ApplicationDocument).filter_by(stage="draft").one()
        assert draft.meta["word_count"] > 250


def test_stock_phrases_are_surfaced_as_warnings(profile):
    from app.services.letters import validate_letter

    letter = "Sehr geehrte Damen und Herren,\n\nIch bin überzeugt, dass ich passe.\n\nMit freundlichen Grüßen\n[Name]"
    setup = local_setup()
    setup.provider.responses["FactCheckResult"] = FactCheckResult(claims=[])
    with session_scope() as s:
        job = new_job(s)
        _, report, _ = validate_letter(s, setup.target("writer"), job, letter, profile)
    assert any(w["type"] == "style" and "überzeugt" in w["detail"] for w in report["warnings"])


# --------------------------------------------------------------------------- IMAP status / reload


def test_reload_settings_prefers_edited_env_file(tmp_path, monkeypatch):
    import app.settings as settings_mod

    (tmp_path / ".env").write_text("IMAP_HOST=imap.new-host.de\nIMAP_USER=me@web.de\nIMAP_PASSWORD=new-app-password\n", encoding="utf-8")
    monkeypatch.setattr(settings_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("IMAP_HOST", "imap.old-host.de")  # value injected at container start
    try:
        s = settings_mod.reload_settings()
        assert s.imap_host == "imap.new-host.de" and s.imap_password.get_secret_value() == "new-app-password"
        assert settings_mod.get_settings() is s
    finally:
        settings_mod._current = None  # back to plain environment for other tests


def _imap_settings(**kw):
    return Settings(_env_file=None, **{"imap_host": "imap.web.de", "imap_user": "me@web.de", "imap_password": "pw", **kw})


def test_check_imap_reports_connected_and_folders(monkeypatch):
    from app.email_ingest import service

    class OkIMAP:
        def __init__(self, *a, **k): pass
        def login(self, u, p): return "OK", []
        def list(self): return "OK", [b'(\HasNoChildren) "/" "INBOX"', b'(\HasNoChildren) "/" "Newsletter"', b'(\Trash) "/" "Papierkorb"']
        def logout(self): return "OK", []

    monkeypatch.setattr(service.imaplib, "IMAP4_SSL", OkIMAP)
    monkeypatch.setattr(service, "get_settings", lambda: _imap_settings())
    status = service.check_imap()
    assert status["ok"] is True and status["folders"] == ["INBOX", "Newsletter"] and status["error"] is None


def test_check_imap_reports_login_failure_with_hint(monkeypatch):
    from app.email_ingest import service

    class BadLogin:
        def __init__(self, *a, **k): pass
        def login(self, u, p): raise service.imaplib.IMAP4.error(b"authentication failed")
        def logout(self): return "OK", []

    monkeypatch.setattr(service.imaplib, "IMAP4_SSL", BadLogin)
    monkeypatch.setattr(service, "get_settings", lambda: _imap_settings())
    status = service.check_imap()
    assert status["ok"] is False and "authentication failed" in status["error"] and "POP3/IMAP" in status["error"]

    def unreachable(*a, **k):
        raise OSError("Name or service not known")

    monkeypatch.setattr(service.imaplib, "IMAP4_SSL", unreachable)
    assert "cannot reach imap.web.de:993" in service.check_imap()["error"]
    monkeypatch.setattr(service, "get_settings", lambda: Settings(_env_file=None, imap_host=""))
    assert "not configured" in service.check_imap()["error"]


def test_ops_retry_button_reloads_env_and_polls_on_success(monkeypatch):
    import app.web.routes as routes
    from app.email_ingest import service
    from app.services import tasks

    calls = []
    monkeypatch.setattr(routes, "reload_settings", lambda: calls.append("reload"))
    monkeypatch.setattr(service, "check_imap", lambda: {"ok": True, "folders": ["INBOX", "Newsletter"], "error": None})
    monkeypatch.setattr(tasks, "run_in_background", lambda name, fn: calls.append(name) or True)
    r = TestClient(app).post("/ops/imap/test", follow_redirects=False)
    assert calls == ["reload", "email"] and "IMAP+connected" in r.headers["location"]
    monkeypatch.setattr(service, "check_imap", lambda: {"ok": False, "folders": [], "error": "login failed"})
    r = TestClient(app).post("/ops/imap/test", follow_redirects=False)
    assert "err=" in r.headers["location"] and "email" not in calls[2:]
    page = TestClient(app).get("/ops").text
    assert "Mailbox (IMAP)" in page and "Reload .env" in page
