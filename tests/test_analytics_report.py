import io
from datetime import timedelta

from fastapi.testclient import TestClient
from pypdf import PdfReader

from app.db import session_scope
from app.main import app
from app.models import Job, StatusEvent, utcnow
from app.services.analytics_report import period_label, report_data, week_key


def _pdf_text(r) -> str:
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert "attachment" in r.headers["content-disposition"]
    return " ".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(r.content)).pages)


def _applied(title, company, applied_at, answer=None, answer_at=None):
    with session_scope() as s:
        j = Job(title=title, company=company, source="linkedin", status=answer or "APPLIED", track="main",
                normalized_company=company.lower(), normalized_title=title.lower(), first_seen=applied_at - timedelta(days=2))
        s.add(j)
        s.flush()
        s.add(StatusEvent(job_id=j.id, from_status="NEW", to_status="APPLIED", changed_at=applied_at))
        if answer:
            s.add(StatusEvent(job_id=j.id, from_status="APPLIED", to_status=answer, changed_at=answer_at))


def test_report_counts_the_chosen_weeks_and_their_answers():
    now = utcnow()
    two_weeks_ago, last_week = now - timedelta(days=14), now - timedelta(days=7)
    _applied("Backend Engineer", "Nordlicht Mobility", two_weeks_ago, "REJECTED", last_week)
    _applied("Data Engineer", "Beispiel Robotics", two_weeks_ago, "INTERVIEW", two_weeks_ago + timedelta(days=3))
    _applied("ML Engineer", "Acme", last_week)
    with session_scope() as s:
        d = report_data(s, [week_key(two_weeks_ago)], "en")
        assert d["outcome"]["applied"] == 2 and d["outcome"]["answered"] == 2 and d["outcome"]["rejected"] == 1
        assert d["outcome"]["interview"] == 1 and d["outcome"]["median_days"] in (3, 5)
        both = report_data(s, [week_key(two_weeks_ago), week_key(last_week)], "en")
        rows = dict(both["rows"])
        assert rows[week_key(last_week)]["applied"] == 1 and rows[week_key(last_week)]["rejections"] == 1  # answer counted when it came
        assert both["total"]["applied"] == 3 and both["outcome"]["waiting"] == 1


def test_report_downloads_in_english_and_german_for_picked_weeks():
    now = utcnow()
    _applied("Backend Engineer", "Nordlicht Mobility", now - timedelta(days=7), "REJECTED", now - timedelta(days=2))
    c = TestClient(app)
    wk = week_key(now - timedelta(days=7))
    en = _pdf_text(c.get(f"/analytics/report.pdf?weeks={wk}"))
    for section in ("Job Search Report", "Week by week", "What became of the applications", "Timeline of employer answers",
                    "Nordlicht Mobility", "Response rates"):
        assert section in en, section
    de = _pdf_text(c.get(f"/analytics/report.pdf?weeks={wk}&weeks={week_key(now)}&lang=de"))
    for section in ("Bewerbungsbericht", "Woche für Woche", "Zeitleiste der Antworten", "Absage"):
        assert section in de, section
    assert _pdf_text(c.get("/analytics/report.pdf"))  # every week
    page = c.get("/analytics").text
    assert 'name="weeks"' in page and "Bericht herunterladen (Deutsch)" in page


def test_period_labels():
    assert period_label(["2026-W46", "2026-W47"], "en").startswith("Weeks 46–47 2026")
    assert period_label(["2026-W46", "2026-W47"], "de").startswith("KW 46–47 2026 (09.11.2026")
    assert period_label(["2026-W41", "2026-W44"], "en") == "Weeks 41, 44 2026"
