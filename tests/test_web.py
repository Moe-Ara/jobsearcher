import time

from fastapi.testclient import TestClient

from app.db import session_scope
from app.main import app
from app.models import Job, StatusEvent

client = TestClient(app)


def _add_job() -> int:
    r = client.post("/add/job", data={
        "title": "Applied AI Engineer", "company": "Acme", "location": "Berlin",
        "description": "Python, LLM integration, Docker, CI/CD for industrial AI products. " * 5, "analyze_now": "1",
    }, follow_redirects=False)
    assert r.status_code == 303
    job_id = int(r.headers["location"].split("/jobs/")[1].split("?")[0])
    for _ in range(50):  # analysis runs in a background thread
        with session_scope() as s:
            if s.get(Job, job_id).fit_score is not None:
                break
        time.sleep(0.05)
    return job_id


def test_pages_render():
    job_id = _add_job()
    for path in ["/", "/add", "/companies", "/ops", "/analytics", f"/jobs/{job_id}", "/api/health", "/api/jobs", "/api/stats"]:
        assert client.get(path).status_code == 200, path
    assert "Applied AI Engineer" in client.get("/").text


def test_status_actions_and_history():
    job_id = _add_job()
    r = client.post(f"/jobs/{job_id}/action/applied", headers={"HX-Request": "true"})
    assert r.status_code == 200 and "act-applied active" in r.text
    client.post(f"/jobs/{job_id}/status", data={"status": "INTERVIEW", "note": "first round"})
    with session_scope() as s:
        job = s.get(Job, job_id)
        assert job.status == "INTERVIEW" and job.applied_at is not None
        assert [e.to_status for e in s.query(StatusEvent).order_by(StatusEvent.id)] == ["APPLIED", "INTERVIEW"]
    # unreviewed view hides it now
    assert "Applied AI Engineer" not in client.get("/").text
    assert "Applied AI Engineer" in client.get("/?applied=1&unreviewed=0").text
    data = client.get("/analytics").text
    assert "Interview rate" in data


def test_search_and_api():
    _add_job()
    assert client.get("/api/jobs", params={"q": "docker industrial", "unreviewed": "0", "applied": "1"}).json()["total"] == 1
    assert client.get("/api/jobs", params={"q": "kotlin", "applied": "1"}).json()["total"] == 0


def test_cross_origin_post_blocked():
    r = client.post("/ops/rescore", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_manual_url_rejects_private_hosts():
    r = client.post("/add/job", data={"url": "http://127.0.0.1:8000/admin"}, follow_redirects=False)
    assert r.status_code == 303 and "err=" in r.headers["location"]


def test_browser_form_posts_are_accepted_and_cross_site_blocked():
    # what a real browser sends for a form on our own page (Origin may be "null" depending on referrer policy)
    assert client.post("/ops/rescore", headers={"Sec-Fetch-Site": "same-origin", "Origin": "null"},
                       follow_redirects=False).status_code == 303
    assert client.post("/ops/rescore", headers={"Origin": "http://testserver"}, follow_redirects=False).status_code == 303
    assert client.post("/ops/rescore", headers={"Sec-Fetch-Site": "cross-site", "Origin": "https://evil.example"}).status_code == 403
    assert client.post("/ops/rescore", headers={"Origin": "null"}).status_code == 403  # sandboxed cross-site frame
    assert client.get("/").headers["referrer-policy"] == "same-origin"


def test_redirect_keeps_message_before_fragment():
    from app.web.routes import _redirect

    assert _redirect("/ops#mailbox", err="x y").headers["location"] == "/ops?err=x+y#mailbox"
    assert _redirect("/jobs/1#letters", msg="ok").headers["location"] == "/jobs/1?msg=ok&focus=letters#letters"
