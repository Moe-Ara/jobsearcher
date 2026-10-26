from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.main import app
from app.settings import get_settings
from app.web import auth


def test_without_a_password_everything_stays_open():
    c = TestClient(app)
    assert c.get("/internships").status_code == 200
    assert c.get("/login", follow_redirects=False).status_code == 303


def test_with_a_password_pages_and_api_need_sign_in(monkeypatch):
    monkeypatch.setattr(get_settings(), "app_password", SecretStr("s3cret"))
    c = TestClient(app)
    r = c.get("/internships?x=1", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login?next=/internships%3Fx%3D1"
    assert c.get("/api/jobs").status_code == 401
    assert c.get("/api/health").status_code == 200 and c.get("/static/style.css").status_code == 200
    assert c.get("/api/jobs", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.post("/login", data={"password": "wrong", "next": "/"}).status_code == 401
    r = c.post("/login", data={"password": "s3cret", "next": "//evil.example"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"  # no open redirect
    assert c.get("/internships").status_code == 200  # cookie set by the sign-in


def test_session_cookies_expire_and_die_with_a_password_change():
    cookie = auth.make_cookie("pw", now=1_000_000)
    assert auth.valid_cookie(cookie, "pw", now=1_000_000 + 3600)
    assert not auth.valid_cookie(cookie, "pw", now=1_000_000 + auth.MAX_AGE + 1)
    assert not auth.valid_cookie(cookie, "other", now=1_000_000 + 3600)
    assert not auth.valid_cookie("123.forged", "pw", now=200)


def test_pages_are_compressed_and_static_assets_cached():
    c = TestClient(app)
    r = c.get("/internships", headers={"Accept-Encoding": "gzip"})
    assert r.headers.get("content-encoding") == "gzip" or len(r.content) < 1024
    from app.web.templating import asset

    url = asset("style.css")
    assert "?v=" in url and "immutable" in c.get(url).headers["cache-control"]
