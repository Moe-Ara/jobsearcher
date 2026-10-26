"""Optional sign-in: set APP_PASSWORD in .env to protect every page and the API (needed as soon as the app is
reachable by anyone but you — e.g. hosted on a server). Without it the app stays open, as it is meant to run on
127.0.0.1 only.

Browser: a signed, HttpOnly session cookie (valid 30 days; changing the password signs everyone out).
Scripts / JSON API: "Authorization: Bearer <APP_PASSWORD>".
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from urllib.parse import quote

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse

from app.settings import get_settings

COOKIE = "js_session"
MAX_AGE = 30 * 24 * 3600
OPEN_PATHS = ("/login", "/logout", "/api/health", "/static/")

router = APIRouter()


def _password() -> str | None:
    pw = get_settings().app_password
    return pw.get_secret_value() if pw and pw.get_secret_value() else None


def _sign(issued: int, password: str) -> str:
    key = hashlib.sha256(f"jobsearcher-session:{password}".encode()).digest()
    return hmac.new(key, str(issued).encode(), hashlib.sha256).hexdigest()


def make_cookie(password: str, now: float | None = None) -> str:
    issued = int(now or time.time())
    return f"{issued}.{_sign(issued, password)}"


def valid_cookie(value: str | None, password: str, now: float | None = None) -> bool:
    if not value or "." not in value:
        return False
    issued, sig = value.split(".", 1)
    if not issued.isdigit() or (now or time.time()) - int(issued) > MAX_AGE:
        return False
    return hmac.compare_digest(sig, _sign(int(issued), password))


def authorized(request: Request) -> bool:
    password = _password()
    if password is None or request.url.path.startswith(OPEN_PATHS):
        return True
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer ") and hmac.compare_digest(auth[7:].encode(), password.encode()):
        return True
    return valid_cookie(request.cookies.get(COOKIE), password)


def deny(request: Request):
    path = request.url.path
    if path.startswith("/api/"):
        return JSONResponse({"detail": "Sign in required"}, status_code=401)
    if request.method in ("GET", "HEAD"):
        target = path + (f"?{request.url.query}" if request.url.query else "")
        return RedirectResponse(f"/login?next={quote(target)}", status_code=303)
    return PlainTextResponse("Sign in required", status_code=401)


_FORM = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in · JobSearcher</title><link rel="stylesheet" href="/static/style.css"></head>
<body class="login-page"><main class="login-box"><div class="brand"><span class="brand-mark">JS</span>JobSearcher</div>
<form method="post" action="/login"><input type="hidden" name="next" value="{next}">
<label>Password<input type="password" name="password" autofocus required autocomplete="current-password"></label>
{error}<button class="btn primary" type="submit">Sign in</button></form></main></body></html>"""


def _safe_next(target: str) -> str:
    return target if target.startswith("/") and not target.startswith("//") else "/"


@router.get("/login", response_class=HTMLResponse)
def login_form(next: str = "/"):
    from html import escape

    if _password() is None:
        return RedirectResponse("/", status_code=303)
    return HTMLResponse(_FORM.format(next=escape(_safe_next(next)), error=""))


@router.post("/login")
async def login(request: Request, password: str = Form(""), next: str = Form("/")):
    from html import escape

    expected = _password()
    if expected is None:
        return RedirectResponse("/", status_code=303)
    if not hmac.compare_digest(password.encode(), expected.encode()):
        await asyncio.sleep(1.0)  # slows down guessing
        return HTMLResponse(_FORM.format(next=escape(_safe_next(next)),
                                         error='<p class="flash bad">Wrong password.</p>'), status_code=401)
    response = RedirectResponse(_safe_next(next), status_code=303)
    secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    response.set_cookie(COOKIE, make_cookie(expected), max_age=MAX_AGE, httponly=True, samesite="lax", secure=secure)
    return response


@router.post("/logout")
def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(COOKIE)
    return response
