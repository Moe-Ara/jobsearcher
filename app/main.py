"""FastAPI application: HTML dashboard + JSON API + in-process scheduler."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.config_files import config_store, ensure_config
from app.db import run_migrations
from app.scheduler import start_scheduler, stop_scheduler
from app.services.companies import seed_if_empty
from app.settings import get_settings
from app.web import api, applications, auth, routes, setup
from app.web.templating import STATIC_DIR


def _configure_logging() -> None:
    logging.basicConfig(
        level=get_settings().log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "apscheduler", "alembic.runtime.plugins", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _configure_logging()
    ensure_config()  # first start: your config/ is created from config.example/
    config_store.reload()
    run_migrations()
    seed_if_empty()
    from app.services import tasks

    tasks.recover_interrupted()  # letter/humanize runs killed by the last restart → shown as failed, not lost silently
    from app.services.application_agent import mark_interrupted

    mark_interrupted()  # application-agent runs killed by the last restart → package marked failed
    start_scheduler()
    if get_settings().scheduler_enabled:  # the preparation queue continues where the restart stopped it
        from app.services import bulk

        bulk.resume_after_restart()
    if get_settings().imap_configured:  # background connection check → Operations shows the mailbox status at once
        from app.email_ingest.service import check_imap

        tasks.run_in_background("imap-check", check_imap)
    yield
    stop_scheduler()


app = FastAPI(title="JobSearcher", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(auth.router)
app.include_router(routes.router)
app.include_router(setup.router)
app.include_router(applications.router)
app.include_router(api.router)


def _same_origin(request: Request) -> bool:
    # Modern browsers state it directly; this is the most reliable signal.
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site:
        return fetch_site in ("same-origin", "none")
    host = request.headers.get("host", "")
    origin = request.headers.get("origin")
    if origin and origin != "null":
        return urlsplit(origin).netloc == host
    referer = request.headers.get("referer")
    if referer:
        return urlsplit(referer).netloc == host
    # No browser provenance headers at all (scripts, curl, the JSON API from the host): allowed. A cross-site
    # browser request always carries Sec-Fetch-Site or Origin, so it can't take this path.
    return origin is None


@app.middleware("http")
async def same_origin_posts(request: Request, call_next):
    """Basic CSRF protection: state-changing requests must come from this app's own pages. Sign-in when
    APP_PASSWORD is set. Static files are cached long (their URLs carry a version, see templating.asset)."""
    if request.method not in ("GET", "HEAD", "OPTIONS") and not _same_origin(request):
        return PlainTextResponse("Cross-origin request blocked", status_code=403)
    if not auth.authorized(request):
        return auth.deny(request)
    response = await call_next(request)
    if request.url.path.startswith("/static/") and response.status_code == 200:
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable" if request.url.query else "public, max-age=3600"
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    # same-origin (not no-referrer): with no-referrer, browsers send "Origin: null" on form posts
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("X-Frame-Options", "DENY")
    return response


# Pages are mostly text (a job list is ~300 KB of HTML): compressed ~8x. Added last = runs outermost.
app.add_middleware(GZipMiddleware, minimum_size=1024)
