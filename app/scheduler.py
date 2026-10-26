"""Lightweight in-process scheduler (APScheduler). Runs in the single uvicorn worker."""

from __future__ import annotations

import logging
from datetime import timedelta

from apscheduler.schedulers.background import BackgroundScheduler

from app.config_files import get_prefs
from app.models import utcnow
from app.services import tasks
from app.settings import get_settings

log = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def start_scheduler() -> BackgroundScheduler | None:
    global _scheduler
    if not get_settings().scheduler_enabled:
        log.info("Scheduler disabled (SCHEDULER_ENABLED=false)")
        return None
    prefs = get_prefs().scheduler
    sched = BackgroundScheduler(
        # misfire_grace_time=None: runs missed while the PC slept are not dropped — each job runs once (coalesce)
        # right after waking, instead of waiting for its next interval.
        timezone="UTC", job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": None}
    )
    first = utcnow() + timedelta(minutes=1)
    # Companies are checked when due (per-priority interval); the tick only decides how often we look.
    sched.add_job(
        lambda: tasks.run_exclusive("companies", tasks.task_check_due_companies),
        "interval", minutes=prefs.tick_minutes, next_run_time=first, id="companies",
    )
    # Always scheduled: the task skips itself while IMAP isn't configured, so fixing .env later needs no restart.
    sched.add_job(
        lambda: tasks.run_exclusive("email", tasks.task_poll_email),
        "interval", minutes=prefs.email_check_minutes, next_run_time=first + timedelta(minutes=1), id="email",
    )
    # Retention: delete jobs older than the window (default 14 days) a few times a day; first run shortly after start.
    sched.add_job(
        lambda: tasks.run_exclusive("retention", tasks.task_retention),
        "interval", hours=6, next_run_time=first + timedelta(minutes=2), id="retention",
    )
    sched.add_job(
        lambda: tasks.run_exclusive("analysis", tasks.task_analyze),
        "interval", minutes=prefs.analysis_minutes, next_run_time=first + timedelta(minutes=3), id="analysis",
    )
    sched.start()
    _scheduler = sched
    log.info("Scheduler started: %s", ", ".join(j.id for j in sched.get_jobs()))
    return sched


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def scheduled_jobs() -> list[dict[str, str]]:
    if not _scheduler:
        return []
    return [{"id": j.id, "next_run": j.next_run_time.strftime("%Y-%m-%d %H:%M UTC") if j.next_run_time else "-"} for j in _scheduler.get_jobs()]
