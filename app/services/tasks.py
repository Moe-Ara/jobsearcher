"""Background tasks shared by the scheduler and the UI buttons. One lock per task kind prevents overlap."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from app.models import utcnow


log = logging.getLogger(__name__)

_locks: dict[str, threading.Lock] = {}
_last: dict[str, dict[str, Any]] = {}
_registry_lock = threading.Lock()
_current = threading.local()  # name of the task running in this thread (for progress())

# User-started long runs are tracked on disk too: if the app restarts mid-run, the next start reports the run as
# interrupted instead of it silently vanishing (task state is otherwise in memory only).
TRACKED_PREFIXES = ("letter-", "humanize", "apply-", "bulk-")
_state_lock = threading.Lock()


def _state_file() -> Path:
    from app.settings import data_dir

    return data_dir() / "running_tasks.json"


def _tracked(name: str) -> bool:
    return name.startswith(TRACKED_PREFIXES)


def _update_state(name: str, started: datetime | None) -> None:
    with _state_lock:
        path = _state_file()
        try:
            state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, ValueError):
            state = {}
        if started is None:
            state.pop(name, None)
        else:
            state[name] = started.isoformat()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state), encoding="utf-8")
        except OSError:
            log.warning("Could not write %s", path)


def recover_interrupted() -> list[str]:
    """At startup: runs that were still marked as running were killed by a restart → report them as failed."""
    with _state_lock:
        path = _state_file()
        try:
            state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, ValueError):
            state = {}
        for name, started in state.items():
            when = datetime.fromisoformat(started)
            _last[name] = {"started": when, "finished": utcnow(), "running": False, "interrupted": True,
                           "error": f"Interrupted — the app was restarted while this was running (started {when:%Y-%m-%d %H:%M} UTC). "
                                    "Nothing was saved from this run; please start it again."}
        if state:
            log.warning("Interrupted by restart: %s", ", ".join(state))
            path.write_text("{}", encoding="utf-8")
        return list(state)


def progress(stage: str) -> None:
    """Report the current stage of the task running in this thread (shown in the UI). No-op outside tasks."""
    name = getattr(_current, "name", None)
    if name and name in _last:
        _last[name] = {**_last[name], "stage": stage, "stage_at": utcnow()}


def _lock(name: str) -> threading.Lock:
    with _registry_lock:
        return _locks.setdefault(name, threading.Lock())


def run_exclusive(name: str, fn: Callable[[], Any]) -> Any:
    """Run fn unless a task with the same name is already running (then skip)."""
    lock = _lock(name)
    if not lock.acquire(blocking=False):
        log.info("Task %s already running — skipped", name)
        return None
    started: datetime = utcnow()
    _last[name] = {"started": started, "running": True}
    _current.name = name
    if _tracked(name):
        _update_state(name, started)
    try:
        result = fn()
        _last[name] = {"started": started, "finished": utcnow(), "running": False, "result": result}
        log.info("Task %s finished: %s", name, result)
        return result
    except Exception as exc:  # noqa: BLE001 - background task must never crash the process
        log.exception("Task %s failed", name)
        _last[name] = {"started": started, "finished": utcnow(), "running": False, "error": str(exc)}
        return None
    finally:
        _current.name = None
        if _tracked(name):
            _update_state(name, None)
        lock.release()


def run_in_background(name: str, fn: Callable[[], Any]) -> bool:
    if _lock(name).locked():
        return False
    threading.Thread(target=run_exclusive, args=(name, fn), name=f"task-{name}", daemon=True).start()
    return True


def task_status() -> dict[str, dict[str, Any]]:
    return dict(_last)


def is_running(name: str) -> bool:
    return _lock(name).locked()


# --------------------------------------------------------------------------- task bodies


def _analyse_if_new(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("new") or result.get("merged"):
        run_in_background("analysis", task_analyze)  # don't wait for the next analysis tick
    return result


def task_check_due_companies() -> Any:
    from app.services.companies import check_companies, due_company_ids

    return _analyse_if_new(check_companies(due_company_ids()))


def task_check_all_companies() -> Any:
    from sqlalchemy import select

    from app.db import session_scope
    from app.models import TargetCompany
    from app.services.companies import check_companies
    from app.sources.registry import ADAPTERS

    with session_scope() as s:
        ids = [tc.id for tc in s.scalars(select(TargetCompany).where(TargetCompany.active.is_(True))) if tc.ats_type in ADAPTERS]
    return _analyse_if_new(check_companies(ids))


def task_retention() -> Any:
    from app.services.backup import backup_if_due
    from app.services.retention import purge_old_jobs

    try:  # a daily database copy before anything is deleted
        backup = backup_if_due()
    except Exception as exc:  # noqa: BLE001 - a failed backup must not stop retention, but is reported
        log.warning("Database backup failed: %s", exc)
        backup = {"error": str(exc)}
    return {**purge_old_jobs(), "backup": backup}


def task_backup() -> Any:
    from app.services.backup import backup_database

    return backup_database()


def task_poll_email() -> Any:
    from app.email_ingest.service import poll_mailbox
    from app.services.enrich import enrich_thin_jobs

    result = poll_mailbox()
    if isinstance(result, dict) and (result.get("new") or result.get("merged")):
        # alert e-mails carry only snippets: look up the full postings before the jobs are analysed
        result["enrichment"] = enrich_thin_jobs(limit=25)
    return result


def task_analyze() -> Any:
    from app.services.analysis import analyze_pending
    from app.services.internships import resolve_distances

    try:  # internships wait for their drive time before the models look at them
        distances = resolve_distances()
    except Exception as exc:  # noqa: BLE001 - best effort, never blocks the analysis
        log.warning("Internship drive times failed: %s", exc)
        distances = {"error": str(exc)}
    result = analyze_pending()
    if isinstance(result, dict) and distances.get("waiting"):
        result["internship_distances"] = distances
    return result


def task_full_cycle() -> Any:
    """Ingest everything that's due, then analyse — what the scheduler does."""
    return {
        "companies": run_exclusive("companies", task_check_due_companies),
        "email": run_exclusive("email", task_poll_email),
        "analysis": run_exclusive("analysis", task_analyze),
    }
