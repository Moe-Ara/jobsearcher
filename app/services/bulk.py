"""Preparation queue — the agents prepare application packages for queued jobs, one after another.

Jobs get into the queue from the "Prepare" button on any job, or from a bulk selection on the Applications page. The
queue is saved on disk (data/bulk_state.json): you can reorder it, move a job to the front or remove it while it runs,
and it continues after an app restart (a job interrupted by the restart is put back at the front).

Each item: prepare only, or prepare and send by e-mail — the latter only when you ticked the confirmation, and only
if the reviewer says SEND and the posting names an address (the same rails as a single send).
A job is never applied for twice: jobs you already applied for — also as a copy of the same vacancy from another
board — are skipped, and so is a second copy of a vacancy in the queue. After 3 jobs fail in a row with the same
error (e.g. the model server is down) the queue pauses instead of failing everything; "Resume" continues it.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from app.models import utcnow
from app.services import tasks

log = logging.getLogger(__name__)

TASK = "bulk-apply"
STOP_AFTER_SAME_ERRORS = 3
MAX_RESULTS = 60

_lock = threading.RLock()


# ---------------------------------------------------------------------------------------------------- persistence

def _state_path():
    from app.services.application_agent import documents_dir

    return documents_dir().parent / "bulk_state.json"


def _empty() -> dict[str, Any]:
    return {"items": [], "results": [], "current": {}, "paused": None, "started": None, "finished": None}


def saved_state() -> dict[str, Any]:
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _empty()
    if "items" not in data:  # file written by the old bulk runner: nothing to continue
        return _empty()
    return {**_empty(), **data}


def _write(state: dict[str, Any]) -> None:
    try:
        _state_path().write_text(json.dumps(state, default=str), encoding="utf-8")
    except OSError as exc:
        log.info("Could not save the preparation queue: %s", exc)


def _update(fn) -> dict[str, Any]:
    with _lock:
        state = saved_state()
        fn(state)
        _write(state)
        return state


# ---------------------------------------------------------------------------------------------------- queue edits

def enqueue(job_ids: list[int], language: str | None = None, send: bool = False) -> dict[str, int]:
    """Add jobs to the end of the queue (a job already queued or being prepared is not added twice)."""
    added = already = 0

    def add(state):
        nonlocal added, already
        queued = {i["job_id"] for i in state["items"]} | {state["current"].get("job_id")}
        for job_id in job_ids:
            if job_id in queued:
                already += 1
                continue
            state["items"].append({"job_id": job_id, "language": language, "send": bool(send),
                                   "added": utcnow().isoformat(timespec="seconds")})
            queued.add(job_id)
            added += 1
        state["paused"] = None

    _update(add)
    return {"added": added, "already_queued": already}


def move(job_id: int, where: str) -> bool:
    """where: up | down | top"""
    def mv(state):
        items = state["items"]
        idx = next((i for i, it in enumerate(items) if it["job_id"] == job_id), None)
        if idx is None:
            return
        item = items.pop(idx)
        new = {"up": max(0, idx - 1), "down": min(len(items), idx + 1), "top": 0}.get(where, idx)
        items.insert(new, item)

    before = [i["job_id"] for i in saved_state()["items"]]
    after = [i["job_id"] for i in _update(mv)["items"]]
    return before != after


def remove(job_id: int) -> bool:
    found = False

    def rm(state):
        nonlocal found
        n = len(state["items"])
        state["items"] = [i for i in state["items"] if i["job_id"] != job_id]
        found = len(state["items"]) < n

    _update(rm)
    return found


def clear() -> int:
    n = len(saved_state()["items"])
    _update(lambda st: st.update(items=[]))
    return n


def queued_ids() -> list[int]:
    st = saved_state()
    ids = [i["job_id"] for i in st["items"]]
    return ([st["current"]["job_id"]] if st["current"].get("job_id") and tasks.is_running(TASK) else []) + ids


# ---------------------------------------------------------------------------------------------------- worker

def start() -> bool:
    """Start the worker in the background (no-op while it runs or when the queue is empty)."""
    if tasks.is_running(TASK) or not saved_state()["items"]:
        return False
    return tasks.run_in_background(TASK, work)


def resume_after_restart() -> bool:
    """App start: a job that was being prepared when the app stopped goes back to the front; the queue continues."""
    state = saved_state()
    cur = state.get("current") or {}
    if cur.get("job_id"):
        def requeue(st):
            if not any(i["job_id"] == cur["job_id"] for i in st["items"]):
                st["items"].insert(0, {k: cur.get(k) for k in ("job_id", "language", "send", "added")})
            st["current"] = {}
        _update(requeue)
    if saved_state()["paused"]:
        return False  # paused after repeated errors: waits for "Resume"
    return start()


def work() -> dict[str, Any]:
    """Prepare queued jobs until the queue is empty (or paused after repeated identical errors)."""
    from app.db import session_scope
    from app.models import Job
    from app.services import application_agent as agent
    from app.services import submit

    _update(lambda st: st.update(started=utcnow().isoformat(timespec="seconds"), finished=None, paused=None))
    seen: dict[tuple[str, str], int] = {}  # vacancy → job of this run that covers it
    streak: list[str] = []
    failed_items: list[dict[str, Any]] = []  # the items of the current error streak
    done: list[dict[str, Any]] = []
    while True:
        if len(streak) >= STOP_AFTER_SAME_ERRORS and len(set(streak[-STOP_AFTER_SAME_ERRORS:])) == 1:
            reason = f"Paused after {STOP_AFTER_SAME_ERRORS} jobs failed with the same error: {streak[-1]}"

            def pause(st):
                st["paused"] = reason
                back = [i for i in failed_items if all(q["job_id"] != i["job_id"] for q in st["items"])]
                st["items"] = back + st["items"]  # "Resume" retries them first

            _update(pause)
            log.warning("Preparation queue paused: %s", reason)
            break
        item: dict[str, Any] = {}

        def pop(st):
            if st["items"]:
                item.update(st["items"].pop(0))
                st["current"] = {**item, "started_ts": time.time()}

        _update(pop)
        if not item:
            break
        row = _process(item, seen, session_scope, Job, agent, submit)
        done.append(row)
        if row.get("error"):
            streak.append(row["error"][:80])
            failed_items.append({k: item.get(k) for k in ("job_id", "language", "send", "added")})
        else:
            streak, failed_items = [], []

        def finish_item(st):
            st["results"] = ([row] + st["results"])[:MAX_RESULTS]
            st["current"] = {}

        _update(finish_item)
    _update(lambda st: st.update(current={}, finished=utcnow().isoformat(timespec="seconds")))
    summary = {"jobs": len(done), "prepared": sum(1 for r in done if r.get("package")),
               "send_verdicts": sum(1 for r in done if r.get("verdict") == "SEND"),
               "sent": sum(1 for r in done if r.get("sent_to")), "failed": sum(1 for r in done if r.get("error")),
               "skipped": sum(1 for r in done if r.get("skipped")), "stopped": saved_state()["paused"], "results": done}
    log.info("Preparation queue: %s", {k: v for k, v in summary.items() if k != "results"})
    return summary


def _process(item, seen, session_scope, Job, agent, submit) -> dict[str, Any]:
    job_id = item["job_id"]
    row: dict[str, Any] = {"job_id": job_id, "at": utcnow().isoformat(timespec="seconds")}
    with session_scope() as s:
        job = s.get(Job, job_id)
        reason = submit.applied_reason(s, job) if job else "job not found"
        key = (job.normalized_company, job.normalized_title) if job else ("", "")
        twin_ids = {t.id for t in submit.vacancy_twins(s, job)} if job else set()
        row.update(title=job.title if job else "?", company=job.company if job else "")
    first = seen.get(key) if all(key) else None
    first = first or next((seen_id for seen_id in seen.values() if seen_id in twin_ids), None)
    if reason or first:
        row["skipped"] = reason or f"same vacancy as job {first} in this queue run"
        return row
    if all(key):
        seen[key] = job_id
    started = time.time()
    left = len(saved_state()["items"])
    tasks.progress(f"Preparing {row['title']} ({left} more queued)")
    try:
        out = agent.run_application(job_id, language=item.get("language"))
        row.update(package=out["package"], verdict=out["verdict"], ats=out["ats"], recruiter=out["recruiter"])
    except agent.ApplicationError as exc:
        if str(exc).startswith("The posting is closed"):  # not a failure: the vacancy is gone (job set to Expired)
            row["skipped"] = str(exc)[:300]
            return row
        row["error"] = str(exc)[:300]
        row["seconds"] = int(time.time() - started)
        return row
    except Exception as exc:  # noqa: BLE001 - one failing job must not stop the queue
        row["error"] = str(exc)[:300]
        row["seconds"] = int(time.time() - started)
        return row
    if item.get("send"):
        try:
            tasks.progress(f"Sending the application for {row['title']}")
            sent = submit.send_application(out["package"])
            row["sent_to"] = sent["to"]
            if saved_state()["items"]:
                tasks.progress("Sent — short pause before the next e-mail")
                submit.pause_between_sends()
        except submit.SubmitError as exc:
            row["not_sent"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - e.g. SMTP down: keep preparing the rest
            row["not_sent"] = f"send failed: {exc}"[:300]
    row["seconds"] = int(time.time() - started)
    return row


def run_bulk(job_ids: list[int], language: str | None = None, send: bool = False) -> dict[str, Any]:
    """Queue these jobs and work through the queue now (blocking) — the bulk form runs it in the background."""
    enqueue(job_ids, language=language, send=send)
    return work()


# ---------------------------------------------------------------------------------------------------- status

def interrupted() -> dict[str, Any] | None:
    """Queued jobs that are not being worked on (paused after errors, or the worker is not running)."""
    if tasks.is_running(TASK):
        return None
    st = saved_state()
    if not st["items"]:
        return None
    return {"remaining": [i["job_id"] for i in st["items"]], "reason": st["paused"] or "the queue is not running",
            "mode": "send" if any(i.get("send") for i in st["items"]) else "prepare"}


def status() -> dict[str, Any]:
    """Queue, current job (with the agent's live step), recent results and an ETA — for the page and the badge."""
    from app.db import session_scope
    from app.models import Job

    st = saved_state()
    running = tasks.is_running(TASK)
    task = tasks.task_status().get(TASK) or {}
    ids = [i["job_id"] for i in st["items"]]
    with session_scope() as s:
        jobs = {j.id: j for j in s.query(Job).filter(Job.id.in_(ids))} if ids else {}
        queue = [{**i, "title": jobs[i["job_id"]].title if i["job_id"] in jobs else "?",
                  "company": jobs[i["job_id"]].company if i["job_id"] in jobs else ""} for i in st["items"]]
    current = dict(st["current"]) if st["current"].get("job_id") else {}  # the worker clears it after each job
    if current:
        with session_scope() as s:
            job = s.get(Job, current["job_id"])
            current.update(title=job.title if job else "?", company=job.company if job else "")
        current["elapsed_s"] = int(time.time() - (current.get("started_ts") or time.time()))
    done = [r["seconds"] for r in st["results"] if r.get("seconds") and not r.get("skipped")]
    eta = None
    if running and done:
        avg = sum(done) / len(done)
        eta = int(max(0, avg - current.get("elapsed_s", 0)) + avg * len(queue))
    return {"running": running, "stage": task.get("stage") if running else None, "current": current, "queue": queue,
            "results": st["results"], "paused": st["paused"], "finished": st["finished"], "eta_s": eta,
            "total": len(queue) + (1 if current else 0)}
