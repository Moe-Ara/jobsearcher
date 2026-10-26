"""Pages and actions of the application agent + reviewer agent (/applications, job page "Application agent")."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from sqlalchemy import select

from app.config_files import get_prefs
from app.db import session_scope
from app.enums import AnalysisStatus, FilterStatus, JobStatus
from app.llm.factory import Role, get_llm
from app.models import ApplicationPackage, Job, utcnow
from app.services import application_agent as agent
from app.services import bulk, latex, submit, tasks, toolbox
from app.services.cv import CvError, cv_path, get_master_cv, render_text
from app.web.templating import templates

router = APIRouter(include_in_schema=False)

FILE_KINDS = ("cv.pdf", "cv.tex", "cv.txt", "letter.pdf", "letter.tex", "letter.txt", "bundle.pdf")


def _redirect(path: str, msg: str = "", err: str = "") -> RedirectResponse:
    from app.web.routes import _redirect as redirect

    return redirect(path, msg=msg, err=err)


@router.post("/jobs/{job_id}/apply")
def apply_start(job_id: int, language: Annotated[str, Form()] = "auto"):
    if agent.busy(job_id):
        return _redirect(f"/jobs/{job_id}#apply", err="The application agent or a letter task is already running for this job")
    if get_llm().target(Role.WRITER) is None:
        return _redirect(f"/jobs/{job_id}#apply", err="No writer model configured — enable the local writer or a cloud provider")
    try:
        get_master_cv()
    except CvError as e:
        return _redirect(f"/jobs/{job_id}#apply", err=str(e))
    try:
        agent.ensure_open(job_id)  # quick check before the agents spend minutes on a closed vacancy
    except agent.PostingClosed as e:
        return _redirect(f"/jobs/{job_id}#apply", err=f"{e} — the job was set to Expired.")
    except agent.ApplicationError as e:
        raise HTTPException(404, str(e)) from e
    try:
        pkg_id = agent.start_package(job_id, language if language in ("de", "en") else None)
    except agent.ApplicationError as e:
        raise HTTPException(404, str(e)) from e
    started = tasks.run_in_background(agent.task_name(job_id),
                                      lambda: agent.run_application(job_id, package_id=pkg_id, check_open=False))
    if not started:
        with session_scope() as s:
            s.delete(s.get(ApplicationPackage, pkg_id))
        return _redirect(f"/jobs/{job_id}#apply", err="Already running")
    return _redirect(f"/applications/{pkg_id}", msg="Application agent started — this page updates by itself")


@router.get("/applications/export.zip")
def applications_zip():
    """All exported application papers in one ZIP (exports any package that isn't on disk yet first)."""
    from app.services import export

    export.export_all()
    stamp = utcnow().strftime("%Y-%m-%d")
    return Response(export.zip_all(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="Bewerbungen_{stamp}.zip"'})


@router.post("/applications/export")
def applications_export():
    from app.services import export

    r = export.export_all()
    failed = f" · {len(r['failed'])} failed (packages {', '.join(map(str, r['failed']))})" if r["failed"] else ""
    return _redirect("/applications", msg=f"Saved the papers of {r['exported']} application(s) to data/applications{failed}")


@router.get("/api/bulk")
def api_bulk():
    """Live progress of the bulk run (the applications page and the header badge poll it)."""
    return JSONResponse(bulk.status())


def package_state(pkg: ApplicationPackage, job: Job | None) -> dict[str, Any]:
    """What you still have to do with a package: applied (sent / marked / the job is Applied or later) · ready (SEND) ·
    review (REVISE) · closed (the job moved on — skipped, expired, prioritised … anything but NEW: it leaves the to-do
    tabs and is only listed under All) · running · failed — plus where to apply."""
    sub = (pkg.meta or {}).get("submission")
    if pkg.status in ("running", "failed"):
        state = pkg.status
    elif sub or (job is not None and job.status in submit.APPLIED_STATES):
        state = "applied"
    elif job is None or job.status != JobStatus.NEW:
        state = "closed"
    else:
        state = "ready" if pkg.verdict == "SEND" else "review"
    applied_at = (sub or {}).get("at") or (job.applied_at.isoformat() if job is not None and job.applied_at else None)
    return {"state": state, "applied_at": applied_at, "sent_to": (sub or {}).get("to"),
            "channel": submit.channel(job) if job is not None else {"channel": "portal", "url": None}}


@router.get("/applications", response_class=HTMLResponse)
def applications_page(request: Request):
    prefs = get_prefs()
    with session_scope() as s:
        packages = agent.recent_packages(s, None)  # every package — no cap
        jobs = {j.id: j for j in s.scalars(select(Job).where(Job.id.in_({p.job_id for p in packages})))} if packages else {}
        shortlist = s.scalars(
            select(Job).where(Job.active.is_(True), Job.filter_status != FilterStatus.REJECTED, Job.track == "main",
                              Job.analysis_status == AnalysisStatus.DONE, Job.fit_score.is_not(None),
                              Job.status.in_([JobStatus.NEW, JobStatus.REVIEW, JobStatus.PRIORITY]))
            .order_by(Job.fit_score.desc()).limit(40)
        ).all()
        channels = {j.id: submit.channel(j)["channel"] for j in shortlist}
        already = {j.id: r for j in shortlist if (r := submit.applied_reason(s, j))}  # e.g. applied via another board
        with_package = {p.job_id for p in packages}
        states = {p.id: package_state(p, jobs.get(p.job_id)) for p in packages}
        counts = {k: sum(1 for st in states.values() if st["state"] == k) for k in ("ready", "review", "applied", "closed", "running", "failed")}
        llm = get_llm()
        return templates.TemplateResponse(request, "applications.html", {
            "request": request, "packages": packages, "jobs": jobs, "shortlist": shortlist, "with_package": with_package,
            "app_prefs": prefs.application, "writer": llm.target(Role.WRITER), "validator": llm.target(Role.VALIDATOR),
            "screen": llm.target(Role.SCREEN), "analysis": llm.target(Role.ANALYSIS), "cv_file": str(cv_path()),
            "tasks": tasks.task_status(), "msg": request.query_params.get("msg"), "err": request.query_params.get("err"),
            "tools": sorted(toolbox.REGISTRY.values(), key=lambda t: (t.agent, t.name)), "channels": channels, "already": already, "states": states, "counts": counts,
            "bulk_task": tasks.task_status().get(bulk.TASK), "bulk_running": tasks.is_running(bulk.TASK),
            "bulk_interrupted": bulk.interrupted(), "queue": bulk.status(), "queued": set(bulk.queued_ids()),
            "latex_engine": (latex.engine() or [None])[0] if latex.available() else None,
        })


@router.get("/applications/{pkg_id}", response_class=HTMLResponse)
def application_detail(request: Request, pkg_id: int):
    with session_scope() as s:
        pkg = s.get(ApplicationPackage, pkg_id)
        if pkg is None:
            raise HTTPException(404, "Application package not found")
        job = s.get(Job, pkg.job_id)
        letter = agent.package_letter(s, pkg.job_id, pkg.letter_group_id, pkg.reviews or [])
        task_key = agent.task_name(pkg.job_id)
        if pkg.status == "running" and not tasks.is_running(task_key) and tasks.is_running(bulk.TASK):
            task_key = bulk.TASK  # built inside a bulk run
        task = tasks.task_status().get(task_key, {})
        running = pkg.status == "running" and tasks.is_running(task_key)
        if pkg.status == "running" and not running:  # the run died with the app (restart) or failed before saving
            pkg.status, pkg.error = "failed", task.get("error") or "Interrupted — the app was restarted during the run."
        return templates.TemplateResponse(request, "application.html", {
            "request": request, "pkg": pkg, "job": job, "letter": letter, "task": task, "running": running,
            "review": agent.best_review(pkg.reviews or []), "cv_text": render_text(pkg.cv) if pkg.cv else "",
            "app_prefs": get_prefs().application, "now": utcnow(), "documents": agent.available_documents(),
            "task_key": task_key, "send": submit.preview(pkg_id) if pkg.status == "done" and pkg.cv else None,
            "applied_reason": submit.applied_reason(s, job) if job is not None else None,
            "documents_dir": str(agent.documents_dir()),
            "rules_count": len(__import__("app.services.agent_memory", fromlist=["x"]).load_rules()),
            "msg": request.query_params.get("msg"), "err": request.query_params.get("err"),
        })


@router.get("/applications/{pkg_id}/file/{kind}")
def application_file(pkg_id: int, kind: str):
    if kind not in FILE_KINDS:
        raise HTTPException(404, "Unknown file")
    with session_scope() as s:
        pkg = s.get(ApplicationPackage, pkg_id)
        if pkg is None:
            raise HTTPException(404, "Application package not found")
        try:
            data, filename, media = agent.package_file(s, pkg, kind)
        except agent.ApplicationError as e:
            raise HTTPException(404, str(e)) from e
    disposition = "inline" if kind.endswith(".pdf") else "attachment"
    return Response(data, media_type=media, headers={"Content-Disposition": f'{disposition}; filename="{filename}"'})


@router.post("/applications/bulk")
async def applications_bulk(request: Request):
    """Run the agents for the selected jobs. mode=prepare (nothing is sent) or mode=send (sends SEND-verdict packages
    that have an application e-mail address — only because you pressed this button and confirmed)."""
    form = await request.form()
    ids = [int(x) for x in form.getlist("job_ids") if str(x).isdigit()][:50]
    mode = form.get("mode", "prepare")
    if not ids:
        return _redirect("/applications#bulk", err="Select at least one job")
    if mode == "send" and form.get("confirm_send") != "yes":
        return _redirect("/applications#bulk", err="Sending needs the confirmation tick")
    language = form.get("language") if form.get("language") in ("de", "en") else None
    r = bulk.enqueue(ids, language=language, send=mode == "send")
    bulk.start()
    what = "prepare and send" if mode == "send" else "prepare"
    extra = f" · {r['already_queued']} already queued" if r["already_queued"] else ""
    return _redirect("/applications#queue", msg=f"Added {r['added']} job(s) to the preparation queue ({what}){extra}")


@router.post("/applications/bulk/resume")
def applications_bulk_resume():
    """Continue a paused / stopped queue."""
    if not bulk.saved_state()["items"]:
        return _redirect("/applications#queue", err="The queue is empty")
    bulk._update(lambda st: st.update(paused=None))
    started = bulk.start()
    return _redirect("/applications#queue", msg="The queue continues" if started else "The queue is already running")


@router.post("/applications/queue/clear")
def applications_queue_clear():
    n = bulk.clear()
    return _redirect("/applications#queue", msg=f"Removed {n} waiting job(s) from the queue (the one being prepared finishes)")


@router.post("/applications/queue/{job_id}/{action}")
def applications_queue_edit(request: Request, job_id: int, action: str):
    """Reorder or skip a waiting job: up | down | top | remove."""
    if action == "remove":
        ok = bulk.remove(job_id)
    elif action in ("up", "down", "top"):
        ok = bulk.move(job_id, action)
    else:
        raise HTTPException(400, "Unknown queue action")
    if request.headers.get("hx-request") == "true":
        return JSONResponse(bulk.status())
    return _redirect("/applications#queue", err="" if ok else "That job is not waiting in the queue any more")


@router.post("/jobs/{job_id}/prepare")
def job_prepare(request: Request, job_id: int, language: Annotated[str, Form()] = "auto"):
    """The "Prepare" button: queue this job for the agents (prepare only — nothing is sent)."""
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        title = job.title
    r = bulk.enqueue([job_id], language=language if language in ("de", "en") else None)
    bulk.start()
    position = bulk.queued_ids().index(job_id) + 1 if job_id in bulk.queued_ids() else None
    if request.headers.get("hx-request") == "true":
        label = "being prepared" if position == 1 and tasks.is_running(bulk.TASK) else f"queued #{position}"
        return HTMLResponse(f'<a class="btn small queued" href="/applications#queue" title="Open the preparation queue">'
                            f'{"✓ " if r["added"] else ""}{label}</a>')
    msg = f"“{title}” is in the preparation queue (#{position})" if position else "Queued"
    return RedirectResponse((request.headers.get("referer") or "/applications").split("#")[0] + f"?msg={msg}", status_code=303)


@router.post("/applications/{pkg_id}/send")
def application_send(pkg_id: int, confirm: Annotated[str, Form()] = "", allow_revise: Annotated[str, Form()] = ""):
    if confirm != "yes":
        return _redirect(f"/applications/{pkg_id}#send", err="Tick the confirmation to send")
    try:
        sent = submit.send_application(pkg_id, allow_revise=allow_revise == "1")
    except submit.SubmitError as e:
        return _redirect(f"/applications/{pkg_id}#send", err=str(e))
    except Exception as e:  # noqa: BLE001 - SMTP problems are shown, not raised
        return _redirect(f"/applications/{pkg_id}#send", err=f"Sending failed: {e}")
    return _redirect(f"/applications/{pkg_id}#send", msg=f"Application sent to {sent['to']} (copy in your mailbox)")


@router.post("/applications/{pkg_id}/revise")
def application_revise(pkg_id: int, feedback: Annotated[str, Form()] = "", learn: Annotated[str, Form()] = ""):
    """Your feedback → a new round of this application (letter + CV revised, reviewed again); lasting preferences are
    learned for every future application (agent_memory)."""
    if len(feedback.strip()) < 3:
        return _redirect(f"/applications/{pkg_id}#revise", err="Write what the agent should change")
    with session_scope() as s:
        pkg = s.get(ApplicationPackage, pkg_id)
        if pkg is None:
            raise HTTPException(404, "Application package not found")
        if pkg.status != "done" or not pkg.cv:
            return _redirect(f"/applications/{pkg_id}#revise", err="Only a finished application can be revised")
        job_id = pkg.job_id
    if agent.busy(job_id):
        return _redirect(f"/applications/{pkg_id}#revise", err="The agent is already working on this job")
    if get_llm().target(Role.WRITER) is None:
        return _redirect(f"/applications/{pkg_id}#revise", err="No writer model configured")
    tasks.run_in_background(agent.task_name(job_id),
                            lambda: agent.revise_with_feedback(pkg_id, feedback, learn=learn == "1"))
    return _redirect(f"/applications/{pkg_id}", msg="Revising with your feedback — this page updates by itself (2–6 minutes).")


@router.get("/agent-memory", response_class=HTMLResponse)
def agent_memory_page(request: Request):
    from app.services import agent_memory

    return templates.TemplateResponse(request, "agent_memory.html", {
        "request": request, "rules": agent_memory.load_rules(), "path": str(agent_memory.memory_path())})


@router.post("/agent-memory/add")
def agent_memory_add(text: Annotated[str, Form()] = "", applies_to: Annotated[str, Form()] = "all",
                     cv_exclude: Annotated[str, Form()] = ""):
    from app.services import agent_memory

    if text.strip():
        agent_memory.add_rules([{"rule": text, "applies_to": applies_to if applies_to in ("letter", "cv", "all") else "all",
                                 "cv_exclude": [t.strip() for t in cv_exclude.split(",") if t.strip()]}],
                               learned_from="added by you")
    return _redirect("/agent-memory", msg="Rule added")


@router.post("/agent-memory/{rule_id}/update")
def agent_memory_update(rule_id: int, text: Annotated[str, Form()] = "", applies_to: Annotated[str, Form()] = "all"):
    from app.services import agent_memory

    agent_memory.update_rule(rule_id, text, applies_to)
    return _redirect("/agent-memory", msg="Rule saved")


@router.post("/agent-memory/{rule_id}/delete")
def agent_memory_delete(rule_id: int):
    from app.services import agent_memory

    agent_memory.delete_rule(rule_id)
    return _redirect("/agent-memory", msg="Rule deleted — the agent no longer follows it")


@router.post("/applications/{pkg_id}/mark-applied")
def application_mark_applied(pkg_id: int):
    try:
        submit.mark_applied(pkg_id)
    except submit.SubmitError as e:
        raise HTTPException(404, str(e)) from e
    return _redirect(f"/applications/{pkg_id}#send", msg="Marked as applied")


@router.post("/applications/{pkg_id}/delete")
def application_delete(pkg_id: int):
    with session_scope() as s:
        pkg = s.get(ApplicationPackage, pkg_id)
        if pkg is None:
            raise HTTPException(404, "Application package not found")
        if pkg.status == "running" and tasks.is_running(agent.task_name(pkg.job_id)):
            return _redirect(f"/applications/{pkg_id}", err="Still running")
        job_id = pkg.job_id
        exported = (pkg.meta or {}).get("export", {}).get("dir")
        s.delete(pkg)  # the cover letter versions stay on the job page
    if exported:
        import shutil

        from app.services.export import export_root

        shutil.rmtree(export_root() / exported, ignore_errors=True)
    return _redirect(f"/jobs/{job_id}#apply", msg="Application package deleted (its cover letter is kept)")
