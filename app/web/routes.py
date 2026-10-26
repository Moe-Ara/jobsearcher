"""HTML dashboard routes (server-rendered, progressive enhancement with htmx)."""

from __future__ import annotations

from typing import Annotated, Any
from urllib.parse import urlencode

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import func, nulls_last, select
from sqlalchemy.orm import selectinload

from app.config_files import config_store, get_prefs, get_profile
from app.core.normalize import canonical_country, normalize_company
from app.db import session_scope
from app.email_ingest.service import process_eml_bytes
from app.enums import ACTIONS, RELEVANCE_REASONS, CoverLetterStrategy, FilterStatus, JobStatus, RelevanceLabel
from app.llm.factory import Role, get_llm, reset_llm
from app.models import EmailMessage, Job, JobAnalysis, SourceRun, TargetCompany, utcnow
from app.scheduler import scheduled_jobs
from app.services import analytics, tasks
from app.services.analysis import analyze_job, analyze_selected, latest_screening, mark_all_for_reanalysis, rescore_all, usage_summary
from app.services.humanizer import humanize, text_language
from app.services.letter_templates import (
    delete_template,
    get_template,
    list_templates,
    save_template,
    select_templates,
    templates_dir,
)
from app.services.application_agent import packages_for
from app.services.application_notes import FIELD_LABELS, get_notes, get_strategy, save_notes, set_strategy
from app.services.letters import detect_language, finish_letter, fit_from_analysis, generate_letter, letter_groups, run_editor, run_humanizer
from app.services.calibration import CalibrationQuery, calibration_candidates, labeled_job_ids, ranking_metrics, set_relevance
from app.services.ingest import invalidate_alias_cache
from app.services.jobs import rebuild_search_text, set_status
from app.services.manual import ManualIngestError, add_company_from_url, ingest_manual
from app.services.queries import JobFilters, dashboard_stats, filter_options, search_jobs
from app.settings import get_settings, reload_settings
from app.sources.registry import ATS_TYPES, detect_ats
from app.web.templating import templates

router = APIRouter(include_in_schema=False)


NOTE_PLACEHOLDERS = {
    "why_interested": "e.g. AI applied to industrial systems, not another chatbot — reminds me of my thesis",
    "why_fit": "e.g. thesis: classifiers + LLM planning in one pipeline; last job: Java, Python, testing, Jenkins",
    "emphasize": "e.g. AI thesis, software engineering experience, Docker/Jenkins",
    "avoid": "e.g. they ask for Azure — I have basically none, don't pretend. Don't make me sound senior.",
}

STRATEGY_LABELS = [
    (CoverLetterStrategy.NONE.value, "None yet"),
    (CoverLetterStrategy.AI_WITHOUT_PERSONAL_INPUT.value, "AI, without personal notes"),
    (CoverLetterStrategy.PERSONAL_INPUT_AI_EDITED.value, "My notes, written up by AI"),
    (CoverLetterStrategy.MANUAL.value, "Written by me"),
]


def _redirect(path: str, msg: str = "", err: str = "") -> RedirectResponse:
    params = {k: v for k, v in (("msg", msg), ("err", err)) if v}
    path, _, fragment = path.partition("#")  # the query must come before the #fragment
    if fragment == "letters" and params:
        params["focus"] = "letters"  # the job page shows this message inside the letters section, where you land
    sep = "&" if "?" in path else "?"
    url = f"{path}{sep}{urlencode(params)}" if params else path
    return RedirectResponse(f"{url}#{fragment}" if fragment else url, status_code=303)


def _is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request") == "true"


def _load_job(s, job_id: int) -> Job:
    job = s.scalars(
        select(Job).where(Job.id == job_id).options(
            selectinload(Job.sources), selectinload(Job.current_analysis), selectinload(Job.status_events),
            selectinload(Job.target_company),
        )
    ).first()
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


# --------------------------------------------------------------------------- dashboard


def queue_positions() -> dict[int, int]:
    """job id → position in the preparation queue (1 = being prepared / next)."""
    from app.services import bulk

    return {job_id: i for i, job_id in enumerate(bulk.queued_ids(), 1)}


def apply_panel(s, job: Job) -> dict[str, Any]:
    """Everything the job page's Apply panel needs: queue position, the newest finished package, where to apply."""
    from app.services import bulk, submit

    pkgs = packages_for(s, job.id)
    latest = next((p for p in pkgs if p.status == "done" and p.cv), None)
    return {"queue_pos": queue_positions().get(job.id), "queue_running": tasks.is_running(bulk.TASK),
            "latest_pkg": latest, "apply_channel": submit.channel(job), "applied_reason": submit.applied_reason(s, job)}


def latest_packages(s, job_ids: list[int]) -> dict[int, Any]:
    """job id → its newest application package (for the badge on the job cards)."""
    from app.models import ApplicationPackage

    out: dict[int, Any] = {}
    if job_ids:
        for p in s.scalars(select(ApplicationPackage).where(ApplicationPackage.job_id.in_(job_ids))
                           .order_by(ApplicationPackage.id)):
            out[p.job_id] = p
    return out


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    prefs = get_prefs()
    params = dict(request.query_params)
    f = JobFilters.from_params(params, prefs.dashboard.default_since_days)
    with session_scope() as s:
        jobs, total = search_jobs(s, f, prefs.dashboard.page_size)
        ctx = {
            "request": request,
            "jobs": jobs,
            "pkgs": latest_packages(s, [j.id for j in jobs]), "queued": queue_positions(),
            "total": total,
            "f": f,
            "params": params,
            "stats": dashboard_stats(s, f.since_days),
            "options": filter_options(s),
            "page_size": prefs.dashboard.page_size,
            "llm": get_llm(),
            "analysis_running": tasks.is_running("analysis"),
        }
        return templates.TemplateResponse(request, "dashboard.html", ctx)


@router.get("/side-jobs", response_class=HTMLResponse)
def side_jobs_page(request: Request):
    """Part-time / mini-job / temporary work (job.track == "side"), ranked by the side-job rules."""
    p = request.query_params
    f = {"q": p.get("q", "").strip(), "kind": p.get("kind", ""), "sort": p.get("sort", "score"),
         "max_min": int(p["max_min"]) if p.get("max_min", "").isdigit() else None, "all": p.get("all") == "1",
         "hidden": p.get("hidden") == "1"}
    limit = 200
    with session_scope() as s:
        base = [Job.track == "side", Job.active.is_(True)]
        visible = base + ([] if f["hidden"] else [Job.filter_status != FilterStatus.REJECTED])
        stmt = select(Job).where(*visible)
        if not f["all"]:
            stmt = stmt.where(Job.status.in_([JobStatus.NEW, JobStatus.PRIORITY, JobStatus.REVIEW]))
        kind_cond = {
            "minijob": Job.employment_type.contains("Minijob"), "teilzeit": Job.employment_type.contains("Teilzeit"),
            "befristet": Job.employment_type.contains("befristet"), "remote": Job.remote_status == "remote",
        }
        if f["kind"] in kind_cond:
            stmt = stmt.where(kind_cond[f["kind"]])
        if f["q"]:
            for term in f["q"].lower().split():
                stmt = stmt.where(Job.search_text.contains(term, autoescape=True))
        order = {"newest": [Job.first_seen.desc()]}.get(f["sort"], [nulls_last(Job.fit_score.desc()), Job.first_seen.desc()])
        jobs = list(s.scalars(stmt.order_by(*order).limit(1000)))
        dist = lambda j: (j.raw_source_data or {}).get("drive_minutes")  # noqa: E731 - minutes by car
        if f["kind"] == "near":
            jobs = [j for j in jobs if dist(j) is not None and dist(j) <= 20]
        if f["max_min"] is not None:
            jobs = [j for j in jobs if j.remote_status == "remote" or (dist(j) is not None and dist(j) <= f["max_min"])]
        if f["sort"] == "distance":
            jobs.sort(key=lambda j: (dist(j) is None, dist(j) if dist(j) is not None else 999, -(j.fit_score or 0)))
        jobs = jobs[:limit]
        everything = list(s.scalars(select(Job).where(*base, Job.filter_status != FilterStatus.REJECTED)))
        stats = {
            "total": len(everything), "minijob": sum("Minijob" in (j.employment_type or "") for j in everything),
            "teilzeit": sum("Teilzeit" in (j.employment_type or "") for j in everything),
            "befristet": sum("befristet" in (j.employment_type or "") for j in everything),
            "remote": sum(j.remote_status == "remote" for j in everything),
            "near": sum(dist(j) is not None and dist(j) <= 20 for j in everything),
        }
        return templates.TemplateResponse(request, "side_jobs.html", {
            "request": request, "jobs": jobs, "f": f, "stats": stats, "limit": limit, "prefs": get_prefs().side_jobs,
            "home": __import__("app.services.drive_time", fromlist=["x"]).home_point()[0] or "home"})


@router.get("/internships", response_class=HTMLResponse)
def internships_page(request: Request):
    """Internships and trainee programmes (job.track == "intern"): your field, near home or remote, open to graduates."""
    from collections import Counter

    from app.services.drive_time import home_point
    from app.services.internships import PENDING_RULE

    p = request.query_params
    f = {"q": p.get("q", "").strip(), "sort": p.get("sort", "score"), "remote": p.get("remote") == "1",
         "max_min": int(p["max_min"]) if p.get("max_min", "").isdigit() else None, "all": p.get("all") == "1",
         "hidden": p.get("hidden") == "1"}
    prefs = get_prefs()
    with session_scope() as s:
        every = list(s.scalars(select(Job).where(Job.track == "intern", Job.active.is_(True))))
        waiting = [j for j in every if any(r.get("rule") == PENDING_RULE for r in j.filter_reasons or [])]
        shown = [j for j in every if j.filter_status != FilterStatus.REJECTED and j not in waiting]
        rejected = [j for j in every if j.filter_status == FilterStatus.REJECTED]
        labels = {"intern_field": "not your field", "intern_distance": "too far", "intern_location": "not near you",
                  "intern_enrollment": "require enrollment", "intern_format": "Werkstudent / school / FSJ formats",
                  "defence": "defence",
                  "nationals_only": "nationals only"}
        why = Counter(next((labels.get(r["rule"], "other") for r in j.filter_reasons or [] if r.get("severity") == "reject"), "other")
                      for j in rejected)
        jobs = rejected if f["hidden"] else shown
        if not f["all"]:
            jobs = [j for j in jobs if j.status in (JobStatus.NEW, JobStatus.PRIORITY, JobStatus.REVIEW)]
        if f["q"]:
            terms = f["q"].lower().split()
            jobs = [j for j in jobs if all(t in (j.search_text or "") for t in terms)]
        dist = lambda j: (j.raw_source_data or {}).get("drive_minutes")  # noqa: E731 - minutes by car
        if f["remote"]:
            jobs = [j for j in jobs if j.remote_status == "remote"]
        if f["max_min"] is not None:
            jobs = [j for j in jobs if j.remote_status == "remote" or (dist(j) is not None and dist(j) <= f["max_min"])]
        if f["sort"] == "distance":
            jobs.sort(key=lambda j: (dist(j) is None, dist(j) or 0, -(j.fit_score or 0)))
        elif f["sort"] == "newest":
            jobs.sort(key=lambda j: j.first_seen, reverse=True)
        elif f["hidden"]:  # the near misses first: in your field, but needs enrollment / a bit too far
            order = {"intern_enrollment": 0, "intern_distance": 1, "intern_location": 2, "intern_field": 4, "intern_format": 5}
            rule = lambda j: next((r["rule"] for r in j.filter_reasons or [] if r.get("severity") == "reject"), "")  # noqa: E731
            jobs.sort(key=lambda j: (order.get(rule(j), 3), dist(j) or 999))
        else:
            jobs.sort(key=lambda j: (j.fit_score is None, -(j.fit_score or 0), -j.first_seen.timestamp()))
        stats = {"shown": len(shown), "good": sum((j.fit_score or 0) >= prefs.scoring.thresholds.get("review", 6.5) for j in shown),
                 "near": sum(dist(j) is not None and dist(j) <= 20 for j in shown),
                 "remote": sum(j.remote_status == "remote" for j in shown),
                 "pending_analysis": sum(j.fit_score is None for j in shown), "waiting": len(waiting),
                 "hidden": len(rejected), "why": why.most_common()}
        return templates.TemplateResponse(request, "internships.html", {
            "request": request, "jobs": jobs[:200], "f": f, "stats": stats, "ip": prefs.internships,
            "home": home_point(prefs)[0] or "home", "pkgs": latest_packages(s, [j.id for j in jobs[:200]]),
            "queued": queue_positions()})


@router.post("/jobs/{job_id}/action/{action}", response_class=HTMLResponse)
def job_action(request: Request, job_id: int, action: str):
    if action not in ACTIONS:
        raise HTTPException(400, "Unknown action")
    response = None
    with session_scope() as s:
        job = _load_job(s, job_id)
        target = ACTIONS[action]
        new_status = JobStatus.NEW if job.status == target else target  # click again = undo
        set_status(s, job, new_status, note="dashboard")
        s.flush()
        if _is_htmx(request):
            card = "partials/side_card.html" if job.track == "side" else "partials/job_card.html"
            response = templates.TemplateResponse(request, card, {"request": request, "job": job, "queued": queue_positions()})
    return response or RedirectResponse(request.headers.get("referer") or "/", status_code=303)


@router.post("/jobs/{job_id}/notes", response_class=HTMLResponse)
def job_notes(request: Request, job_id: int, notes: Annotated[str, Form()] = "", cv_version: Annotated[str, Form()] = ""):
    with session_scope() as s:
        job = _load_job(s, job_id)
        job.notes = notes.strip()[:20000] or None
        if cv_version or job.cv_version:
            job.cv_version = cv_version.strip()[:120] or None
        rebuild_search_text(job)
        if _is_htmx(request):
            return HTMLResponse('<span class="saved">Saved ✓</span>')
    return _redirect(f"/jobs/{job_id}", msg="Notes saved")


# --------------------------------------------------------------------------- job detail


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_detail(request: Request, job_id: int):
    with session_scope() as s:
        job = _load_job(s, job_id)
        history = s.scalars(
            select(JobAnalysis).where(JobAnalysis.job_id == job_id).order_by(JobAnalysis.created_at.desc()).limit(25)
        ).all()
        screen_rec, screening = latest_screening(s, job_id)
        llm = get_llm()
        task = tasks.task_status().get(f"letter-{job_id}", {})
        lang, lang_reason = detect_language(job)
        return templates.TemplateResponse(request, "job.html", {
            "request": request, "job": job, "a": (job.current_analysis.result if job.current_analysis else None),
            "analysis": job.current_analysis, "history": history, "llm": llm,
            "screening": screening, "screen_rec": screen_rec,
            "letters": letter_groups(s, job_id), "letter_running": tasks.is_running(f"letter-{job_id}"),
            "packages": packages_for(s, job_id), "apply_running": tasks.is_running(f"apply-{job_id}"),
            **apply_panel(s, job),
            "app_prefs": get_prefs().application,
            "letter_error": task.get("error") if not task.get("running") else None,
            "letter_task": task, "now": utcnow(),
            "analyze_error": tasks.task_status().get(f"analyze-{job_id}", {}).get("error"),
            "detected_language": lang, "language_reason": lang_reason,
            "writer": llm.target(Role.WRITER), "editor": llm.target(Role.EDITOR),
            "german_editor_default": get_prefs().writing.german_editor_by_default,
            "humanizer": llm.target(Role.HUMANIZER), "humanizer_default": get_prefs().writing.humanizer_by_default,
            "templates": list_templates(), "templates_enabled": get_prefs().writing.use_templates,
            "auto_templates": select_templates(job, lang, get_prefs().writing.templates_per_letter),
            "fit": fit_from_analysis(job) if get_prefs().writing.use_fit_analysis else None,
            "notes": get_notes(s, job_id), "note_labels": FIELD_LABELS, "note_placeholders": NOTE_PLACEHOLDERS,
            "strategy": get_strategy(s, job_id), "strategy_labels": STRATEGY_LABELS,
            "focus_letters": request.query_params.get("focus") == "letters",
            "flash_msg": request.query_params.get("msg"), "flash_err": request.query_params.get("err"),
            "msg": request.query_params.get("msg"), "err": request.query_params.get("err"),
        })


@router.post("/jobs/{job_id}/status")
def job_status(job_id: int, status: Annotated[str, Form()], note: Annotated[str, Form()] = ""):
    if status not in JobStatus.__members__:
        raise HTTPException(400, "Unknown status")
    with session_scope() as s:
        job = _load_job(s, job_id)
        set_status(s, job, JobStatus(status), note=note.strip()[:500] or None)  # never triggers generation
    return _redirect(f"/jobs/{job_id}", msg=f"Status set to {status}")


# --------------------------------------------------------------------------- application material


def _notes_from_form(why_interested: str | None, why_fit: str | None, emphasize: str | None, avoid: str | None) -> dict:
    """Only fields the form actually sent (None = not on the form) are written."""
    sent = {"why_interested": why_interested, "why_fit": why_fit, "emphasize": emphasize, "avoid": avoid}
    return {k: v for k, v in sent.items() if v is not None}


@router.post("/jobs/{job_id}/application-notes")
def application_notes_save(job_id: int, why_interested: Annotated[str | None, Form()] = None,
                           why_fit: Annotated[str | None, Form()] = None, emphasize: Annotated[str | None, Form()] = None,
                           avoid: Annotated[str | None, Form()] = None):
    with session_scope() as s:
        _load_job(s, job_id)
        save_notes(s, job_id, **_notes_from_form(why_interested, why_fit, emphasize, avoid))
    return _redirect(f"/jobs/{job_id}#letters", msg="Personal application notes saved")


@router.post("/jobs/{job_id}/letter-strategy")
def letter_strategy_set(job_id: int, strategy: Annotated[str, Form()]):
    if strategy not in CoverLetterStrategy.__members__:
        raise HTTPException(400, "Unknown strategy")
    with session_scope() as s:
        _load_job(s, job_id)
        set_strategy(s, job_id, strategy)
    return _redirect(f"/jobs/{job_id}#letters", msg="Letter strategy saved")


@router.post("/jobs/{job_id}/letters/generate")
def letter_generate(job_id: int, language: Annotated[str, Form()] = "auto", use_editor: Annotated[str, Form()] = "",
                    use_humanizer: Annotated[str, Form()] = "", template: Annotated[str, Form()] = "auto",
                    why_interested: Annotated[str | None, Form()] = None, why_fit: Annotated[str | None, Form()] = None,
                    emphasize: Annotated[str | None, Form()] = None, avoid: Annotated[str | None, Form()] = None):
    if tasks.is_running(f"apply-{job_id}"):
        return _redirect(f"/jobs/{job_id}#letters", err="The application agent is working on this job's letter — wait until it has finished")
    lang = language if language in ("de", "en") else None
    notes_sent = _notes_from_form(why_interested, why_fit, emphasize, avoid)
    if notes_sent:  # the notes typed above the Generate buttons are saved first, so the letter uses them
        with session_scope() as s:
            _load_job(s, job_id)
            save_notes(s, job_id, **notes_sent)
    editor_flag, humanizer_flag = use_editor == "1", use_humanizer == "1"
    choice = template if template in ("auto", "none") or get_template(template) else "auto"
    started = tasks.run_in_background(
        f"letter-{job_id}",
        lambda: generate_letter(job_id, language=lang, use_editor=editor_flag, use_humanizer=humanizer_flag,
                                template=choice),
    )
    return _redirect(f"/jobs/{job_id}#letters", msg="Generating cover letter…" if started else "A letter is already being generated")


@router.post("/jobs/{job_id}/letters/{document_id}/edit")
def letter_edit(job_id: int, document_id: int):
    if tasks.is_running(f"apply-{job_id}"):
        return _redirect(f"/jobs/{job_id}#letters", err="The application agent is working on this job's letter — wait until it has finished")
    started = tasks.run_in_background(f"letter-{job_id}", lambda: run_editor(job_id, document_id))
    return _redirect(f"/jobs/{job_id}#letters", msg="Running German style editor…" if started else "A letter task is already running")


@router.post("/jobs/{job_id}/letters/group/{group_id}/finish")
def letter_finish(job_id: int, group_id: str):
    if tasks.is_running(f"apply-{job_id}"):
        return _redirect(f"/jobs/{job_id}#letters", err="The application agent is working on this job's letter — wait until it has finished")
    if not group_id.isalnum() or len(group_id) > 32:
        raise HTTPException(400, "Bad letter version")
    started = tasks.run_in_background(f"letter-{job_id}", lambda: finish_letter(job_id, group_id))
    return _redirect(f"/jobs/{job_id}#letters", msg="Finishing this version…" if started else "A letter task is already running")


@router.post("/jobs/{job_id}/letters/{document_id}/humanize")
def letter_humanize(job_id: int, document_id: int):
    if tasks.is_running(f"apply-{job_id}"):
        return _redirect(f"/jobs/{job_id}#letters", err="The application agent is working on this job's letter — wait until it has finished")
    started = tasks.run_in_background(f"letter-{job_id}", lambda: run_humanizer(job_id, document_id))
    return _redirect(f"/jobs/{job_id}#letters", msg="Running humanizer…" if started else "A letter task is already running")


# --------------------------------------------------------------------------- cover-letter templates


@router.get("/templates", response_class=HTMLResponse)
def templates_page(request: Request, edit: str = ""):
    items = list_templates()
    return templates.TemplateResponse(request, "letter_templates.html", {
        "request": request, "items": items, "editing": get_template(edit) if edit else None,
        "enabled": get_prefs().writing.use_templates, "per_letter": get_prefs().writing.templates_per_letter,
        "folder": str(templates_dir()),
    })


@router.post("/templates")
def templates_save(title: Annotated[str, Form()] = "", text: Annotated[str, Form()] = "",
                   language: Annotated[str, Form()] = "auto", tags: Annotated[str, Form()] = "",
                   notes: Annotated[str, Form()] = "", template_id: Annotated[str, Form()] = ""):
    try:
        t = save_template(title, text, language=language, tags=tags, notes=notes, template_id=template_id or None)
    except ValueError as e:
        return _redirect("/templates", err=str(e))
    return _redirect(f"/templates#t-{t.id}", msg=f"Template “{t.title}” saved ({t.body_words} words, {len(t.paragraphs)} paragraphs)")


@router.post("/templates/{template_id}/delete")
def templates_delete(template_id: str):
    ok = delete_template(template_id)
    return _redirect("/templates", msg="Template deleted" if ok else "", err="" if ok else "Template not found")


# --------------------------------------------------------------------------- standalone humanizer

HUMANIZE_MAX_CHARS = 20000
_humanize_pending: dict = {}  # last submitted input, so the form keeps it while the rewrite runs / after an error


def _humanize_task(text: str, keep_frame: bool) -> dict:
    with session_scope() as s:
        r = humanize(text, keep_letter_frame=keep_frame, session=s)
    return {"input": text, "keep_frame": keep_frame, "text": r.text, "model": r.model, "chunks": r.chunks,
            "warnings": r.warnings, "duration_ms": r.duration_ms, "language": text_language(text)}


@router.get("/humanize", response_class=HTMLResponse)
def humanize_page(request: Request):
    last = tasks.task_status().get("humanize", {})
    return templates.TemplateResponse(request, "humanize.html", {
        "request": request, "target": get_llm().target(Role.HUMANIZER), "running": tasks.is_running("humanize"),
        "result": last.get("result"), "error": last.get("error") if not last.get("running") else None,
        "pending": _humanize_pending, "max_chars": HUMANIZE_MAX_CHARS, "stage": last.get("stage"),
        "interrupted": last.get("interrupted"),
    })


@router.post("/humanize")
def humanize_submit(text: Annotated[str, Form()] = "", keep_frame: Annotated[str, Form()] = ""):
    text = text.strip()
    if not text:
        return _redirect("/humanize", err="Paste some text first")
    if len(text) > HUMANIZE_MAX_CHARS:
        return _redirect("/humanize", err=f"Text too long (max {HUMANIZE_MAX_CHARS} characters)")
    if get_llm().target(Role.HUMANIZER) is None:
        return _redirect("/humanize", err="Humanizer not configured (LOCAL_LLM_ENABLED + LOCAL_HUMANIZER_ENABLED)")
    started = tasks.run_in_background("humanize", lambda: _humanize_task(text, keep_frame == "1"))
    if started:
        _humanize_pending.update(text=text, keep_frame=keep_frame == "1")
    return _redirect("/humanize", msg="Humanizing… the page refreshes automatically" if started else "A rewrite is already running")


@router.post("/jobs/{job_id}/analyze")
def job_analyze(job_id: int):
    def work():
        with session_scope() as s:
            job = s.get(Job, job_id)
            if job:
                job.analysis_attempts = 0
                return analyze_job(s, job)

    tasks.run_in_background(f"analyze-{job_id}", work)
    return _redirect(f"/jobs/{job_id}", msg="Analysis started — refresh in a few seconds")


# --------------------------------------------------------------------------- add / ingest


@router.get("/add", response_class=HTMLResponse)
def add_page(request: Request):
    return templates.TemplateResponse(request, "add.html", {
        "request": request, "msg": request.query_params.get("msg"), "err": request.query_params.get("err"),
        "preview": None,
    })


@router.post("/add/job")
def add_job(
    url: Annotated[str, Form()] = "", title: Annotated[str, Form()] = "", company: Annotated[str, Form()] = "",
    location: Annotated[str, Form()] = "", description: Annotated[str, Form()] = "",
    analyze_now: Annotated[str, Form()] = "",
):
    try:
        res = ingest_manual(url=url, title=title, company=company, location=location, description=description)
    except ManualIngestError as exc:
        return _redirect("/add", err=str(exc))
    if analyze_now == "1":
        def work():
            with session_scope() as s:
                job = s.get(Job, res.job_id)
                return analyze_job(s, job) if job else None

        tasks.run_in_background(f"analyze-{res.job_id}", work)
    return _redirect(f"/jobs/{res.job_id}", msg=res.message + (" — analysing…" if analyze_now == "1" else ""))


@router.post("/add/company")
def add_company(
    careers_url: Annotated[str, Form()], name: Annotated[str, Form()] = "", priority: Annotated[str, Form()] = "B",
    country: Annotated[str, Form()] = "", city: Annotated[str, Form()] = "", sector: Annotated[str, Form()] = "",
):
    try:
        company_id, message = add_company_from_url(careers_url, name, priority, country, city, sector)
    except ManualIngestError as exc:
        return _redirect("/add", err=str(exc))
    from app.services.companies import check_company

    tasks.run_in_background(f"company-{company_id}", lambda: check_company(company_id))
    return _redirect("/companies", msg=f"{message}. Checking its jobs now…")


@router.post("/add/eml", response_class=HTMLResponse)
async def add_eml(request: Request, file: Annotated[UploadFile, File()], preview: Annotated[str, Form()] = ""):
    data = await file.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        return _redirect("/add", err="File too large (max 10 MB)")
    result = process_eml_bytes(data, dry_run=preview == "1")
    if preview == "1":
        return templates.TemplateResponse(request, "add.html", {"request": request, "preview": result, "msg": None, "err": None})
    if not result.get("parser"):
        return _redirect("/add", err="No parser matched this email's sender (see preferences.yaml → email).")
    if result.get("skipped"):
        return _redirect("/add", err="This email was already processed.")
    return _redirect("/", msg=f"Email parsed with {result['parser']}: {result.get('found', 0)} jobs, {result.get('new', 0)} new, {result.get('merged', 0)} merged")


# --------------------------------------------------------------------------- companies


@router.get("/companies", response_class=HTMLResponse)
def companies_page(request: Request):
    with session_scope() as s:
        rows = s.scalars(select(TargetCompany).order_by(TargetCompany.priority, TargetCompany.company_name)).all()
        counts = dict(s.execute(
            select(Job.target_company_id, func.count(Job.id)).where(Job.active.is_(True)).group_by(Job.target_company_id)
        ).all())
        edit_id = request.query_params.get("edit")
        return templates.TemplateResponse(request, "companies.html", {
            "request": request, "companies": rows, "counts": counts, "ats_types": ATS_TYPES,
            "edit_id": int(edit_id) if edit_id and edit_id.isdigit() else None,
            "msg": request.query_params.get("msg"), "err": request.query_params.get("err"),
        })


def _company_from_form(tc: TargetCompany, form) -> None:
    name = (form.get("company_name") or "").strip()
    if not name:
        raise ValueError("Company name is required")
    tc.company_name = name
    tc.normalized_name = normalize_company(name)
    tc.aliases = [a.strip() for a in (form.get("aliases") or "").split(",") if a.strip()]
    tc.country = canonical_country(form.get("country")) or (form.get("country") or "").strip() or None
    tc.city = (form.get("city") or "").strip() or None
    tc.sector = (form.get("sector") or "").strip() or None
    tc.priority = (form.get("priority") or "B")[:1].upper()
    tc.careers_url = (form.get("careers_url") or "").strip() or None
    ats_type = (form.get("ats_type") or "").strip()
    ident = (form.get("ats_identifier") or "").strip()
    if tc.careers_url and (not ats_type or ats_type == "auto"):
        m = detect_ats(tc.careers_url)
        ats_type, ident = m.ats_type, ident or (m.identifier or "")
        tc.ats_config = {**(tc.ats_config or {}), **(m.options or {})}
    tc.ats_type = ats_type if ats_type in ATS_TYPES else None
    tc.ats_identifier = ident or None
    pattern = (form.get("job_link_pattern") or "").strip()
    cfg = dict(tc.ats_config or {})
    if pattern:
        cfg["job_link_pattern"] = pattern
    else:
        cfg.pop("job_link_pattern", None)
    tc.ats_config = cfg
    tc.active = form.get("active") == "1"
    tc.notes = (form.get("notes") or "").strip() or None


@router.post("/companies")
async def company_create(request: Request):
    form = await request.form()
    with session_scope() as s:
        tc = TargetCompany(company_name="", normalized_name="", aliases=[], ats_config={})
        try:
            _company_from_form(tc, form)
        except ValueError as exc:
            return _redirect("/companies", err=str(exc))
        if s.scalars(select(TargetCompany).where(TargetCompany.normalized_name == tc.normalized_name)).first():
            return _redirect("/companies", err=f"{tc.company_name} already exists")
        s.add(tc)
    invalidate_alias_cache()
    return _redirect("/companies", msg=f"Added {tc.company_name}")


# Declared before "/companies/{company_id}", which would otherwise capture "import" (422).
@router.post("/companies/import")
def company_import():
    from app.services.companies import import_companies_yaml

    res = import_companies_yaml()
    return _redirect("/companies", msg=f"Imported {res['created']} new companies from companies.yaml")


@router.post("/companies/{company_id}")
async def company_update(request: Request, company_id: int):
    form = await request.form()
    with session_scope() as s:
        tc = s.get(TargetCompany, company_id)
        if tc is None:
            raise HTTPException(404)
        try:
            _company_from_form(tc, form)
        except ValueError as exc:
            return _redirect("/companies", err=str(exc))
    invalidate_alias_cache()
    return _redirect("/companies", msg="Saved")


@router.post("/companies/{company_id}/check")
def company_check(company_id: int):
    from app.services.companies import check_company

    started = tasks.run_in_background(f"company-{company_id}", lambda: check_company(company_id))
    return _redirect("/companies", msg="Check started — refresh in a moment" if started else "Already running")


@router.post("/companies/{company_id}/delete")
def company_delete(company_id: int):
    with session_scope() as s:
        tc = s.get(TargetCompany, company_id)
        if tc:
            s.delete(tc)  # jobs keep existing (FK set null); provenance is kept
    invalidate_alias_cache()
    return _redirect("/companies", msg="Company removed (its jobs are kept)")


# --------------------------------------------------------------------------- operations


@router.get("/ops", response_class=HTMLResponse)
def ops_page(request: Request):
    llm = get_llm()
    local = local_status(llm)
    with session_scope() as s:
        runs = s.scalars(select(SourceRun).order_by(SourceRun.started_at.desc()).limit(60)).all()
        emails = s.scalars(select(EmailMessage).order_by(EmailMessage.processed_at.desc()).limit(15)).all()
        latest_ids = select(func.max(SourceRun.id)).group_by(SourceRun.source_name)
        sources = s.scalars(select(SourceRun).where(SourceRun.id.in_(latest_ids)).order_by(SourceRun.source_name)).all()
        usage = usage_summary(s)
        split = {"local": {"calls": 0, "cost": 0.0}, "cloud": {"calls": 0, "cost": 0.0}}
        for r in usage["rows"]:
            if r["provider"] == "heuristic":
                continue
            bucket = split["local" if r["provider"] == "local" else "cloud"]
            bucket["calls"] += r["calls"] - r["cached"]
            bucket["cost"] = round(bucket["cost"] + r["cost"], 4)
        return templates.TemplateResponse(request, "ops.html", {
            "request": request, "runs": runs, "emails": emails, "usage": usage, "usage_split": split,
            "sources": sources, "profile_warnings": config_store.profile_warnings(),
            "task_status": tasks.task_status(), "scheduled": scheduled_jobs(), "llm": llm,
            "roles": [(r.value, llm.target(r)) for r in Role], "local": local, "imap": imap_status(),
            "settings": get_settings(), "profile": get_profile(), "prefs": get_prefs(),
            "msg": request.query_params.get("msg"), "err": request.query_params.get("err"),
        })


def local_status(llm) -> dict | None:
    """Small local-model status block: reachability, installed/loaded models, last success/error."""
    if llm.local is None:
        return None
    from app.llm.local_provider import HEALTH

    health = llm.local.health()
    models = sorted({t.model for t in llm.targets.values() if t and t.is_local and t.model})
    return {
        "base_url": llm.local.base_url,
        "health": health,
        "models": [{"name": m, "installed": llm.local.is_installed(m, health["installed"]) if health["reachable"] else None,
                    "loaded": any(x["name"] == m for x in health["loaded"])} for m in models],
        "state": HEALTH,
    }


_OPS_TASKS = {
    "companies_due": ("companies", tasks.task_check_due_companies, "Checking due companies"),
    "companies_all": ("companies", tasks.task_check_all_companies, "Checking all companies"),
    "email": ("email", tasks.task_poll_email, "Polling mailbox"),
    "analysis": ("analysis", tasks.task_analyze, "Analysing pending jobs"),
    "retention": ("retention", tasks.task_retention, "Deleting jobs older than the retention window"),
    "backup": ("backup", tasks.task_backup, "Backing up the database to data/backups"),
    "enrich": ("enrich", lambda: __import__("app.services.enrich", fromlist=["x"]).enrich_thin_jobs(limit=60),
               "Finding the full postings of snippet-only e-mail jobs"),
}


@router.post("/ops/mailbox-cleanup")
def ops_mailbox_cleanup(mode: Annotated[str, Form()] = "preview", confirm: Annotated[str, Form()] = ""):
    """Preview first; moving needs mode=run + the confirmation tick."""
    from app.email_ingest.cleanup import clean_seen_alerts

    run = mode == "run"
    if run and confirm != "yes":
        return _redirect("/ops#cleanup", err="Tick the confirmation to move the e-mails")
    try:
        r = clean_seen_alerts(dry_run=not run)
    except Exception as e:  # noqa: BLE001 - IMAP problems are shown, not raised
        return _redirect("/ops#cleanup", err=f"Mailbox clean-up failed: {e}")
    if not r.get("ok"):
        return _redirect("/ops#cleanup", err=r.get("error") or "Mailbox clean-up failed")
    where = ", ".join(f"{k}: {v}" for k, v in r["folders"].items()) or "none"
    if run:
        return _redirect("/ops#cleanup", msg=f"Moved {r['moved']} processed job-alert e-mails to {r['target']} ({where}).")
    return _redirect("/ops#cleanup", msg=f"Preview: {r['matched']} processed job-alert e-mails would be moved to "
                                         f"{r['target']} ({where}). Nothing has been changed yet.")


@router.post("/ops/purge-alert-junk")
def ops_purge_alert_junk():
    from app.email_ingest.junk import purge_alert_junk

    r = purge_alert_junk()
    return _redirect("/ops", msg=f"Removed {r['deleted']} non-jobs created from alert e-mail interface text"
                     + (f" (kept {r['kept_because_you_used_them']} you worked with)" if r["kept_because_you_used_them"] else ""))


@router.post("/ops/run/{task}")
def ops_run(task: str):
    if task not in _OPS_TASKS:
        raise HTTPException(400)
    name, fn, label = _OPS_TASKS[task]
    started = tasks.run_in_background(name, fn)
    return _redirect("/ops", msg=f"{label}…" if started else f"{name} is already running")


@router.post("/ops/imap/test")
def ops_imap_test():
    """Re-read .env from disk (no restart needed), test the IMAP login, and poll the mailbox if it works."""
    from app.email_ingest.service import check_imap

    reload_settings()
    reset_llm()
    status = check_imap()
    if not status["ok"]:
        return _redirect("/ops#mailbox", err=f"IMAP not connected: {status['error']}")
    started = tasks.run_in_background("email", tasks.task_poll_email)
    folders = ", ".join(status["folders"][:8]) + (" …" if len(status["folders"]) > 8 else "")
    return _redirect("/ops#mailbox", msg=f"IMAP connected ({len(status['folders'])} folders: {folders})."
                     + (" Polling the mailbox now…" if started else ""))


@router.post("/ops/reload")
def ops_reload():
    try:
        config_store.reload()
    except Exception as exc:  # noqa: BLE001 - surface YAML errors to the user
        return _redirect("/ops", err=f"Config error: {exc}")
    reset_llm()
    n = rescore_all()
    return _redirect("/ops", msg=f"Config reloaded; filters and scores recomputed for {n} jobs")


@router.post("/ops/rescore")
def ops_rescore():
    n = rescore_all()
    return _redirect("/ops", msg=f"Re-filtered and re-scored {n} jobs (no LLM calls)")


@router.post("/ops/reanalyze")
def ops_reanalyze():
    n = mark_all_for_reanalysis()
    tasks.run_in_background("analysis", tasks.task_analyze)
    return _redirect("/ops", msg=f"Queued {n} jobs for analysis (unchanged inputs are served from cache)")


# --------------------------------------------------------------------------- analytics


@router.get("/analytics/report.pdf")
def analytics_report(days: str = ""):
    """The analytics as a designed PDF report (charts, rates, packages, takeaways). ?days=30|90 or all time."""
    from app.services.analytics_report import build_report

    period = int(days) if days.isdigit() else None
    with session_scope() as s:
        pdf = build_report(s, period, name=get_prefs().writing.applicant_name or "")
    stamp = utcnow().strftime("%Y-%m-%d")
    suffix = f"_last{period}d" if period else ""
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="JobSearch_Report_{stamp}{suffix}.pdf"'})


@router.get("/analytics", response_class=HTMLResponse)
def analytics_page(request: Request):
    with session_scope() as s:
        return templates.TemplateResponse(request, "analytics.html", {"request": request, "data": analytics.summary(s)})


# --------------------------------------------------------------------------- relevance labels & calibration


@router.post("/jobs/{job_id}/relevance", response_class=HTMLResponse)
async def job_relevance(request: Request, job_id: int):
    form = await request.form()
    label = str(form.get("label") or "")
    if label not in RelevanceLabel.__members__:
        raise HTTPException(400, "Unknown label")
    with session_scope() as s:
        job = _load_job(s, job_id)
        set_relevance(job, label, [str(r) for r in form.getlist("reasons")])
        if _is_htmx(request):
            return templates.TemplateResponse(request, "partials/relevance.html", {"request": request, "job": job})
    return RedirectResponse(request.headers.get("referer") or f"/jobs/{job_id}", status_code=303)


def _calibration_query(params) -> CalibrationQuery:
    try:
        offset = max(0, int(params.get("offset") or 0))
    except ValueError:
        offset = 0
    return CalibrationQuery(
        unrated_only=params.get("unrated", "1") == "1",
        include_rejected=params.get("rejected", "0") == "1",
        order=params.get("order") if params.get("order") in ("shuffled", "score", "newest") else "shuffled",
        offset=offset,
    )


@router.get("/calibrate", response_class=HTMLResponse)
def calibrate_page(request: Request):
    q = _calibration_query(request.query_params)
    with session_scope() as s:
        job, remaining = calibration_candidates(s, q)
        return templates.TemplateResponse(request, "calibrate.html", {
            "request": request, "job": job, "remaining": remaining, "q": q,
            "a": job.current_analysis.result if job and job.current_analysis else None,
            "metrics": ranking_metrics(s), "reasons": RELEVANCE_REASONS,
            "analysis_running": tasks.is_running("analysis"),
            "last_run": tasks.task_status().get("calibration-analysis"),
            "msg": request.query_params.get("msg"), "err": request.query_params.get("err"),
        })


@router.post("/calibrate/label")
async def calibrate_label(request: Request):
    form = await request.form()
    q = _calibration_query(form)
    label = str(form.get("label") or "")
    job_id = int(form.get("job_id") or 0)
    if label == "SKIP":
        q.offset += 1
    elif label in RelevanceLabel.__members__:
        with session_scope() as s:
            set_relevance(_load_job(s, job_id), label, [str(r) for r in form.getlist("reasons")])
        if not q.unrated_only:  # the labeled job stays in the list → move on
            q.offset += 1
    params = {"unrated": "1" if q.unrated_only else "0", "rejected": "1" if q.include_rejected else "0",
              "order": q.order, "offset": q.offset}
    return RedirectResponse(f"/calibrate?{urlencode(params)}", status_code=303)


@router.post("/calibrate/analyze")
def calibrate_analyze():
    with session_scope() as s:
        ids = labeled_job_ids(s)
    if not ids:
        return _redirect("/calibrate", err="Label some jobs first")

    def work():
        return tasks.run_exclusive("analysis", lambda: analyze_selected(ids))  # shares the batch lock

    started = tasks.run_in_background("calibration-analysis", work)
    return _redirect("/calibrate", msg=f"Analysing {len(ids)} labeled jobs (screening, then deep analysis)…"
                     if started else "A calibration run is already in progress")


def imap_status() -> dict:
    from app.email_ingest.service import IMAP_STATUS

    return {**IMAP_STATUS, "running": tasks.is_running("imap-check") or tasks.is_running("email")}
