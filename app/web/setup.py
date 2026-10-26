"""Setup & settings pages: CV import, search preferences, personal details, the setup checklist, advanced editors."""

from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from app.config_files import get_prefs, get_profile
from app.services import config_writer, onboarding, tasks
from app.services.config_writer import ConfigError
from app.web.templating import templates

router = APIRouter(include_in_schema=False)

# Countries offered as tick boxes (any other country can be typed in).
COUNTRIES = ["Germany", "Austria", "Switzerland", "Luxembourg", "Netherlands", "Belgium", "France", "United Kingdom",
             "Ireland", "UAE", "Saudi Arabia", "Qatar"]
EDITABLE = ("profile.yaml", "cv.yaml", "preferences.yaml", "companies.yaml")


def _redirect(path: str, msg: str = "", err: str = "") -> RedirectResponse:
    from app.web.routes import _redirect as redirect

    return redirect(path, msg=msg, err=err)


def _lines(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"[\n,;]", text or "") if x.strip()]


@router.get("/setup", response_class=HTMLResponse)
def setup_page(request: Request):
    from app.services.drive_time import home_point

    prefs, profile = get_prefs(), get_profile()
    steps = onboarding.checklist()
    required = [s for s in steps if s["required"]]
    return templates.TemplateResponse(request, "setup.html", {
        "request": request, "steps": steps, "done": sum(s["done"] for s in required), "total": len(required),
        "prefs": prefs, "profile": profile, "countries": COUNTRIES, "home": home_point(prefs)[0],
        "drafts": onboarding.drafts(), "importing": tasks.is_running(onboarding.TASK),
        "import_state": tasks.task_status().get(onboarding.TASK) or {},
    })


@router.post("/setup/cv")
async def setup_cv(file: Annotated[UploadFile | None, File()] = None, text: Annotated[str, Form()] = ""):
    if tasks.is_running(onboarding.TASK):
        return _redirect("/setup#cv", err="A CV import is already running")
    try:
        if file is not None and file.filename:
            data = await file.read()
            if len(data) > 10_000_000:
                raise ValueError("The file is larger than 10 MB")
            cv_text = onboarding.extract_text(file.filename, data)
        else:
            cv_text = text
        if len(cv_text.strip()) < 200:
            raise ValueError("That is too little text for a CV — upload the file or paste the whole CV")
    except ValueError as exc:
        return _redirect("/setup#cv", err=str(exc))
    tasks.run_in_background(onboarding.TASK, lambda: onboarding.run_cv_import(cv_text))
    return _redirect("/setup#cv", msg="Reading your CV — this takes 1–3 minutes. The page updates when the draft is ready.")


@router.post("/setup/cv/save")
def setup_cv_save(profile_yaml: Annotated[str, Form()], cv_yaml: Annotated[str, Form()]):
    try:
        onboarding.save_drafts({"profile.yaml": profile_yaml, "cv.yaml": cv_yaml})
    except ConfigError as exc:
        return _redirect("/setup#cv", err=f"Not saved — please fix: {str(exc)[:600]}")
    from app.services.analysis import rescore_all

    rescore_all()
    return _redirect("/setup", msg="Profile and CV saved (previous versions are in config/.backups). New jobs are matched against them; "
                                   "use Operations → Re-analyse to re-check jobs you already have.")


@router.post("/setup/cv/discard")
def setup_cv_discard():
    onboarding.discard_drafts()
    return _redirect("/setup#cv", msg="Draft discarded")


@router.post("/setup/search")
def setup_search(
    countries: Annotated[list[str] | None, Form()] = None, other_countries: Annotated[str, Form()] = "",
    queries: Annotated[str, Form()] = "", home: Annotated[str, Form()] = "",
    max_years: Annotated[int, Form()] = 6, retention_days: Annotated[int, Form()] = 14,
    internships: Annotated[str, Form()] = "", intern_minutes: Annotated[int, Form()] = 45,
    side_jobs: Annotated[str, Form()] = "", side_minutes: Annotated[int, Form()] = 60,
):
    from app.services.drive_time import geocode, place_query

    targets = list(dict.fromkeys([*(countries or []), *_lines(other_countries)]))
    if not targets:
        return _redirect("/setup#search", err="Pick at least one country")
    changes: dict = {
        "locations.target_countries": targets,
        "search.queries": _lines(queries) or get_prefs().search.queries,
        "hard_filters.max_years_experience_required": max(0, min(max_years, 40)),
        "retention.max_job_age_days": max(3, min(retention_days, 90)),
        "internships.enabled": bool(internships), "internships.max_drive_minutes": max(5, min(intern_minutes, 240)),
        "side_jobs.enabled": bool(side_jobs), "side_jobs.max_drive_minutes": max(5, min(side_minutes, 240)),
    }
    note = ""
    home = home.strip()
    if home and home != (get_prefs().locations.home or get_prefs().side_jobs.home):
        q = place_query(home)
        point = geocode([q]).get(q) if q else None  # network: OpenStreetMap, one request
        changes.update({"locations.home": home, "locations.home_lat": point[0] if point else None,
                        "locations.home_lon": point[1] if point else None})
        note = f" Home located at {point[0]:.3f}, {point[1]:.3f}." if point else " (Could not find that town on the map — drive times stay off.)"
    try:
        config_writer.update("preferences.yaml", changes)
    except ConfigError as exc:
        return _redirect("/setup#search", err=f"Not saved: {str(exc)[:400]}")
    from app.services.analysis import rescore_all

    rescore_all()  # the filters depend on these settings
    return _redirect("/setup#search", msg="Search preferences saved and applied to your jobs." + note)


@router.post("/setup/personal")
def setup_personal(applicant_name: Annotated[str, Form()] = "", citizenships: Annotated[str, Form()] = "",
                   enrolled: Annotated[str, Form()] = ""):
    try:
        config_writer.update("preferences.yaml", {"writing.applicant_name": applicant_name.strip()})
        config_writer.update("profile.yaml", {"citizenships": _lines(citizenships), "enrolled_student": bool(enrolled)})
    except ConfigError as exc:
        return _redirect("/setup#personal", err=f"Not saved: {str(exc)[:400]}")
    from app.services.analysis import rescore_all

    rescore_all()
    return _redirect("/setup#personal", msg="Personal details saved")


@router.get("/platforms", response_class=HTMLResponse)
def platforms_page(request: Request):
    from app.services.platforms import overview

    return templates.TemplateResponse(request, "platforms.html", {"request": request, "overview": overview()})


@router.get("/setup/edit/{name}", response_class=HTMLResponse)
def setup_edit(request: Request, name: str):
    if name not in EDITABLE:
        return _redirect("/setup", err="Unknown file")
    path = config_writer.config_path(name)
    return templates.TemplateResponse(request, "setup_edit.html", {
        "request": request, "name": name, "text": path.read_text(encoding="utf-8") if path.exists() else ""})


@router.post("/setup/edit/{name}")
def setup_edit_save(name: str, text: Annotated[str, Form()]):
    if name not in EDITABLE:
        return _redirect("/setup", err="Unknown file")
    try:
        config_writer.write_text(name, text.replace("\r\n", "\n"))
    except ConfigError as exc:
        return _redirect(f"/setup/edit/{name}", err=f"Not saved — {str(exc)[:600]}")
    if name == "companies.yaml":
        from app.services.companies import import_companies_yaml

        res = import_companies_yaml()
        return _redirect("/setup", msg=f"{name} saved · sources imported: {res}")
    return _redirect("/setup", msg=f"{name} saved (previous version in config/.backups)")
