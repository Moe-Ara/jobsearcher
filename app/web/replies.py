"""Replies from employers: every rejection, invitation and confirmation found in your mailbox, the job it belongs to,
and one-click fixes (assign to a job, change what it is)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import func, or_, select

from app.db import session_scope
from app.email_ingest import replies as R
from app.models import EmailMessage, Job
from app.web.templating import templates

router = APIRouter()

KINDS = ("rejection", "interview", "next_step", "received", "other")
FILTERS = {"all": None, "attention": ("interview", "next_step"), "rejection": ("rejection",), "received": ("received",),
           "unmatched": None, "other": ("other",)}


def _jobs_by_id(s, ids: set[int]) -> dict[int, Job]:
    return {j.id: j for j in s.scalars(select(Job).where(Job.id.in_(ids)))} if ids else {}


@router.get("/replies", response_class=HTMLResponse)
def replies_page(request: Request, show: str = "all", q: str = ""):
    show = show if show in FILTERS else "all"
    with session_scope() as s:
        base = select(EmailMessage).where(EmailMessage.reply_kind.is_not(None))
        counts = dict(s.execute(select(EmailMessage.reply_kind, func.count()).where(EmailMessage.reply_kind.is_not(None))
                                .group_by(EmailMessage.reply_kind)).all())
        counts["unmatched"] = s.scalar(select(func.count()).select_from(EmailMessage).where(
            EmailMessage.reply_kind.in_(("rejection", "interview", "next_step")), EmailMessage.job_id.is_(None))) or 0
        if show == "unmatched":
            base = base.where(EmailMessage.reply_kind.in_(("rejection", "interview", "next_step")), EmailMessage.job_id.is_(None))
        elif FILTERS[show]:
            base = base.where(EmailMessage.reply_kind.in_(FILTERS[show]))
        else:
            base = base.where(EmailMessage.reply_kind != "other")
        if q.strip():
            like = f"%{q.strip()}%"
            base = base.where(or_(EmailMessage.subject.ilike(like), EmailMessage.from_addr.ilike(like),
                                  EmailMessage.snippet.ilike(like)))
        rows = list(s.scalars(base.order_by(EmailMessage.received_at.desc()).limit(300)))
        ids = {r.job_id for r in rows if r.job_id}
        cands: dict[int, list[int]] = {}
        for r in rows:
            if not r.job_id and (r.match_note or "").startswith("possible jobs:"):
                cands[r.id] = [int(x) for x in r.match_note.split(":", 1)[1].split(",") if x.strip().isdigit()]
                ids |= set(cands[r.id])
        jobs = _jobs_by_id(s, ids)
        s.expunge_all()
    return templates.TemplateResponse(request, "replies.html", {
        "request": request, "rows": rows, "jobs": jobs, "cands": cands, "show": show, "q": q, "counts": counts,
        "kinds": KINDS, "msg": request.query_params.get("msg"), "err": request.query_params.get("err"),
    })


def _back(show: str, msg: str = "", err: str = "") -> RedirectResponse:
    from urllib.parse import urlencode

    params = {k: v for k, v in {"show": show, "msg": msg, "err": err}.items() if v}
    return RedirectResponse(f"/replies?{urlencode(params)}", status_code=303)


def _reapply(s, row: EmailMessage, job: Job | None) -> str | None:
    note = (f"E-mail ({R.KIND_LABEL[row.reply_kind].lower()}, set by you): “{(row.subject or '')[:140]}” from "
            f"{row.from_addr}")
    return R.apply_effect(s, job, row.reply_kind, note=note, forced=True) if job is not None else None


@router.post("/replies/{reply_id}/assign")
def assign_reply(reply_id: int, job_id: str = Form(""), show: str = Form("all")):
    """This e-mail belongs to that job: link it and move the job along (your choice wins over any rule)."""
    jid = int(job_id.strip().lstrip("#")) if job_id.strip().lstrip("#").isdigit() else None
    with session_scope() as s:
        row = s.get(EmailMessage, reply_id)
        if row is None:
            raise HTTPException(404, "Reply not found")
        job = s.get(Job, jid) if jid else None
        if jid and job is None:
            return _back(show, err=f"There is no job {jid}")
        row.job_id = jid
        row.match_note = "assigned by you" if jid else "unlinked by you"
        row.effect = _reapply(s, row, job) or row.effect if job else None
        return _back(show, msg=(f"Linked to job {jid}" + (f" — status {row.effect}" if row.effect else "")) if jid else "Unlinked")


@router.post("/replies/{reply_id}/kind")
def set_reply_kind(reply_id: int, kind: str = Form(...), show: str = Form("all")):
    """It is a rejection / an invitation / … after all: re-label it and move its job along."""
    if kind not in KINDS:
        raise HTTPException(400, "Unknown kind")
    with session_scope() as s:
        row = s.get(EmailMessage, reply_id)
        if row is None:
            raise HTTPException(404, "Reply not found")
        row.reply_kind = kind
        job = s.get(Job, row.job_id) if row.job_id else None
        effect = _reapply(s, row, job)
        if effect:
            row.effect = effect
        return _back(show, msg=f"Marked as {R.KIND_LABEL[kind].lower()}" + (f" — status {effect}" if effect else ""))


def summary() -> dict[str, Any]:
    """Counts for other pages (applications overview)."""
    with session_scope() as s:
        return dict(s.execute(select(EmailMessage.reply_kind, func.count()).where(EmailMessage.reply_kind.is_not(None))
                              .group_by(EmailMessage.reply_kind)).all())
