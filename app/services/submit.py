"""Submission tool — sends an application by e-mail. ONLY on your explicit command (a button you press).

Channel: e-mail, when the posting names an application address ("Bewerbung bitte an jobs@firma.de"). The message is
the package's application e-mail with the complete application PDF (letter + CV + certificates) attached, sent from
your mailbox (SMTP, same login as IMAP), with a Bcc copy to yourself. Online-form portals are not filled in: those
packages stay "ready" and you submit them yourself (then "Mark as applied").

Safety rails: never twice for the same job; reviewer verdict SEND required unless you override it; noreply / privacy
addresses are never used; a daily cap and a pause between messages; every send is recorded on the package and the job
is set to Applied with a note.
"""

from __future__ import annotations

import logging
import re
import smtplib
import ssl
import time
from datetime import timedelta
from email.message import EmailMessage as MimeMessage
from email.utils import formatdate, make_msgid
from typing import Any

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.config_files import get_prefs
from app.db import session_scope
from app.enums import JobStatus
from app.models import ApplicationPackage, Job, utcnow
from app.services import toolbox
from app.settings import get_settings

log = logging.getLogger(__name__)

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_NEVER = re.compile(r"(no-?reply|do-?not-?reply|noreply|datenschutz|privacy|dsb@|dpo@|newsletter|abuse|postmaster|"
                    r"webmaster|presse|press@|example\.|beispiel)", re.I)
_APPLY_CONTEXT = re.compile(r"(bewerb|application|apply|lebenslauf|unterlagen|cv\b|résumé|resume|karriere|career|"
                            r"recruit|personal|human resources|\bhr\b|ansprechpart|contact person|jobs?@)", re.I)
_GOOD_LOCAL = re.compile(r"^(bewerbung|bewerbungen|jobs?|karriere|career|careers|recruiting|recruitment|hr|personal|"
                         r"talent|apply|application)", re.I)


class SubmitError(ValueError):
    pass


def application_address(job: Job) -> str | None:
    """The address the posting asks applications to be sent to (None if it names none we trust)."""
    text = " ".join(x or "" for x in (job.description, job.requirements, job.responsibilities))
    best, best_score = None, 0
    for m in _EMAIL.finditer(text):
        addr = m.group(0).strip(".").lower()
        if _NEVER.search(addr):
            continue
        context = text[max(0, m.start() - 160): m.end() + 40]
        score = (2 if _APPLY_CONTEXT.search(context) else 0) + (2 if _GOOD_LOCAL.match(addr.split("@")[0]) else 0)
        if score > best_score:
            best, best_score = addr, score
    return best if best_score >= 2 else None


def channel(job: Job) -> dict[str, Any]:
    addr = application_address(job)
    if addr:
        return {"channel": "email", "to": addr}
    return {"channel": "portal", "url": job.career_page_url or job.canonical_url or job.source_url}


def _sent_today() -> int:
    since = utcnow() - timedelta(days=1)
    with session_scope() as s:
        return sum(1 for p in s.scalars(select(ApplicationPackage)) if (p.meta or {}).get("submission")
                   and (p.meta["submission"].get("at") or "") >= since.isoformat())


def _smtp_send(msg: MimeMessage) -> None:
    st = get_settings()
    host = st.smtp_host or (st.imap_host.replace("imap.", "smtp.", 1) if st.imap_host else "")
    user = st.smtp_user or st.imap_user
    password = (st.smtp_password or st.imap_password)
    if not (host and user and password):
        raise SubmitError("Sending is not configured: set IMAP_* (or SMTP_HOST / SMTP_USER / SMTP_PASSWORD) in .env")
    with smtplib.SMTP(host, st.smtp_port, timeout=60) as smtp:
        smtp.starttls(context=ssl.create_default_context())
        smtp.login(user, password.get_secret_value())
        smtp.send_message(msg)


def preview(pkg_id: int) -> dict[str, Any]:
    """What would be sent — shown before you confirm."""
    from app.services.application_agent import bundle_filename

    with session_scope() as s:
        pkg = s.get(ApplicationPackage, pkg_id)
        if pkg is None:
            raise SubmitError("Application package not found")
        job = s.get(Job, pkg.job_id)
        ch = channel(job)
        sent = (pkg.meta or {}).get("submission")
        return {**ch, "subject": (pkg.email or {}).get("subject"), "body": (pkg.email or {}).get("body"),
                "attachment": bundle_filename(job, pkg.language, pkg.cv["name"]) if pkg.cv else None,
                "verdict": pkg.verdict, "already_sent": sent, "job_status": str(job.status)}


# You applied (or were already turned down after applying): never apply again for the same vacancy.
APPLIED_STATES = (JobStatus.APPLIED, JobStatus.HR_SCREEN, JobStatus.INTERVIEW, JobStatus.TECHNICAL, JobStatus.FINAL,
                  JobStatus.OFFER, JobStatus.REJECTED)
_OPEN_STATES = (JobStatus.NEW, JobStatus.REVIEW, JobStatus.PRIORITY, JobStatus.APPLYING)


def vacancy_twins(session: Session, job: Job, statuses: tuple | None = None) -> list[Job]:
    """Other rows for the same vacancy: the same posting found on another board (LinkedIn + StepStone), a re-post,
    or a second e-mail alert — same company and title, or the same posting URL. `statuses`: only twins in these."""
    conds = []
    if job.normalized_company and job.normalized_title:
        conds.append(and_(Job.normalized_company == job.normalized_company, Job.normalized_title == job.normalized_title))
    from app.services.liveness import _looks_like_posting

    # Only a job's own posting URL proves the same vacancy (e.g. the LinkedIn copy links to the employer's posting).
    # Two equal career_page_urls prove nothing: that is often the careers page or board every job of the employer
    # shares (249 Bosch jobs: jobs.smartrecruiters.com/BoschGroup; TUM: portal.mytum.de/jobs/wissenschaftler).
    for url in {u for u in (job.canonical_url, job.career_page_url) if _looks_like_posting(u)}:
        conds.append(Job.canonical_url == url)
    if _looks_like_posting(job.canonical_url):
        conds.append(Job.career_page_url == job.canonical_url)
    if not conds:
        return []
    q = select(Job).where(Job.id != job.id, or_(*conds))
    if statuses:
        q = q.where(Job.status.in_(statuses))  # uses the status index: fast enough for every ingested posting
    return list(session.scalars(q))


def applied_reason(session: Session, job: Job) -> str | None:
    """Why this job must not get another application (None = it may)."""
    rows = [job, *vacancy_twins(session, job)]
    packages = session.scalars(select(ApplicationPackage).where(ApplicationPackage.job_id.in_([j.id for j in rows]))).all()
    for j in rows:
        twin = "" if j.id == job.id else f" — same vacancy as job {j.id} ({j.source})"
        for p in packages:
            if p.job_id != j.id:
                continue
            meta = p.meta or {}
            if meta.get("submission"):
                return f"This application was already sent{twin}"
            if meta.get("sending"):
                return (f"A send started at {meta['sending'].get('at', '?')[:16]} was never confirmed{twin} — check "
                        "your Sent folder; if it went out, press “Mark as applied”")
        if j.status in APPLIED_STATES:
            return f"You already applied for this job{twin}"
    return None


def inherit_applied(session: Session, job: Job) -> int | None:
    """A posting that turns up again (another board, a re-post) after you applied: Applied too, so it never shows up
    as a new job and is never prepared twice. Returns the id of the job you applied through."""
    from app.services.jobs import set_status

    if job.status not in _OPEN_STATES:
        return None
    applied = next(iter(vacancy_twins(session, job, APPLIED_STATES)), None)
    if applied is None:
        return None
    set_status(session, job, JobStatus.APPLIED, note=f"Same vacancy as job {applied.id} — you already applied there")
    if applied.applied_at:
        job.applied_at = applied.applied_at
    return applied.id


def _mark_twins_applied(session: Session, job: Job, how: str) -> list[int]:
    from app.services.jobs import set_status

    marked = []
    for twin in vacancy_twins(session, job):
        if twin.status in _OPEN_STATES:
            set_status(session, twin, JobStatus.APPLIED, note=f"Same vacancy as job {job.id} — {how} there")
            marked.append(twin.id)
    return marked


@toolbox.tool("send_application", "application", "Send an application e-mail with the complete PDF — only on your command.")
def send_application(pkg_id: int, allow_revise: bool = False) -> dict[str, Any]:
    """Send one package by e-mail. Raises SubmitError when a safety rail says no."""
    from app.services.application_agent import bundle_filename, package_file
    from app.services.jobs import set_status

    p = get_prefs().application
    if p.max_sends_per_day > 0 and _sent_today() >= p.max_sends_per_day:
        raise SubmitError(f"Daily limit of {p.max_sends_per_day} sent applications reached")
    with session_scope() as s:
        pkg = s.get(ApplicationPackage, pkg_id)
        if pkg is None or pkg.status != "done" or not pkg.cv:
            raise SubmitError("The package is not finished")
        job = s.get(Job, pkg.job_id)
        reason = applied_reason(s, job)
        if reason:
            raise SubmitError(reason)
        if pkg.verdict != "SEND" and not allow_revise:
            raise SubmitError("The reviewer said REVISE — tick “send anyway” to override")
        ch = channel(job)
        if ch["channel"] != "email":
            raise SubmitError("The posting names no application e-mail address — apply via the employer's portal")
        pdf, _, _ = package_file(s, pkg, "bundle.pdf")
        filename = bundle_filename(job, pkg.language, pkg.cv["name"])
        sender = pkg.cv["contact"].get("email") or f"{get_settings().imap_user}"
        msg = MimeMessage()
        msg["From"] = f"{pkg.cv['name']} <{sender}>"
        msg["To"] = ch["to"]
        msg["Bcc"] = sender
        msg["Subject"] = (pkg.email or {}).get("subject") or f"Bewerbung: {job.title}"
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid(domain=sender.split("@")[-1] if "@" in sender else None)
        msg.set_content((pkg.email or {}).get("body") or "")
        msg.add_attachment(pdf, maintype="application", subtype="pdf", filename=filename)
        # Recorded BEFORE the e-mail goes out: should the app die between sending and saving, the marker blocks a
        # second send of this vacancy instead of silently allowing a duplicate.
        meta = dict(pkg.meta or {})
        meta["sending"] = {"at": utcnow().isoformat(timespec="seconds"), "to": ch["to"]}
        pkg.meta = meta
        flag_modified(pkg, "meta")
        s.commit()
        try:
            _smtp_send(msg)
        except Exception:
            meta = dict(pkg.meta or {})
            meta.pop("sending", None)
            pkg.meta = meta
            flag_modified(pkg, "meta")
            s.commit()
            raise
        meta = dict(pkg.meta or {})
        meta.pop("sending", None)
        meta["submission"] = {"at": utcnow().isoformat(timespec="seconds"), "channel": "email", "to": ch["to"],
                              "message_id": msg["Message-ID"], "attachment": filename, "bytes": len(pdf)}
        pkg.meta = meta
        flag_modified(pkg, "meta")
        set_status(s, job, JobStatus.APPLIED, note=f"Application sent by e-mail to {ch['to']} (JobSearcher)")
        twins = _mark_twins_applied(s, job, "sent")
        if twins:
            meta["submission"]["also_marked"] = twins
        log.info("Sent application for job %s to %s", job.id, ch["to"])
        return meta["submission"]


def mark_applied(pkg_id: int) -> None:
    """You submitted it yourself (portal): record it and set the job to Applied."""
    from app.services.jobs import set_status

    with session_scope() as s:
        pkg = s.get(ApplicationPackage, pkg_id)
        if pkg is None:
            raise SubmitError("Application package not found")
        job = s.get(Job, pkg.job_id)
        meta = dict(pkg.meta or {})
        meta.setdefault("submission", {"at": utcnow().isoformat(timespec="seconds"), "channel": "manual"})
        meta.pop("sending", None)  # an unconfirmed send you checked in your Sent folder
        pkg.meta = meta
        flag_modified(pkg, "meta")
        if job.status not in APPLIED_STATES:
            set_status(s, job, JobStatus.APPLIED, note="Applied (marked in JobSearcher)")
        _mark_twins_applied(s, job, "applied")


def pause_between_sends() -> None:
    time.sleep(max(0, get_prefs().application.send_pause_seconds))
