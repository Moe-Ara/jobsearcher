"""Saves every finished application package as files — the papers you send or upload.

data/applications/
  ready to send/   reviewer verdict SEND
  needs review/    verdict REVISE (read the reviewer's points in info.txt first)
    2026-10-20 Company - Job title/
      Bewerbung_<Name>_<Company>.pdf   cover letter + CV + certificates (the one file most employers want)
      Anschreiben_<Name>.pdf           cover letter alone
      Lebenslauf_<Name>.pdf            CV alone
      E-Mail.txt                       subject + body of the application e-mail
      info.txt                         where to apply, verdict, scores, the reviewer's open points

Packages are exported when they finish (and again when you change them); exporting all re-writes every folder.
"""

from __future__ import annotations

import io
import logging
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified

from app.db import session_scope
from app.models import ApplicationPackage, Job, utcnow

log = logging.getLogger(__name__)

GROUPS = {"SEND": "ready to send", "REVISE": "needs review"}


def export_root() -> Path:
    from app.services.application_agent import documents_dir

    return documents_dir().parent / "applications"


def _safe(text: str, limit: int) -> str:
    text = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", " ", text or "")
    return re.sub(r"\s+", " ", text).strip(" .")[:limit].strip(" .") or "-"


def folder_name(pkg: ApplicationPackage, job: Job) -> str:
    from app.services.application_agent import _GENDER_SUFFIX

    day = (pkg.created_at or utcnow()).strftime("%Y-%m-%d")
    title = _GENDER_SUFFIX.sub("", job.title or "")
    return f"{day} {_safe(job.company, 40)} - {_safe(title, 60)} (#{pkg.id})"


def _info(pkg: ApplicationPackage, job: Job) -> str:
    from app.services import submit

    ch = submit.channel(job)
    best = next((r for r in (pkg.reviews or []) if r.get("best")), (pkg.reviews or [{}])[-1] if pkg.reviews else {})
    lines = [f"{job.title}", f"{job.company} · {job.location or job.country or ''}".strip(" ·"), ""]
    if ch["channel"] == "email":
        lines.append(f"Apply by e-mail to: {ch['to']}  (or use “Send application” in JobSearcher)")
    else:
        lines.append("Apply on the employer's site: " + (ch.get("url") or job.career_page_url or job.source_url or "—"))
    lines += ["", f"Reviewer verdict: {pkg.verdict} · ATS {pkg.ats_score} · recruiter {pkg.recruiter_score}",
              f"JobSearcher: /applications/{pkg.id}   (job /jobs/{job.id})"]
    if pkg.verdict != "SEND" and best.get("reasons"):
        lines += ["", "Open points from the reviewer:"] + [f"- {r}" for r in best["reasons"]]
    if best.get("gap_notes"):
        lines += ["", "Not counted against the documents (about your experience):"] + [f"- {g}" for g in best["gap_notes"]]
    return "\n".join(lines) + "\n"


def export_package(pkg_id: int) -> Path | None:
    """Write the package's papers to data/applications/<group>/<folder>. Returns the folder (None if not finished)."""
    from app.services.application_agent import ApplicationError, package_file

    with session_scope() as s:
        pkg = s.get(ApplicationPackage, pkg_id)
        if pkg is None or pkg.status != "done" or not pkg.cv:
            return None
        job = s.get(Job, pkg.job_id)
        files: dict[str, bytes] = {}
        for kind in ("bundle.pdf", "letter.pdf", "cv.pdf"):
            try:
                data, name, _ = package_file(s, pkg, kind)
                files[name] = data
            except ApplicationError as exc:  # e.g. no finished letter: export what exists
                log.info("Export of package %s: no %s (%s)", pkg_id, kind, exc)
        email = pkg.email or {}
        if email.get("body"):
            files["E-Mail.txt" if pkg.language == "de" else "Email.txt"] = (
                f"Betreff: {email.get('subject', '')}\n\n{email['body']}\n" if pkg.language == "de"
                else f"Subject: {email.get('subject', '')}\n\n{email['body']}\n").encode("utf-8")
        files["info.txt"] = _info(pkg, job).encode("utf-8")
        target = export_root() / GROUPS.get(pkg.verdict or "", "needs review") / folder_name(pkg, job)
        old = (pkg.meta or {}).get("export", {}).get("dir")
        if old and (export_root() / old) != target:
            shutil.rmtree(export_root() / old, ignore_errors=True)  # verdict or title changed: no stale copy
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (target / name).write_bytes(data)
        meta = dict(pkg.meta or {})
        meta["export"] = {"dir": target.relative_to(export_root()).as_posix(), "at": utcnow().isoformat(timespec="seconds"),
                          "files": sorted(files)}
        pkg.meta = meta
        flag_modified(pkg, "meta")
        return target


def export_all() -> dict[str, Any]:
    with session_scope() as s:
        ids = list(s.scalars(select(ApplicationPackage.id).where(ApplicationPackage.status == "done")
                             .order_by(ApplicationPackage.id)))
    done, failed = 0, []
    for pkg_id in ids:
        try:
            if export_package(pkg_id):
                done += 1
        except Exception as exc:  # noqa: BLE001 - one broken package must not stop the rest
            log.warning("Export of package %s failed: %s", pkg_id, exc)
            failed.append(pkg_id)
    return {"exported": done, "failed": failed, "folder": str(export_root())}


def zip_all() -> bytes:
    """Every exported folder in one ZIP (for the browser download)."""
    root = export_root()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(root).as_posix())
    return buf.getvalue()
