"""Removes "jobs" that older alert parsers created from e-mail interface text ("Anmelden", "Mindestgrundgehalt",
"Arbeitsort", "Weil du dir …", "Diesen Job melden" …). Jobs you worked with are never touched."""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select

from app.db import session_scope
from app.models import Job

JUNK_TITLE = re.compile(
    r"^(anmelden|mindestgrundgehalt|arbeitsort|mehr jobs anzeigen|profil überprüfen|weil du|diesen job melden|du passt|"
    r"erhöhe deine|top-fähigkeiten|bei diesem job|nicht mehr auf jobsuche|unsere top-empfehlung|"
    r"(view|ansehen|anzeigen|details|mehr|more|apply|bewerben)$)", re.IGNORECASE)
JUNK_URL = re.compile(r"secure\.indeed\.com/account|/login|magiclink", re.IGNORECASE)


def is_junk(job: Job) -> bool:
    if job.source not in ("indeed", "stepstone", "linkedin"):
        return False
    return bool(JUNK_TITLE.match(job.title or "") or JUNK_TITLE.match(job.company or "")
                or JUNK_URL.search(job.source_url or ""))


def purge_alert_junk() -> dict[str, Any]:
    from app.services.retention import _interacted

    deleted = kept = 0
    with session_scope() as s:
        for job in s.scalars(select(Job).where(Job.source.in_(("indeed", "stepstone", "linkedin")))).all():
            if not is_junk(job):
                continue
            if job.relevance_label == "BAD":  # labelling junk "bad" while calibrating is not working with it
                job.relevance_label = "UNRATED"
            if _interacted(job, s):
                kept += 1
                continue
            s.delete(job)
            deleted += 1
    return {"deleted": deleted, "kept_because_you_used_them": kept}
