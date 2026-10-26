"""Mailbox clean-up tool — moves job-alert e-mails the app has already processed out of your folders.

Only alert e-mails that JobSearcher has read successfully (a row in email_messages with a parser and no error) are
touched, matched by their Message-ID, and only after `email.cleanup_min_age_days`. Everything else — application
confirmations, recruiter mail, alerts that failed to parse — stays where it is. Target: the trash folder (default,
recoverable) or any folder you name (an archive). Always run as a preview first (dry_run=True), then for real.
"""

from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import Any

from sqlalchemy import select

from app.config_files import get_prefs
from app.db import session_scope
from app.email_ingest.parsers.boards import all_parsers
from app.email_ingest.service import _connect, decode_mailbox_name, mailbox_folders, parse_list_response, quote_mailbox
from app.models import EmailMessage, utcnow
from app.services import toolbox
from app.settings import get_settings

log = logging.getLogger(__name__)

_TRASH_NAMES = re.compile(r"^(papierkorb|trash|deleted( items| messages)?|gelöscht(e objekte)?|bin)$", re.I)
_MSGID = re.compile(rb"message-id:\s*(<[^>\r\n]+>)", re.I)


def _trash_folder(imap) -> str | None:
    typ, data = imap.list()
    if typ != "OK":
        return None
    folders = parse_list_response(data or [])
    for flags, _, name in folders:
        if "\\trash" in flags:
            return name
    for _, delim, name in folders:
        leaf = decode_mailbox_name(name).split(delim)[-1] if delim and delim != "NIL" else decode_mailbox_name(name)
        if _TRASH_NAMES.match(leaf.strip()):
            return name
    return None


def _processed_ids(min_age_days: int) -> set[str]:
    cutoff = utcnow() - timedelta(days=min_age_days)
    with session_scope() as s:
        rows = s.scalars(select(EmailMessage.message_id).where(EmailMessage.parser.is_not(None),
                                                               EmailMessage.error.is_(None),
                                                               EmailMessage.processed_at <= cutoff))
        return {m.strip() for m in rows if m}


def _move(imap, uids: list[bytes], target: str) -> None:
    caps = {c.decode().upper() if isinstance(c, bytes) else str(c).upper() for c in (imap.capabilities or ())}
    uid_set = b",".join(uids).decode()
    if "MOVE" in caps:
        typ, _ = imap.uid("MOVE", uid_set, quote_mailbox(target))
        if typ == "OK":
            return
    typ, _ = imap.uid("COPY", uid_set, quote_mailbox(target))
    if typ != "OK":
        raise RuntimeError(f"could not copy to {target}")
    imap.uid("STORE", uid_set, "+FLAGS.SILENT", r"(\Deleted)")
    if "UIDPLUS" in caps:
        imap.uid("EXPUNGE", uid_set)
    else:
        imap.expunge()


@toolbox.tool("clean_mailbox", "scout", "Move job-alert e-mails the app has already processed to the trash (preview first).")
def clean_seen_alerts(dry_run: bool = True) -> dict[str, Any]:
    prefs = get_prefs().email
    settings = get_settings()
    if not settings.imap_configured:
        return {"ok": False, "error": "IMAP is not configured"}
    known = _processed_ids(prefs.cleanup_min_age_days)
    senders = sorted({pat for p in all_parsers(prefs.enabled_parsers, prefs.generic_senders) for pat in p.sender_patterns})
    imap = _connect(settings)
    report: dict[str, Any] = {"ok": True, "dry_run": dry_run, "folders": {}, "matched": 0, "moved": 0, "target": None}
    try:
        target = prefs.cleanup_target if prefs.cleanup_target != "trash" else _trash_folder(imap)
        if not target:
            return {**report, "ok": False, "error": "No trash folder found — set email.cleanup_target to a folder name"}
        report["target"] = decode_mailbox_name(target)
        for folder in mailbox_folders(imap, settings.imap_folder):
            if folder == target:
                continue
            typ, _ = imap.select(quote_mailbox(folder), readonly=dry_run)
            if typ != "OK":
                continue
            uids: set[bytes] = set()
            for sender in senders:
                typ, data = imap.uid("search", None, f'(FROM "{sender}")')
                if typ == "OK" and data and data[0]:
                    uids |= set(data[0].split())
            matched = []
            ordered = sorted(uids, key=int)
            for i in range(0, len(ordered), 200):
                chunk = b",".join(ordered[i:i + 200]).decode()
                typ, data = imap.uid("fetch", chunk, "(UID BODY.PEEK[HEADER.FIELDS (MESSAGE-ID)])")
                for part in data or []:
                    if not isinstance(part, tuple):
                        continue
                    uid = re.search(rb"UID (\d+)", part[0])
                    mid = _MSGID.search(part[1] or b"")
                    if uid and mid and mid.group(1).decode(errors="replace").strip() in known:
                        matched.append(uid.group(1))
            if not matched:
                continue
            report["folders"][decode_mailbox_name(folder)] = len(matched)
            report["matched"] += len(matched)
            if not dry_run:
                for i in range(0, len(matched), 100):
                    _move(imap, matched[i:i + 100], target)
                report["moved"] += len(matched)
    finally:
        try:
            imap.logout()
        except Exception:  # noqa: BLE001
            pass
    log.info("Mailbox clean-up (%s): %d processed alert e-mails %s → %s", "preview" if dry_run else "run",
             report["matched"], "would move" if dry_run else "moved", report["target"])
    return report
