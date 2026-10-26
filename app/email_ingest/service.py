"""Job-alert email ingestion: IMAP mailbox polling and .eml uploads share the same processing path."""

from __future__ import annotations

import base64
import email
import imaplib
import re
import logging
from datetime import timedelta
from email.message import Message
from email.policy import default as default_policy
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any

from sqlalchemy import select

from app.config_files import get_prefs
from app.db import session_scope
from app.email_ingest.parsers.boards import all_parsers, pick_parser
from app.models import EmailMessage, SourceRun, utcnow
from app.services.ingest import ingest_posting
from app.settings import get_settings
from app.sources.base import JobPosting

log = logging.getLogger(__name__)


def _bodies(msg: Message) -> tuple[str, str]:
    html, text = "", ""
    for part in msg.walk() if msg.is_multipart() else [msg]:
        if part.get_content_maintype() == "multipart" or part.get_filename():
            continue
        ctype = part.get_content_type()
        try:
            payload = part.get_content() if hasattr(part, "get_content") else part.get_payload(decode=True)
        except (LookupError, KeyError):
            raw = part.get_payload(decode=True) or b""
            payload = raw.decode("utf-8", errors="replace")
        if isinstance(payload, bytes):
            payload = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
        if ctype == "text/html" and not html:
            html = payload
        elif ctype == "text/plain" and not text:
            text = payload
    return html, text


def parse_message(msg: Message) -> tuple[str | None, list[JobPosting]]:
    prefs = get_prefs()
    parsers = all_parsers(prefs.email.enabled_parsers, prefs.email.generic_senders)
    from_addr = parseaddr(str(msg.get("From", "")))[1]
    subject = str(msg.get("Subject", ""))
    parser = pick_parser(parsers, from_addr, subject)
    if parser is None:
        return None, []
    html, text = _bodies(msg)
    if not html and text:
        html = "<br>".join(f'<a href="{w}">{w}</a>' if w.startswith("http") else w for w in text.split())
    return parser.name, parser.parse(html, text)


def process_message(msg: Message, *, dry_run: bool = False) -> dict[str, Any]:
    message_id = str(msg.get("Message-ID") or "").strip() or f"nomid:{hash(msg.as_bytes())}"
    from_addr = parseaddr(str(msg.get("From", "")))[1]
    subject = str(msg.get("Subject", ""))[:1000]
    try:
        received = parsedate_to_datetime(str(msg.get("Date"))).replace(tzinfo=None) if msg.get("Date") else None
    except (TypeError, ValueError):
        received = None

    parser_name, postings = parse_message(msg)
    if dry_run:
        return {"parser": parser_name, "jobs": [p.model_dump(include={"title", "company", "location", "url", "source"}) for p in postings]}

    with session_scope() as s:
        if s.scalars(select(EmailMessage.id).where(EmailMessage.message_id == message_id)).first():
            return {"parser": parser_name, "skipped": "already processed"}
        em = EmailMessage(message_id=message_id, from_addr=from_addr, subject=subject, received_at=received, parser=parser_name)
        s.add(em)
        s.flush()
        em_id = em.id

    counts = {"new": 0, "merged": 0, "updated": 0, "unchanged": 0}
    errors = []
    for posting in postings:
        try:
            with session_scope() as s:
                res = ingest_posting(s, posting, email_message_id=em_id)
                counts[res.outcome] = counts.get(res.outcome, 0) + 1
        except Exception as exc:  # noqa: BLE001 - keep going with the other jobs in the email
            log.warning("Failed to ingest job from email %s: %s", message_id, exc)
            errors.append(str(exc)[:300])
    with session_scope() as s:
        em = s.get(EmailMessage, em_id)
        em.jobs_found = len(postings)
        em.error = "; ".join(errors[:3]) or None
    return {"parser": parser_name, "found": len(postings), **counts, "errors": errors}


def process_eml_bytes(data: bytes, *, dry_run: bool = False) -> dict[str, Any]:
    msg = email.message_from_bytes(data, policy=default_policy)
    return process_message(msg, dry_run=dry_run)


# Folders never worth scanning (matched on the last path segment, decoded). Junk/Spam IS scanned: alerts get
# misfiled there, and only known alert senders are parsed anyway.
_SKIP_FOLDER_RE = re.compile(
    r"^(trash|deleted( items| messages)?|papierkorb|gelöscht.*|drafts?|entwürfe|sent( items| mail| messages)?|gesendet.*"
    r"|outbox|postausgang|templates|vorlagen|notes|notizen|calendar|kalender|contacts|kontakte|tasks|aufgaben)$",
    re.IGNORECASE,
)
_SKIP_FLAGS = {"\\trash", "\\drafts", "\\sent", "\\noselect", "\\nonexistent"}
_LIST_RE = re.compile(r'^\((?P<flags>[^)]*)\)\s+(?P<delim>"(?:[^"\\]|\\.)*"|NIL)\s+(?P<name>.+)$')


def decode_mailbox_name(name: str) -> str:
    """IMAP modified UTF-7 (RFC 3501) → str, e.g. 'Entw&APw-rfe' → 'Entwürfe'."""

    def repl(m: re.Match) -> str:
        chunk = m.group(1)
        if not chunk:
            return "&"
        b64 = chunk.replace(",", "/")
        b64 += "=" * (-len(b64) % 4)
        try:
            return base64.b64decode(b64).decode("utf-16-be")
        except (ValueError, UnicodeDecodeError):
            return m.group(0)

    return re.sub(r"&([A-Za-z0-9+,]*)-", repl, name)


def quote_mailbox(name: str) -> str:
    return '"' + name.replace("\\", "\\\\").replace('"', '\\"') + '"'


def parse_list_response(lines: list[bytes]) -> list[tuple[set[str], str, str]]:
    """LIST lines → [(flags, delimiter, raw mailbox name)]."""
    out = []
    for line in lines:
        if not isinstance(line, bytes):
            continue
        m = _LIST_RE.match(line.decode("utf-8", errors="replace").strip())
        if not m:
            continue
        name = m.group("name").strip()
        if name.startswith('"') and name.endswith('"'):
            name = name[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        delim = m.group("delim").strip('"')
        out.append(({f.lower() for f in m.group("flags").split()}, delim, name))
    return out


def mailbox_folders(imap, setting: str) -> list[str]:
    """IMAP_FOLDER: '*' = every selectable folder except trash/sent/drafts; otherwise a comma-separated list."""
    setting = (setting or "*").strip()
    if setting not in ("*", "ALL", "all"):
        return [f.strip() for f in setting.split(",") if f.strip()]
    typ, data = imap.list()
    if typ != "OK":
        return ["INBOX"]
    folders = parse_list_response(data or [])
    # Gmail exposes every label in "[Gmail]/All Mail" (\All) — scanning it plus spam avoids N label folders.
    all_mail = [n for flags, _, n in folders if "\\all" in flags]
    if all_mail:
        return all_mail + [n for flags, _, n in folders if "\\junk" in flags]
    selected = []
    for flags, delim, name in folders:
        leaf = decode_mailbox_name(name).split(delim)[-1] if delim and delim != "NIL" else decode_mailbox_name(name)
        if flags & _SKIP_FLAGS or _SKIP_FOLDER_RE.match(leaf.strip()):
            continue
        selected.append(name)
    return selected or ["INBOX"]


# Last known mailbox connection state (shown on Operations). Updated by check_imap() and every poll.
IMAP_STATUS: dict[str, Any] = {"ok": None, "checked_at": None, "error": None, "folders": [], "account": None}

_PROVIDER_HINT = ("Check IMAP_USER/IMAP_PASSWORD in .env. Many providers (web.de, GMX, Gmail, Outlook) only allow IMAP "
                  "after enabling it in the mailbox settings (web.de: Einstellungen → POP3/IMAP Abruf → erlauben) and "
                  "require an app-specific password when two-factor login is on.")


def _set_status(ok: bool, error: str | None = None, folders: list[str] | None = None) -> None:
    s = get_settings()
    IMAP_STATUS.update(ok=ok, error=error, checked_at=utcnow(), account=f"{s.imap_user} @ {s.imap_host}:{s.imap_port}",
                       folders=folders if folders is not None else IMAP_STATUS.get("folders", []))


def _connect(s_cfg, timeout: int = 60) -> imaplib.IMAP4_SSL:
    """Connect + login; raises imaplib.IMAP4.error / OSError with an actionable message."""
    try:
        imap = imaplib.IMAP4_SSL(s_cfg.imap_host, s_cfg.imap_port, timeout=timeout)
    except OSError as exc:
        raise OSError(f"cannot reach {s_cfg.imap_host}:{s_cfg.imap_port} ({exc}). Check IMAP_HOST/IMAP_PORT (usually 993).") from exc
    try:
        imap.login(s_cfg.imap_user, s_cfg.imap_password.get_secret_value())
    except imaplib.IMAP4.error as exc:
        try:
            imap.logout()
        except Exception:  # noqa: BLE001
            pass
        raise imaplib.IMAP4.error(f"login failed ({exc.args[0].decode() if exc.args and isinstance(exc.args[0], bytes) else exc}). {_PROVIDER_HINT}") from exc
    return imap


def check_imap() -> dict[str, Any]:
    """Connection test: connect, log in, list the folders that would be scanned. Never changes the mailbox."""
    s_cfg = get_settings()
    if not s_cfg.imap_configured:
        _set_status(False, "IMAP not configured — set IMAP_HOST, IMAP_USER and IMAP_PASSWORD in .env", [])
        return dict(IMAP_STATUS)
    try:
        imap = _connect(s_cfg, timeout=20)
        try:
            folders = [decode_mailbox_name(f) for f in mailbox_folders(imap, s_cfg.imap_folder)]
        finally:
            try:
                imap.logout()
            except Exception:  # noqa: BLE001
                pass
        _set_status(True, None, folders)
    except (imaplib.IMAP4.error, OSError) as exc:
        _set_status(False, str(exc))
    log.info("IMAP check: %s", "connected" if IMAP_STATUS["ok"] else IMAP_STATUS["error"])
    return dict(IMAP_STATUS)


def poll_mailbox() -> dict[str, Any]:
    s_cfg = get_settings()
    if not s_cfg.imap_configured:
        return {"skipped": "IMAP not configured"}
    prefs = get_prefs()
    parsers = all_parsers(prefs.email.enabled_parsers, prefs.email.generic_senders)
    senders = sorted({p for parser in parsers for p in parser.sender_patterns})
    with session_scope() as s:
        run = SourceRun(source_name=f"email:{s_cfg.imap_user}", adapter="imap")
        s.add(run)
        s.flush()
        run_id = run.id

    totals = {"messages": 0, "found": 0, "new": 0, "merged": 0, "updated": 0, "folders": 0}
    errors: list[str] = []
    try:
        imap = _connect(s_cfg)
        try:
            scanned: list[str] = []
            since = (utcnow() - timedelta(days=s_cfg.imap_since_days)).strftime("%d-%b-%Y")
            with session_scope() as s:
                known = set(s.scalars(select(EmailMessage.message_id)))
            budget = prefs.email.max_messages_per_run
            for folder in mailbox_folders(imap, s_cfg.imap_folder):
                typ, _ = imap.select(quote_mailbox(folder), readonly=True)  # read-only: never changes flags
                if typ != "OK":
                    errors.append(f"folder {decode_mailbox_name(folder)}: cannot select")
                    continue
                totals["folders"] += 1
                scanned.append(decode_mailbox_name(folder))
                uids: list[bytes] = []
                for sender in senders:
                    typ, data = imap.uid("search", None, f'(SINCE {since} FROM "{sender}")')
                    if typ == "OK" and data and data[0]:
                        uids.extend(data[0].split())
                for uid in list(dict.fromkeys(uids))[-budget:] if budget > 0 else []:
                    typ, hdr = imap.uid("fetch", uid, "(BODY.PEEK[HEADER.FIELDS (MESSAGE-ID)])")
                    mid = ""
                    if typ == "OK" and hdr and isinstance(hdr[0], tuple):
                        mid = email.message_from_bytes(hdr[0][1]).get("Message-ID", "").strip()
                    if mid and mid in known:  # same alert in several folders (e.g. Gmail labels) → once
                        continue
                    typ, data = imap.uid("fetch", uid, "(BODY.PEEK[])")
                    if typ != "OK" or not data or not isinstance(data[0], tuple):
                        continue
                    budget -= 1
                    if mid:
                        known.add(mid)
                    try:
                        res = process_eml_bytes(data[0][1])
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{decode_mailbox_name(folder)} uid {uid.decode()}: {exc}")
                        continue
                    totals["messages"] += 1
                    for k in ("found", "new", "merged", "updated"):
                        totals[k] += int(res.get(k, 0) or 0)
        finally:
            try:
                imap.logout()
            except Exception:  # noqa: BLE001
                pass
        _set_status(True, None, scanned)
    except (imaplib.IMAP4.error, OSError) as exc:
        errors.append(f"IMAP: {exc}")
        _set_status(False, str(exc))
        log.warning("Mailbox poll failed: %s", exc)

    with session_scope() as s:
        run = s.get(SourceRun, run_id)
        run.finished_at = utcnow()
        run.success = not any(e.startswith("IMAP") for e in errors)
        run.jobs_found = totals["found"]
        run.jobs_new = totals["new"]
        run.jobs_merged = totals["merged"]
        run.jobs_updated = totals["updated"]
        run.error_count = len(errors)
        run.errors = errors[:20]
    return {**totals, "errors": len(errors)}
