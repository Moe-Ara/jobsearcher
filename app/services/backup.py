"""Database backups: a consistent copy of the SQLite database (SQLite's online backup API — safe while the app
runs) into <data dir>/backups, once a day and on demand. The newest `keep` copies are kept.

In Docker the database lives on its own volume (fast Linux file system), so these copies in data/backups are the
ones you can see, copy or restore from. Restore: stop the app, copy a backup over /db/jobsearcher.db (see README).
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def backups_dir() -> Path:
    from app.settings import data_dir

    return data_dir() / "backups"


def _db_path() -> Path | None:
    from app.settings import get_settings

    url = get_settings().database_url
    if not url.startswith("sqlite") or "///" not in url:
        return None  # PostgreSQL etc.: use the database's own backup tooling
    return Path(url.split("///", 1)[-1])


def list_backups() -> list[Path]:
    d = backups_dir()
    return sorted(d.glob("jobsearcher-*.db"), reverse=True) if d.exists() else []


def backup_database(keep: int = 7) -> dict[str, Any]:
    src = _db_path()
    if src is None or not src.exists():
        return {"skipped": "not a SQLite database file"}
    target_dir = backups_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"jobsearcher-{datetime.now():%Y%m%d-%H%M%S}.db"
    tmp = target.with_suffix(".partial")
    source = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True, timeout=30)
    dest = sqlite3.connect(tmp)
    try:
        source.backup(dest, pages=2048)  # copies in steps: writers are only briefly paused
    finally:
        dest.close()
        source.close()
    tmp.replace(target)
    removed = 0
    for old in list_backups()[keep:]:
        old.unlink(missing_ok=True)
        removed += 1
    log.info("Database backup written: %s (%.1f MB)", target.name, target.stat().st_size / 1e6)
    return {"file": target.name, "mb": round(target.stat().st_size / 1e6, 1), "removed_old": removed}


def backup_if_due(max_age_hours: int = 24) -> dict[str, Any]:
    latest = next(iter(list_backups()), None)
    if latest and datetime.fromtimestamp(latest.stat().st_mtime) > datetime.now() - timedelta(hours=max_age_hours):
        return {"skipped": f"latest backup {latest.name} is recent"}
    return backup_database()
