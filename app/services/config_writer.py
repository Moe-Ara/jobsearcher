"""Write settings back into the YAML files from the web UI — keeping your comments and layout (ruamel round-trip).

Every write is validated first (the new file must load into the app's models), the previous version is kept in
config/.backups/ (newest 20 per file), and the in-memory config is reloaded.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

log = logging.getLogger(__name__)

KEEP_BACKUPS = 20


class ConfigError(ValueError):
    pass


def _yaml() -> YAML:
    y = YAML()  # round-trip: comments, key order and quoting survive
    y.preserve_quotes = True
    y.width = 120
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def config_path(name: str) -> Path:
    from app.settings import get_settings

    return Path(get_settings().config_dir) / name


def _validate(name: str, text: str) -> None:
    """The new text must load the way the app loads it — else nothing is written."""
    import yaml

    from app.config_files import CandidateProfile, Preferences
    from app.services.cv import MasterCV

    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Not valid YAML: {exc}") from exc
    model = {"preferences.yaml": Preferences, "profile.yaml": CandidateProfile, "cv.yaml": MasterCV}.get(name)
    if model is not None:
        try:
            model.model_validate(data)
        except Exception as exc:  # pydantic ValidationError — shown to the user as is
            raise ConfigError(str(exc)) from exc


def _backup(path: Path) -> None:
    if not path.exists():
        return
    d = path.parent / ".backups"
    d.mkdir(exist_ok=True)
    (d / f"{path.stem}-{datetime.now():%Y%m%d-%H%M%S}{path.suffix}").write_bytes(path.read_bytes())
    for old in sorted(d.glob(f"{path.stem}-*{path.suffix}"), reverse=True)[KEEP_BACKUPS:]:
        old.unlink(missing_ok=True)


def write_text(name: str, text: str) -> None:
    """Replace a config file with `text` (validated, previous version backed up)."""
    from app.config_files import config_store

    _validate(name, text)
    path = config_path(name)
    _backup(path)
    path.write_text(text, encoding="utf-8")
    config_store.reload()
    log.info("Config written: %s", name)


def update(name: str, changes: dict[str, Any]) -> None:
    """Set dotted keys ("locations.home", "internships.max_drive_minutes") in a config file, keeping everything
    else — comments included — exactly as it is. A value of None removes the key."""
    y = _yaml()
    path = config_path(name)
    doc = y.load(path.read_text(encoding="utf-8")) if path.exists() else None
    if doc is None:
        from ruamel.yaml.comments import CommentedMap

        doc = CommentedMap()
    for dotted, value in changes.items():
        *parents, leaf = dotted.split(".")
        node = doc
        for key in parents:
            if key not in node or node[key] is None:
                node[key] = {}
            node = node[key]
        if value is None:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    buf = io.StringIO()
    y.dump(doc, buf)
    write_text(name, buf.getvalue())
