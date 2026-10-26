"""Test package. Importing anything under tests/ (even outside pytest) must never touch the real database:
the live data/jobsearcher.db is shared with the Docker app, and a second SQLite process on the Windows host breaks
its WAL journal. So the temporary test database is configured here, before any test module imports the app.
The tests also never read your own config/: they use tests/fixtures/config (a fictional candidate, "Alex Beispiel")."""

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="jobsearcher-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["SCHEDULER_ENABLED"] = "false"
os.environ["CONFIG_DIR"] = str(Path(__file__).parent / "fixtures" / "config")
# The suite never depends on your .env: every setting the tests assume is set here (environment beats .env).
os.environ["LOCAL_LLM_ENABLED"] = "true"  # local model roles configured (never called: tests use a fake provider)
for _role in ("SCREEN", "ANALYSIS", "WRITER", "GERMAN_EDITOR", "HUMANIZER"):
    os.environ[f"LOCAL_{_role}_ENABLED"] = "true"
os.environ["CLOUD_FALLBACK_ENABLED"] = "false"
for _key in ("APP_PASSWORD", "CAREERJET_API_KEY", "JOOBLE_API_KEY", "REED_API_KEY", "ADZUNA_APP_ID", "ADZUNA_APP_KEY",
             "SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"):
    os.environ[_key] = ""
