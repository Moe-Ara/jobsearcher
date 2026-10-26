"""Environment-based settings (secrets live here, never in code or YAML)."""

from __future__ import annotations

import threading
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'jobsearcher.db').as_posix()}"
    config_dir: Path = PROJECT_ROOT / "config"
    # Documents, exported applications, caches and backups. Default: the folder of the SQLite database. In Docker
    # the database has its own fast volume (/db) and this stays the visible ./data folder.
    data_dir: Path | None = None
    # Optional sign-in for every page and the API (see app/web/auth.py) — set it before exposing the app to a network.
    app_password: SecretStr | None = None
    log_level: str = "INFO"
    scheduler_enabled: bool = True

    # LLM
    llm_provider: str = "auto"  # auto | anthropic | openai | heuristic
    anthropic_api_key: SecretStr | None = None
    anthropic_refusal_fallback: bool = True
    openai_api_key: SecretStr | None = None
    # Job search APIs (free keys) for the Gulf aggregators — see app/sources/careerjet.py and jooble.py
    careerjet_api_key: SecretStr | None = None
    careerjet_user_ip: str = "127.0.0.1"  # Careerjet asks for the end user's IP; this is a personal tool
    jooble_api_key: SecretStr | None = None
    reed_api_key: SecretStr | None = None  # UK: reed.co.uk/developers/jobseeker
    adzuna_app_id: str | None = None  # UK + Germany: developer.adzuna.com
    adzuna_app_key: SecretStr | None = None
    openai_base_url: str | None = None
    llm_analysis_model: str = ""
    llm_triage_model: str = ""
    llm_timeout_seconds: float = 120.0

    # Local inference (Ollama). Model-role → model mapping lives here; per-role runtime options
    # (context size, keep-alive, temperature, thinking) live in preferences.yaml → local_llm.
    local_llm_enabled: bool = False
    local_llm_base_url: str = "http://ollama:11434"
    local_llm_timeout_seconds: float = 900.0  # first load of a 24B model with CPU offload can be slow
    local_screen_model: str = "hf.co/Qwen/Qwen3-8B-GGUF:Q5_K_M"
    local_analysis_model: str = "hf.co/Qwen/Qwen3-14B-GGUF:Q5_K_M"
    local_writer_model: str = "hf.co/Qwen/Qwen3-14B-GGUF:Q5_K_M"
    local_german_editor_model: str = "mistral-small-3.1-text:q4_k_s"  # text-only copy, see scripts/setup-local-models
    local_validator_model: str = ""  # empty = the writer model (fast, fits fully in VRAM)
    # Optional "humanizer" rewrite pass (dedicated completion model; English/Chinese-trained, German tested separately)
    local_humanizer_model: str = "hf.co/jialinyyzz/humanizer:Q6_K"
    local_screen_enabled: bool = True
    local_analysis_enabled: bool = True
    local_writer_enabled: bool = True
    local_german_editor_enabled: bool = True
    local_humanizer_enabled: bool = True
    # When a local call fails, retry it on the configured cloud provider (Anthropic / OpenAI-compatible).
    cloud_fallback_enabled: bool = False

    # IMAP email alerts
    imap_host: str = ""
    imap_port: int = 993
    imap_user: str = ""
    imap_password: SecretStr | None = None
    imap_folder: str = "*"  # "*" = all folders except trash/sent/drafts; or a comma-separated list
    imap_since_days: int = 14
    # Sending applications (only on your command). Empty = derived from IMAP (imap.web.de → smtp.web.de, same login).
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: SecretStr | None = None

    # HTTP fetching
    http_user_agent: str = (
        "Mozilla/5.0 (compatible; JobSearcher/1.0; personal job discovery tool)"
    )
    http_timeout_seconds: float = 25.0
    http_min_interval_seconds: float = 1.0  # politeness delay per host

    @property
    def imap_configured(self) -> bool:
        return bool(self.imap_host and self.imap_user and self.imap_password)


_current: Settings | None = None
_lock = threading.Lock()


def get_settings() -> Settings:
    global _current
    with _lock:
        if _current is None:
            _current = Settings()
        return _current


def data_dir() -> Path:
    s = get_settings()
    if s.data_dir:
        return Path(s.data_dir)
    url = s.database_url
    return Path(url.split("///", 1)[-1]).parent if url.startswith("sqlite") and "///" in url else PROJECT_ROOT / "data"


def reload_settings() -> Settings:
    """Re-read .env from disk (edited after start) without restarting.

    In Docker, compose injects .env as process environment at container start, and those values would normally win
    over the file. Here the file's values take precedence, so edits made since then become effective immediately.
    """
    global _current
    from dotenv import dotenv_values

    overrides = {}
    path = PROJECT_ROOT / ".env"
    if path.is_file():
        for key, value in dotenv_values(path).items():
            if value is not None and key.lower() in Settings.model_fields:
                overrides[key.lower()] = value
    with _lock:
        _current = Settings(**overrides)
        return _current
