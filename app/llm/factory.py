"""Builds the LLMSetup (role routing) from environment settings + preferences. Cached; reset on config reload."""

from __future__ import annotations

import logging
import threading

from app.config_files import get_prefs
from app.llm.base import LLMProvider
from app.llm.roles import LLMSetup, Role, RoleTarget, build_setup
from app.settings import get_settings

log = logging.getLogger(__name__)

__all__ = ["LLMSetup", "Role", "RoleTarget", "get_llm", "reset_llm", "build_cloud"]

_lock = threading.Lock()
_cached: LLMSetup | None = None


def get_llm() -> LLMSetup:
    global _cached
    with _lock:
        if _cached is None:
            _cached = _build()
        return _cached


def reset_llm() -> None:
    global _cached
    with _lock:
        _cached = None


def build_cloud() -> tuple[LLMProvider | None, tuple[str, str] | None]:
    """The configured cloud provider (Anthropic / OpenAI-compatible) and its (strong, cheap) models, if any."""
    s = get_settings()
    choice = s.llm_provider.lower().strip()
    if choice == "auto":
        if s.anthropic_api_key and s.anthropic_api_key.get_secret_value():
            choice = "anthropic"
        elif (s.openai_api_key and s.openai_api_key.get_secret_value()) or s.openai_base_url:
            choice = "openai"
        else:
            choice = "none"
    provider: LLMProvider | None = None
    if choice == "anthropic":
        from app.llm.anthropic_provider import AnthropicProvider

        key = s.anthropic_api_key.get_secret_value() if s.anthropic_api_key else None
        provider = AnthropicProvider(key or None, timeout=s.llm_timeout_seconds, refusal_fallback=s.anthropic_refusal_fallback)
    elif choice == "openai":
        from app.llm.openai_provider import OpenAICompatibleProvider

        key = s.openai_api_key.get_secret_value() if s.openai_api_key else None
        provider = OpenAICompatibleProvider(key, s.openai_base_url, timeout=s.llm_timeout_seconds)
    elif choice not in ("heuristic", "none", "local"):
        raise ValueError(f"Unknown LLM_PROVIDER '{s.llm_provider}'")
    if provider is None:
        return None, None
    strong, cheap = provider.default_models()
    return provider, (s.llm_analysis_model or strong, s.llm_triage_model or cheap)


def _build() -> LLMSetup:
    s = get_settings()
    local = None
    if s.local_llm_enabled:
        from app.llm.local_provider import LocalLLMProvider

        local = LocalLLMProvider(s.local_llm_base_url, timeout=s.local_llm_timeout_seconds)
    cloud, cloud_models = build_cloud()
    setup = build_setup(s, get_prefs(), local=local, cloud=cloud, cloud_models=cloud_models)
    if not setup.has_llm:
        log.warning("No LLM configured — using the free heuristic analyzer (enable LOCAL_LLM or set an API key).")
    for role in Role:
        t = setup.target(role)
        log.info("LLM role %-9s → %s%s", role, t.label if t else "none",
                 f" (fallback {t.fallback.label})" if t and t.fallback else "")
    return setup
