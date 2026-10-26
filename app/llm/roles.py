"""Model roles → (provider, model, options). The only place that decides which backend serves what.

Roles:
  screen     cheap structured screening of every new job        (local: Qwen3-8B)
  extract    field extraction from pasted text                  (follows screen)
  analysis   deep fit analysis of promising jobs                 (local: Qwen3-14B)
  writer     cover-letter drafts                                 (local: Qwen3-14B)
  editor     German style editing of a factually correct letter  (local: Mistral Small 3.1 24B)
  validator  factual claim validation of letters                 (local: Qwen3-14B, the writer model)
  humanizer  optional natural-rewrite pass (letters + standalone) (local only: jialinyyzz/humanizer)

Resolution per role: local (if LOCAL_LLM_ENABLED and the role is enabled) → otherwise the cloud provider
(LLM_PROVIDER / API keys) → otherwise none (screen/analysis then use the free heuristic). With
CLOUD_FALLBACK_ENABLED, local targets get the cloud target as fallback for failed calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from app.config_files import Preferences
from app.llm.base import LLMProvider
from app.settings import Settings


class Role(StrEnum):
    SCREEN = "screen"
    EXTRACT = "extract"
    ANALYSIS = "analysis"
    WRITER = "writer"
    EDITOR = "editor"
    VALIDATOR = "validator"
    HUMANIZER = "humanizer"


# Roles served only by a dedicated local model (no cloud equivalent → no cloud target / fallback).
LOCAL_ONLY_ROLES = {Role.HUMANIZER}


# Cloud model tier and effort per role (cloud providers have one strong + one cheap model).
_CLOUD_TIER: dict[Role, tuple[str, str | None]] = {
    Role.SCREEN: ("cheap", "low"),
    Role.EXTRACT: ("cheap", "low"),
    Role.ANALYSIS: ("strong", "medium"),
    Role.WRITER: ("strong", "medium"),
    Role.EDITOR: ("strong", "medium"),
    Role.VALIDATOR: ("strong", "low"),
    Role.HUMANIZER: ("strong", None),  # unused: local-only role
}


@dataclass
class RoleTarget:
    role: Role
    provider: LLMProvider
    model: str
    options: dict[str, Any] = field(default_factory=dict)
    effort: str | None = None
    fallback: RoleTarget | None = None

    @property
    def is_local(self) -> bool:
        return self.provider.name == "local"

    @property
    def max_tokens(self) -> int:
        return int(self.options.get("max_tokens") or 4000)

    @property
    def label(self) -> str:
        return f"{self.provider.name}:{self.model}"


@dataclass
class LLMSetup:
    targets: dict[Role, RoleTarget | None] = field(default_factory=dict)
    local: LLMProvider | None = None
    cloud: LLMProvider | None = None
    local_ctx: dict[str, int] = field(default_factory=dict)  # model → unified num_ctx
    prefs: Preferences | None = None

    def target(self, role: Role | str) -> RoleTarget | None:
        return self.targets.get(Role(role))

    @property
    def has_llm(self) -> bool:
        return self.target(Role.SCREEN) is not None or self.target(Role.ANALYSIS) is not None

    @property
    def provider_name(self) -> str:
        names = sorted({t.provider.name for t in self.targets.values() if t})
        return "+".join(names) if names else "heuristic"

    def retarget(self, base: RoleTarget, role: Role) -> RoleTarget:
        """Same provider+model as `base`, with `role`'s options — used to validate with the model already loaded."""
        configured = self.target(role)
        if configured and configured.model:
            return configured
        opts = dict(base.options)
        if self.prefs and base.is_local:
            opts.update(self.prefs.local_llm.options(role).model_dump())
            opts["num_ctx"] = self.local_ctx.get(base.model, opts.get("num_ctx"))
        effort = _CLOUD_TIER[role][1] if not base.is_local else None
        return replace(base, role=role, options=opts, effort=effort)

    @classmethod
    def single(cls, provider: LLMProvider | None, analysis_model: str, triage_model: str) -> LLMSetup:
        """Everything on one (cloud-style) provider: strong model + cheap model. Used for cloud mode and tests."""
        if provider is None:
            return cls()
        return cls(targets={r: None if r in LOCAL_ONLY_ROLES else _cloud_target(r, provider, analysis_model, triage_model)
                            for r in Role}, cloud=provider)


def _cloud_target(role: Role, provider: LLMProvider, strong: str, cheap: str) -> RoleTarget:
    tier, effort = _CLOUD_TIER[role]
    max_tokens = {Role.SCREEN: 1200, Role.EXTRACT: 500, Role.ANALYSIS: 6000, Role.WRITER: 3000,
                  Role.EDITOR: 3000, Role.VALIDATOR: 3000, Role.HUMANIZER: 2048}[role]
    return RoleTarget(role, provider, strong if tier == "strong" else cheap, {"max_tokens": max_tokens}, effort)


def local_role_models(settings: Settings) -> dict[Role, tuple[str, bool]]:
    """Role → (model, enabled) from LOCAL_* settings."""
    return {
        Role.SCREEN: (settings.local_screen_model, settings.local_screen_enabled),
        Role.EXTRACT: (settings.local_screen_model, settings.local_screen_enabled),
        Role.ANALYSIS: (settings.local_analysis_model, settings.local_analysis_enabled),
        Role.WRITER: (settings.local_writer_model, settings.local_writer_enabled),
        Role.EDITOR: (settings.local_german_editor_model, settings.local_german_editor_enabled),
        # Default: the writer model (Qwen3-14B fits fully in VRAM → ~4x faster fact checks than the partly
        # CPU-offloaded 24B editor; the extra model swap costs ~10 s).
        Role.VALIDATOR: (settings.local_validator_model or settings.local_writer_model, settings.local_writer_enabled),
        Role.HUMANIZER: (settings.local_humanizer_model, settings.local_humanizer_enabled),
    }


def build_setup(
    settings: Settings,
    prefs: Preferences,
    *,
    local: LLMProvider | None,
    cloud: LLMProvider | None,
    cloud_models: tuple[str, str] | None,
) -> LLMSetup:
    role_models = local_role_models(settings)
    # Roles sharing a model share the largest num_ctx: Ollama reloads a model when num_ctx changes.
    local_ctx: dict[str, int] = {}
    if local:
        for role, (model, enabled) in role_models.items():
            if model and enabled:
                local_ctx[model] = max(local_ctx.get(model, 0), prefs.local_llm.options(role).num_ctx)

    targets: dict[Role, RoleTarget | None] = {}
    for role in Role:
        cloud_t = _cloud_target(role, cloud, *cloud_models) if cloud and cloud_models and role not in LOCAL_ONLY_ROLES else None
        model, enabled = role_models[role]
        if local and enabled:
            opts = prefs.local_llm.options(role).model_dump()
            if model:
                opts["num_ctx"] = local_ctx[model]
            t = RoleTarget(role, local, model, opts)
            if settings.cloud_fallback_enabled and cloud_t:
                t.fallback = cloud_t
            targets[role] = t
        else:
            targets[role] = cloud_t
    return LLMSetup(targets=targets, local=local, cloud=cloud, local_ctx=local_ctx, prefs=prefs)
