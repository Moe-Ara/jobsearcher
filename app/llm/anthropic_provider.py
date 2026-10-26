"""Anthropic (Claude) provider — structured outputs validated against a Pydantic model, plus plain text."""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from app.llm.base import LLMError, LLMProvider, LLMUnavailable, LLMResponse, T, TextResponse

log = logging.getLogger(__name__)

# Models that accept output_config.effort and the server-side refusal fallback ("default" form).
_EFFORT_PREFIXES = ("claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-opus-4-8", "claude-opus-4-7")
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None, *, timeout: float = 120.0, refusal_fallback: bool = True) -> None:
        # api_key=None lets the SDK resolve credentials itself (ANTHROPIC_API_KEY, auth profile, ...).
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=3)
        self.refusal_fallback = refusal_fallback

    def default_models(self) -> tuple[str, str]:
        return "claude-opus-5-5", "claude-haiku-4-5"

    def _kwargs(self, model: str, system: str, user: str, max_tokens: int, effort: str | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            # The system prompt (+ candidate profile) is identical for every job → cache it.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
        }
        if effort and model.startswith(_EFFORT_PREFIXES):
            kwargs["output_config"] = {"effort": effort}
        if self.refusal_fallback and model in _FALLBACK_MODELS:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def _send(self, method, kwargs: dict[str, Any]):
        try:
            resp = method(**kwargs)
        except anthropic.RateLimitError as exc:
            raise LLMError(f"Rate limited by Anthropic: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailable(f"Cannot reach Anthropic API: {exc}") from exc
        except anthropic.AnthropicError as exc:  # e.g. output failed schema validation client-side
            raise LLMError(f"Anthropic SDK error: {exc}") from exc
        if resp.stop_reason == "refusal":
            raise LLMError("Model declined the request (refusal)")
        if resp.stop_reason == "max_tokens":
            raise LLMError("Output truncated (max_tokens reached)")
        return resp

    def structured(self, *, model, system, user, schema: type[T], max_tokens=4000, effort=None, options=None) -> LLMResponse[T]:
        kwargs = self._kwargs(model, system, user, max_tokens, effort)
        kwargs["output_format"] = schema
        resp = self._send(self.client.beta.messages.parse, kwargs)
        parsed = resp.parsed_output
        if parsed is None:
            raise LLMError("No structured output returned")
        usage = resp.usage
        return LLMResponse(
            parsed=parsed,
            model=resp.model or model,
            input_tokens=(usage.input_tokens or 0) + (usage.cache_creation_input_tokens or 0),
            output_tokens=usage.output_tokens or 0,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
        )

    def text(self, *, model, system, user, max_tokens=2000, effort=None, options=None) -> TextResponse:
        resp = self._send(self.client.beta.messages.create, self._kwargs(model, system, user, max_tokens, effort))
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text").strip()
        if not text:
            raise LLMError("Empty response")
        usage = resp.usage
        return TextResponse(
            text=text,
            model=resp.model or model,
            input_tokens=(usage.input_tokens or 0) + (usage.cache_creation_input_tokens or 0),
            output_tokens=usage.output_tokens or 0,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
        )
