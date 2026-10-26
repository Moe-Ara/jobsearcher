"""Provider-neutral LLM interface: structured JSON (validated with Pydantic) and plain-text generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Any failure of a provider call (network, refusal, invalid output, model missing)."""


class LLMUnavailable(LLMError):
    """The backend itself is unavailable (server unreachable, model not installed) — not the job's fault.
    Batch runs stop on this instead of counting a failed attempt against every job."""


@dataclass
class LLMResponse(Generic[T]):
    parsed: T
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class TextResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0


class LLMProvider(ABC):
    name: str
    is_paid: bool = True

    @abstractmethod
    def structured(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 4000,
        effort: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMResponse[T]:
        """Return `schema`-validated output. `system` is stable; `user` carries untrusted data in delimiters."""

    @abstractmethod
    def text(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 2000,
        effort: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> TextResponse:
        """Free-form text generation (used for letter editing / correction)."""

    def default_models(self) -> tuple[str, str]:
        """(analysis_model, triage_model)"""
        raise NotImplementedError
