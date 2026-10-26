"""OpenAI / OpenAI-compatible provider (OpenAI, Azure-compatible proxies, Ollama, vLLM, LM Studio, OpenRouter)."""

from __future__ import annotations

import json
import logging

import openai
from pydantic import ValidationError

from app.llm.base import LLMError, LLMProvider, LLMUnavailable, LLMResponse, T, TextResponse

log = logging.getLogger(__name__)


def _strict_schema(schema: dict) -> dict:
    """Make a Pydantic JSON schema acceptable to strict json_schema mode (all required, no extras)."""

    def fix(node):
        if isinstance(node, dict):
            node.pop("default", None)
            node.pop("title", None)
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)
        return node

    return fix(json.loads(json.dumps(schema)))


class OpenAICompatibleProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str | None, base_url: str | None, *, timeout: float = 120.0) -> None:
        self.client = openai.OpenAI(api_key=api_key or "not-needed", base_url=base_url or None, timeout=timeout, max_retries=3)
        self.base_url = base_url
        self._json_schema_supported = True
        self.is_paid = not base_url or "openai.com" in base_url or "openrouter" in base_url

    def default_models(self) -> tuple[str, str]:
        return "gpt-4.1", "gpt-4.1-mini"

    def _call(self, model, messages, max_tokens, response_format):
        return self.client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, response_format=response_format, temperature=0.1
        )

    def text(self, *, model, system, user, max_tokens=2000, effort=None, options=None) -> TextResponse:
        try:
            resp = self.client.chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=(options or {}).get("temperature", 0.4),
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except openai.RateLimitError as exc:
            raise LLMError(f"Rate limited: {exc}") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") from exc
        except openai.APIConnectionError as exc:
            raise LLMUnavailable(f"Cannot reach LLM endpoint: {exc}") from exc
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            raise LLMError("Empty response")
        usage = resp.usage
        return TextResponse(
            text=content, model=resp.model or model,
            input_tokens=(usage.prompt_tokens or 0) if usage else 0, output_tokens=(usage.completion_tokens or 0) if usage else 0,
        )

    def structured(self, *, model, system, user, schema: type[T], max_tokens=4000, effort=None, options=None) -> LLMResponse[T]:
        json_schema = _strict_schema(schema.model_json_schema())
        system_full = (
            f"{system}\n\nReturn ONLY a JSON object that conforms to this JSON schema:\n{json.dumps(json_schema)}"
        )
        messages = [{"role": "system", "content": system_full}, {"role": "user", "content": user}]
        in_tok = out_tok = 0
        last_error = ""
        for attempt in range(2):  # one repair retry on invalid JSON
            try:
                if self._json_schema_supported:
                    rf = {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": json_schema, "strict": True}}
                    try:
                        resp = self._call(model, messages, max_tokens, rf)
                    except openai.BadRequestError as exc:
                        log.info("json_schema response_format rejected (%s); falling back to json_object", exc)
                        self._json_schema_supported = False
                        resp = self._call(model, messages, max_tokens, {"type": "json_object"})
                else:
                    resp = self._call(model, messages, max_tokens, {"type": "json_object"})
            except openai.RateLimitError as exc:
                raise LLMError(f"Rate limited: {exc}") from exc
            except openai.APIStatusError as exc:
                raise LLMError(f"API error {exc.status_code}: {exc.message}") from exc
            except openai.APIConnectionError as exc:
                raise LLMUnavailable(f"Cannot reach LLM endpoint: {exc}") from exc

            if resp.usage:
                in_tok += resp.usage.prompt_tokens or 0
                out_tok += resp.usage.completion_tokens or 0
            choice = resp.choices[0]
            content = choice.message.content or ""
            if getattr(choice.message, "refusal", None):
                raise LLMError(f"Model refused: {choice.message.refusal}")
            try:
                parsed = schema.model_validate_json(_extract_json(content))
                return LLMResponse(parsed=parsed, model=resp.model or model, input_tokens=in_tok, output_tokens=out_tok)
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)[:500]
                messages += [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"That was not valid against the schema: {last_error}. Return only the corrected JSON object."},
                ]
        raise LLMError(f"Invalid structured output after retry: {last_error}")


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in response")
    return text[start : end + 1]
