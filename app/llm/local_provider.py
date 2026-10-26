"""LocalLLMProvider — local GPU inference through an Ollama server (separate container / process).

Uses Ollama's native /api/chat because it supports what the local pipeline needs and the OpenAI shim doesn't
expose cleanly: JSON-schema constrained output (`format`), per-request `keep_alive`, `num_ctx`, and Qwen3's
`think` switch. Ollama loads/unloads models on demand; with OLLAMA_MAX_LOADED_MODELS=1 only one model is
resident at a time, so callers batch work per model (see services.analysis).

All requests go through GPU_LOCK so an interactive request (cover letter) and a background batch never
interleave model swaps request-by-request.
"""

from __future__ import annotations

import json
import contextlib
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import ValidationError

from app.llm.base import LLMError, LLMProvider, LLMResponse, LLMUnavailable, T, TextResponse

log = logging.getLogger(__name__)

# Re-entrant so a pipeline can hold it across several calls (letter generation) while each call also takes it.
GPU_LOCK = threading.RLock()

# Interactive runs (a letter or an application you started and wait for) have priority over background batches:
# while one is active, background requests pause before taking the GPU — so a scheduled analysis batch can't keep the
# GPU between the agent's stages (or swap models under it).
_priority = threading.local()
_interactive_runs = 0
_counter_lock = threading.Lock()


@contextlib.contextmanager
def interactive():
    """Mark the current thread's pipeline as interactive for its whole duration (re-entrant)."""
    global _interactive_runs
    depth = getattr(_priority, "depth", 0)
    _priority.depth = depth + 1
    if depth == 0:
        with _counter_lock:
            _interactive_runs += 1
    try:
        yield
    finally:
        _priority.depth = depth
        if depth == 0:
            with _counter_lock:
                _interactive_runs -= 1


def interactive_active() -> bool:
    return _interactive_runs > 0


def wait_for_interactive(max_wait_s: float = 3600) -> None:
    """Background callers wait here while an interactive run is active (never the interactive thread itself)."""
    if getattr(_priority, "depth", 0):
        return
    waited = 0.0
    while _interactive_runs > 0 and waited < max_wait_s:
        time.sleep(0.5)
        waited += 0.5

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)



def is_qwen3(model: str) -> bool:
    """Qwen3 base models and JobSearcher's own fine-tunes of them (finetune/: jobsearcher-*-ft) share the think switch."""
    name = model.lower()
    return "qwen3" in name or name.split("/")[-1].startswith("jobsearcher-")

def log_timing(model: str, data: dict[str, Any], waited: float) -> None:
    """One line per call: where the seconds went (queue, model load, prompt prefill, generation)."""
    ns = 1e9
    load = (data.get("load_duration") or 0) / ns
    pre_n, pre_s = int(data.get("prompt_eval_count") or 0), (data.get("prompt_eval_duration") or 0) / ns
    gen_n, gen_s = int(data.get("eval_count") or 0), (data.get("eval_duration") or 0) / ns
    log.info("ollama %s: wait %.1fs · load %.1fs · prefill %d tok %.1fs (%.0f tok/s) · gen %d tok %.1fs (%.1f tok/s)",
             model.split("/")[-1], waited, load, pre_n, pre_s, pre_n / pre_s if pre_s else 0, gen_n, gen_s,
             gen_n / gen_s if gen_s else 0)


@dataclass
class LocalHealthState:
    last_success_at: datetime | None = None
    last_success_model: str | None = None
    last_latency_ms: int | None = None
    last_error: str | None = None
    last_error_at: datetime | None = None
    requests: int = 0
    failures: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def ok(self, model: str, latency_ms: int) -> None:
        with self.lock:
            self.last_success_at, self.last_success_model, self.last_latency_ms = datetime.now(UTC).replace(tzinfo=None), model, latency_ms
            self.requests += 1

    def fail(self, message: str) -> None:
        with self.lock:
            self.last_error, self.last_error_at = message[:500], datetime.now(UTC).replace(tzinfo=None)
            self.requests += 1
            self.failures += 1


HEALTH = LocalHealthState()


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve $defs/$ref so grammar converters that don't follow references still constrain nested objects."""
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            return {k: resolve(v) for k, v in node.items() if k not in ("$defs", "title")}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def extract_json(text: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    if text.startswith("```"):
        text = text.strip("`")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in model output")
    return text[start : end + 1]


def clean_text_output(text: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    return text


class LocalLLMProvider(LLMProvider):
    name = "local"
    is_paid = False  # local GPU time is free in the cost tracker

    def __init__(self, base_url: str, *, timeout: float = 900.0, transport: httpx.BaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            base_url=self.base_url, timeout=httpx.Timeout(timeout, connect=10.0), transport=transport
        )
        self._no_think_param: set[str] = set()  # models whose template rejects the `think` field

    def default_models(self) -> tuple[str, str]:  # pragma: no cover - roles are configured explicitly
        raise NotImplementedError("Local models are configured per role")

    # ------------------------------------------------------------------ transport

    def _chat(self, model: str, messages: list[dict[str, str]], *, fmt: dict | None, max_tokens: int,
              options: dict[str, Any] | None) -> dict[str, Any]:
        opts = dict(options or {})
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": opts.get("keep_alive", "5m"),
            "options": self._sampling(opts, max_tokens),
        }
        if fmt is not None:
            body["format"] = fmt
        think = opts.get("think")
        if think is not None:
            if model in self._no_think_param:
                self._soft_think_switch(model, body, think)
            else:
                body["think"] = think

        started = time.monotonic()
        wait_for_interactive()
        with GPU_LOCK:
            waited = time.monotonic() - started
            resp = self._post(body)
            if resp.status_code == 400 and "think" in body and "think" in resp.text.lower():
                # Model template doesn't support the API switch → use Qwen3's in-prompt soft switch instead.
                log.info("Model %s rejects the 'think' parameter; using prompt soft switch", model)
                self._no_think_param.add(model)
                body.pop("think")
                self._soft_think_switch(model, body, think)
                resp = self._post(body)
        latency = int((time.monotonic() - started) * 1000)
        data = self._check(resp, model, latency)
        log_timing(model, data, waited)
        return data

    @staticmethod
    def _sampling(opts: dict[str, Any], max_tokens: int) -> dict[str, Any]:
        keys = ("num_ctx", "temperature", "top_p", "top_k", "min_p", "repeat_penalty")
        out = {k: opts.get(k) for k in keys}
        out["num_predict"] = max_tokens
        return {k: v for k, v in out.items() if v is not None}

    def _check(self, resp: httpx.Response, model: str, latency: int) -> dict[str, Any]:
        if resp.status_code == 404:
            msg = f"Local model '{model}' is not installed in Ollama — run scripts/setup-local-models"
            HEALTH.fail(msg)
            raise LLMUnavailable(msg)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", resp.text)
            except ValueError:
                detail = resp.text
            msg = f"Local inference error {resp.status_code}: {str(detail)[:300]}"
            HEALTH.fail(msg)
            raise LLMError(msg)
        try:
            data = resp.json()
        except ValueError as exc:
            HEALTH.fail("Invalid JSON from local inference server")
            raise LLMError("Invalid JSON from local inference server") from exc
        HEALTH.ok(model, latency)
        data["_latency_ms"] = latency
        return data

    @staticmethod
    def _soft_think_switch(model: str, body: dict[str, Any], think: bool) -> None:
        if is_qwen3(model):
            body["messages"][-1]["content"] += "\n/think" if think else "\n/no_think"

    def _post(self, body: dict[str, Any], path: str = "/api/chat") -> httpx.Response:
        try:
            return self.client.post(path, json=body)
        except httpx.TimeoutException as exc:
            msg = f"Local model '{body['model']}' timed out ({exc.__class__.__name__})"
            HEALTH.fail(msg)
            raise LLMError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"Local inference server unreachable at {self.base_url}: {exc}"
            HEALTH.fail(msg)
            raise LLMUnavailable(msg) from exc

    # ------------------------------------------------------------------ API

    def structured(self, *, model, system, user, schema: type[T], max_tokens=4000, effort=None, options=None) -> LLMResponse[T]:
        json_schema = inline_refs(schema.model_json_schema())
        system_full = (
            f"{system}\n\nRespond ONLY with a JSON object that conforms to this JSON schema "
            f"(field descriptions explain what to put in each field):\n{json.dumps(json_schema, ensure_ascii=False)}"
        )
        messages = [{"role": "system", "content": system_full}, {"role": "user", "content": user}]
        in_tok = out_tok = 0
        last_error = ""
        for _attempt in range(2):  # one repair round on schema-invalid output
            data = self._chat(model, messages, fmt=json_schema, max_tokens=max_tokens, options=options)
            in_tok += int(data.get("prompt_eval_count") or 0)
            out_tok += int(data.get("eval_count") or 0)
            content = (data.get("message") or {}).get("content") or ""
            try:
                parsed = schema.model_validate_json(extract_json(content))
                return LLMResponse(parsed=parsed, model=data.get("model") or model, input_tokens=in_tok, output_tokens=out_tok)
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)[:400]
                messages = messages + [
                    {"role": "assistant", "content": content[:4000]},
                    {"role": "user", "content": f"That output was invalid: {last_error}. Return only the corrected JSON object."},
                ]
        HEALTH.fail(f"Invalid structured output from {model}: {last_error}")
        raise LLMError(f"Invalid structured output from local model after retry: {last_error}")

    def text(self, *, model, system, user, max_tokens=2000, effort=None, options=None) -> TextResponse:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if (options or {}).get("think") is False and is_qwen3(model):
            # Qwen3 non-thinking mode = an empty think block before the answer. Ollama's template for hf.co GGUFs
            # doesn't insert it for think=false (verified on Ollama 0.35), so the model would reason in plain text;
            # prefilling the assistant turn reproduces Qwen3's official non-thinking template.
            messages.append({"role": "assistant", "content": "<think>\n\n</think>\n\n"})
        data = self._chat(model, messages, fmt=None, max_tokens=max_tokens, options=options)
        text = clean_text_output((data.get("message") or {}).get("content") or "")
        if not text:
            raise LLMError(f"Empty response from local model {model}")
        return TextResponse(
            text=text, model=data.get("model") or model,
            input_tokens=int(data.get("prompt_eval_count") or 0), output_tokens=int(data.get("eval_count") or 0),
        )

    def complete_raw(self, *, model: str, prompt: str, max_tokens: int = 2048,
                     options: dict[str, Any] | None = None) -> TextResponse:
        """Plain text completion with NO chat template (Ollama raw mode) — for completion-trained models whose
        prompt must match their training format byte for byte (e.g. the humanizer). No stop strings: EOS only."""
        opts = dict(options or {})
        body = {
            "model": model, "prompt": prompt, "raw": True, "stream": False,
            "keep_alive": opts.get("keep_alive", "5m"), "options": self._sampling(opts, max_tokens),
        }
        started = time.monotonic()
        wait_for_interactive()
        with GPU_LOCK:
            resp = self._post(body, "/api/generate")
        data = self._check(resp, model, int((time.monotonic() - started) * 1000))
        text = (data.get("response") or "").strip()
        if not text:
            raise LLMError(f"Empty completion from local model {model}")
        return TextResponse(text=text, model=data.get("model") or model,
                            input_tokens=int(data.get("prompt_eval_count") or 0), output_tokens=int(data.get("eval_count") or 0))

    def health(self, timeout: float = 3.0) -> dict[str, Any]:
        """Cheap reachability check: server version, installed models, models currently in memory."""
        out: dict[str, Any] = {"reachable": False, "version": None, "installed": [], "loaded": [], "error": None}
        try:
            out["version"] = self.client.get("/api/version", timeout=timeout).json().get("version")
            out["installed"] = [m.get("name") for m in self.client.get("/api/tags", timeout=timeout).json().get("models", [])]
            out["loaded"] = [
                {"name": m.get("name"), "size_vram_gb": round((m.get("size_vram") or 0) / 1e9, 1), "expires_at": m.get("expires_at")}
                for m in self.client.get("/api/ps", timeout=timeout).json().get("models", [])
            ]
            out["reachable"] = True
        except (httpx.HTTPError, ValueError) as exc:
            out["error"] = f"{exc.__class__.__name__}: {exc}"
        return out

    def cached_health(self, max_age: float = 30.0) -> dict[str, Any]:
        now = time.monotonic()
        cached = getattr(self, "_health_cache", None)
        if cached and now - cached[0] < max_age:
            return cached[1]
        result = self.health()
        self._health_cache = (now, result)
        return result

    def is_installed(self, model: str, installed: list[str]) -> bool:
        names = set(installed)
        return model in names or f"{model}:latest" in names
