"""One path for every LLM call: role target → (cache) → budget → provider → fallback → JobAnalysis record.

Every call is recorded in job_analyses (the existing usage/cost tracker) with provider, model, prompt version,
tokens, latency and cost. Local calls cost 0; cloud calls (including cloud fallback) use the price table.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime
from typing import Any

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config_files import get_prefs
from app.llm.base import LLMError
from app.llm.roles import RoleTarget
from app.models import JobAnalysis, utcnow

log = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CallResult:
    data: Any  # dict for structured calls, str for text calls
    record: JobAnalysis
    target: RoleTarget  # the target that actually answered (primary or fallback)

    @property
    def cached(self) -> bool:
        return bool(self.record.cached)


def input_hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def spent_today(session: Session) -> float:
    start = datetime.combine(utcnow().date(), dtime.min)
    return float(
        session.scalar(select(func.coalesce(func.sum(JobAnalysis.cost_usd), 0.0)).where(JobAnalysis.created_at >= start))
        or 0.0
    )


def _cached(session: Session, kind: str, key: str) -> JobAnalysis | None:
    return session.scalars(
        select(JobAnalysis)
        .where(JobAnalysis.kind == kind, JobAnalysis.input_hash == key, JobAnalysis.error.is_(None),
               JobAnalysis.result.is_not(None))
        .order_by(JobAnalysis.created_at.desc())
        .limit(1)
    ).first()


def llm_call(
    session: Session,
    target: RoleTarget,
    *,
    kind: str,
    job_id: int | None,
    prompt_version: str,
    profile_hash: str,
    system: str,
    user: str,
    schema: type[BaseModel] | None = None,
    use_cache: bool = True,
) -> CallResult:
    """Structured call when `schema` is given, plain text otherwise. Raises LLMError / BudgetExceeded."""
    prefs = get_prefs()
    chain = [target] + ([target.fallback] if target.fallback else [])
    for i, t in enumerate(chain):
        key = input_hash(kind, prompt_version, t.provider.name, t.model, profile_hash, system, user)
        if use_cache and (hit := _cached(session, kind, key)):
            rec = JobAnalysis(
                job_id=job_id, kind=kind, provider=hit.provider, model=hit.model, prompt_version=prompt_version,
                profile_hash=profile_hash, input_hash=key, result=hit.result, cached=True,
            )
            session.add(rec)
            session.flush()
            return CallResult(_unwrap(hit.result, schema), rec, t)

        budget = prefs.llm.daily_budget_usd
        if t.provider.is_paid and budget > 0 and spent_today(session) >= budget:
            raise BudgetExceeded(f"Daily LLM budget of ${budget:.2f} reached")

        rec = JobAnalysis(
            job_id=job_id, kind=kind, provider=t.provider.name, model=t.model, prompt_version=prompt_version,
            profile_hash=profile_hash, input_hash=key,
        )
        # SQLite has one writer at a time: never hold a write transaction while a model thinks (30-90 s locally),
        # or every other task (mail poll, retention, feed checks) times out with "database is locked".
        session.commit()
        started = time.monotonic()
        try:
            if schema is not None:
                resp = t.provider.structured(model=t.model, system=system, user=user, schema=schema,
                                             max_tokens=t.max_tokens, effort=t.effort, options=t.options)
                data: Any = resp.parsed.model_dump()
                rec.result = data
            else:
                resp = t.provider.text(model=t.model, system=system, user=user, max_tokens=t.max_tokens,
                                       effort=t.effort, options=t.options)
                data = resp.text
                rec.result = {"text": data}
        except LLMError as exc:
            rec.error = str(exc)[:2000]
            rec.duration_ms = int((time.monotonic() - started) * 1000)
            session.add(rec)
            session.commit()
            if i + 1 < len(chain):
                log.warning("%s call on %s failed (%s) — falling back to %s", kind, t.label, exc, chain[i + 1].label)
                continue
            raise
        rec.model = resp.model or t.model
        rec.input_tokens = resp.input_tokens
        rec.output_tokens = resp.output_tokens
        rec.cache_read_tokens = getattr(resp, "cache_read_tokens", 0) or 0
        if t.provider.is_paid:  # cache reads are billed at ~10% of the input price
            cost = prefs.llm.cost(t.model, resp.input_tokens, resp.output_tokens) + 0.1 * prefs.llm.cost(t.model, rec.cache_read_tokens, 0)
            rec.cost_usd = round(cost, 6)
        else:
            rec.cost_usd = 0.0
        rec.duration_ms = int((time.monotonic() - started) * 1000)
        session.add(rec)
        session.commit()  # release the write lock right away (the caller's next step may be another long call)
        from app.services import toolbox

        toolbox.note({"tool": f"llm:{kind}", "ok": True, "seconds": round(rec.duration_ms / 1000, 2),
                      "note": f"{t.model.split('/')[-1]} · {rec.output_tokens} tokens"})
        return CallResult(data, rec, t)
    raise LLMError("No LLM target available")  # pragma: no cover - chain is never empty


def _unwrap(result: Any, schema: type[BaseModel] | None) -> Any:
    if schema is None and isinstance(result, dict):
        return result.get("text", "")
    return result
