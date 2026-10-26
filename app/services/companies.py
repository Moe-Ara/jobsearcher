"""Target companies: feed checks (with per-job error isolation and run tracking), YAML import, URL onboarding."""

from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config_files import get_prefs, load_companies_yaml
from app.core.http import get_http
from app.core.normalize import canonical_country, normalize_company
from app.db import session_scope
from app.enums import JobStatus
from app.models import Job, JobSource, SourceRun, TargetCompany, utcnow
from app.services.ingest import ingest_posting, invalidate_alias_cache
from app.services.retention import is_tombstoned
from app.services.jobs import set_status
from app.sources.base import CompanyFeedConfig
from app.sources.registry import ADAPTERS, build_adapter, detect_ats

log = logging.getLogger(__name__)

MAX_ERRORS_STORED = 20


def feed_config(tc: TargetCompany) -> CompanyFeedConfig:
    return CompanyFeedConfig(
        company_name=tc.company_name,
        identifier=tc.ats_identifier,
        careers_url=tc.careers_url,
        options=dict(tc.ats_config or {}),
        country=tc.country,
        city=tc.city,
    )


def country_gate(cfg: CompanyFeedConfig):
    """Feeds of international employers list thousands of jobs worldwide. A posting whose location names only
    countries outside preferences.yaml → locations.target_countries is skipped (not stored, not analysed). Unknown or
    remote-without-country locations pass (the hard filters decide). ats_config.all_countries: true turns it off."""
    from app.core.normalize import canonical_country, infer_country

    if cfg.options.get("all_countries"):
        return lambda text: True
    targets = {canonical_country(c) or c for c in get_prefs().locations.target_countries}

    def keep(text: str | None) -> bool:
        parts = [p.strip() for p in re.split(r"[;|]|\s/\s", text or "") if p.strip()]
        found = {infer_country(p) for p in parts} - {None}
        return not found or bool(found & targets)

    return keep


def check_company(company_id: int) -> dict[str, Any]:
    """Fetch one company's feed and ingest everything. Never raises for source problems."""
    with session_scope() as s:
        tc = s.get(TargetCompany, company_id)
        if tc is None:
            return {"error": "unknown company"}
        if not tc.ats_type or tc.ats_type not in ADAPTERS:
            return {"error": f"No supported ATS configured for {tc.company_name}"}
        run = SourceRun(source_name=f"{tc.ats_type}:{tc.company_name}", adapter=tc.ats_type, target_company_id=tc.id)
        s.add(run)
        s.flush()
        run_id, cfg, ats_type, name = run.id, feed_config(tc), tc.ats_type, tc.company_name

    adapter = build_adapter(ats_type, cfg, get_http())
    errors: list[str] = []
    counts = {"new": 0, "merged": 0, "updated": 0, "unchanged": 0}
    seen_keys: set[str] = set()
    fetch_ok = False
    raws: list[dict[str, Any]] = []
    try:
        raws = adapter.fetch_jobs()
        fetch_ok = True
    except Exception as exc:  # noqa: BLE001 - one bad source must not stop the pipeline
        log.warning("Feed fetch failed for %s: %s", name, exc)
        errors.append(f"fetch: {exc}")

    keep_country = country_gate(cfg)
    for raw in raws:
        try:
            if not keep_country(adapter.location_hint(raw)):
                counts["foreign"] = counts.get("foreign", 0) + 1  # outside your target countries: not stored
                continue
            with session_scope() as s:
                tc = s.get(TargetCompany, company_id)
                ext = adapter.external_id(raw)
                known = ext is not None and s.scalars(
                    select(JobSource.id).where(JobSource.source_key == f"{adapter.source_name()}:{ext}"[:600])
                ).first()
                if not known and ext is not None and is_tombstoned(s, f"{adapter.source_name()}:{ext}"[:600]):
                    counts["purged"] = counts.get("purged", 0) + 1  # deleted by retention: no detail fetch
                    continue
                if not known:
                    raw = adapter.fetch_details(raw)
                posting = adapter.normalize_job(raw)
                if cfg.options.get("track") in ("side", "intern"):
                    posting.track = cfg.options["track"]
                if not keep_country(posting.country or posting.location):
                    counts["foreign"] = counts.get("foreign", 0) + 1
                    continue
                result = ingest_posting(s, posting, target_company=tc, employer_from_posting=bool(cfg.options.get("employer_from_posting")))
                if result.source is not None:
                    seen_keys.add(result.source.source_key)
                counts[result.outcome] = counts.get(result.outcome, 0) + 1
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to ingest a %s posting for %s: %s", ats_type, name, exc)
            errors.append(f"posting {adapter.external_id(raw) if isinstance(raw, dict) else '?'}: {exc}")

    deactivated = 0
    with session_scope() as s:
        tc = s.get(TargetCompany, company_id)
        if fetch_ok and not adapter.partial_listing:
            deactivated = _deactivate_missing(s, tc, adapter.source_name(), seen_keys, len(raws))
        tc.last_checked = utcnow()
        tc.last_status = "ok" if fetch_ok and len(errors) < max(3, len(raws) // 2) else "error"
        tc.last_error = "\n".join(errors[:5]) or None
        tc.last_jobs_found = len(raws) if fetch_ok else tc.last_jobs_found
        run = s.get(SourceRun, run_id)
        run.finished_at = utcnow()
        run.success = tc.last_status == "ok"
        run.jobs_found = len(raws)
        run.jobs_new = counts["new"]
        run.jobs_merged = counts["merged"]
        run.jobs_updated = counts["updated"]
        run.jobs_deactivated = deactivated
        run.error_count = len(errors)
        run.errors = errors[:MAX_ERRORS_STORED]
    log.info("Checked %s: %d found, %d new, %d merged, %d errors", name, len(raws), counts["new"], counts["merged"], len(errors))
    return {"company": name, "found": len(raws), **counts, "deactivated": deactivated, "errors": len(errors)}


def _deactivate_missing(s: Session, tc: TargetCompany, source: str, seen_keys: set[str], found: int) -> int:
    active_sources = s.scalars(
        select(JobSource).where(JobSource.target_company_id == tc.id, JobSource.source == source, JobSource.active.is_(True))
    ).all()
    missing = [js for js in active_sources if js.source_key not in seen_keys]
    if found == 0 and len(active_sources) > 5:
        log.warning("%s returned 0 jobs but %d were active — not deactivating (likely a feed problem)", tc.company_name, len(active_sources))
        return 0
    jobs_touched = set()
    for js in missing:
        js.active = False
        jobs_touched.add(js.job_id)
    for job_id in jobs_touched:
        job = s.get(Job, job_id)
        if job and not any(src.active for src in job.sources):
            job.active = False
            if job.status == JobStatus.NEW:
                set_status(s, job, JobStatus.EXPIRED, note=f"no longer listed on {source}")
    return len(missing)


def due_company_ids() -> list[int]:
    hours = get_prefs().scheduler.company_check_hours
    now = utcnow()
    with session_scope() as s:
        out = []
        for tc in s.scalars(select(TargetCompany).where(TargetCompany.active.is_(True))):
            if not tc.ats_type or tc.ats_type not in ADAPTERS:
                continue
            interval = timedelta(hours=float(hours.get(tc.priority, 24)))
            if tc.last_checked is None or now - tc.last_checked >= interval:
                out.append(tc.id)
        return out


def check_companies(ids: list[int]) -> dict[str, Any]:
    results = [check_company(i) for i in ids]
    return {
        "companies": len(results),
        "new": sum(r.get("new", 0) for r in results),
        "merged": sum(r.get("merged", 0) for r in results),
        "errors": sum(1 for r in results if r.get("errors") or r.get("error")),
    }


# --------------------------------------------------------------------------- create / import


def upsert_company(s: Session, data: dict[str, Any], *, overwrite: bool = False) -> tuple[TargetCompany, bool]:
    name = (data.get("company_name") or data.get("name") or "").strip()
    if not name:
        raise ValueError("company_name is required")
    norm = normalize_company(name)
    tc = s.scalars(select(TargetCompany).where(TargetCompany.normalized_name == norm)).first()
    created = tc is None
    if tc is None:
        tc = TargetCompany(company_name=name, normalized_name=norm, aliases=[], ats_config={})
        s.add(tc)
    elif not overwrite:
        return tc, False

    careers_url = data.get("careers_url") or None
    ats_type = data.get("ats_type") or None
    ident = data.get("ats_identifier") or None
    config = dict(data.get("ats_config") or {})
    if careers_url and not ats_type:
        m = detect_ats(careers_url)
        ats_type, ident = m.ats_type, ident or m.identifier
        config = {**(m.options or {}), **config}
    def given(key: str) -> bool:  # on update, only touch fields the caller actually provided
        return created or (key in data and data[key] not in (None, ""))

    tc.company_name = name
    tc.normalized_name = norm
    if given("aliases"):
        tc.aliases = [a.strip() for a in (data.get("aliases") or []) if a and a.strip()]
    if given("country"):
        tc.country = canonical_country(data.get("country")) or data.get("country") or None
    if given("city"):
        tc.city = data.get("city") or None
    if given("sector"):
        tc.sector = data.get("sector") or None
    if given("priority"):
        tc.priority = (data.get("priority") or "B").upper()[:1]
    if given("careers_url") or given("ats_type"):
        tc.careers_url = careers_url
        tc.ats_type = ats_type
        tc.ats_identifier = ident
        tc.ats_config = config
    if given("active"):
        tc.active = bool(data.get("active", True))
    if given("notes"):
        tc.notes = data.get("notes") or None
    invalidate_alias_cache()
    return tc, created


def import_companies_yaml() -> dict[str, int]:
    created = 0
    with session_scope() as s:
        for row in load_companies_yaml():
            try:
                _, was_created = upsert_company(s, row)
                created += int(was_created)
            except ValueError as exc:
                log.warning("Skipping company row %s: %s", row, exc)
    return {"created": created}


def seed_if_empty() -> None:
    with session_scope() as s:
        if s.scalars(select(TargetCompany.id).limit(1)).first():
            return
    result = import_companies_yaml()
    log.info("Seeded %d target companies from config/companies.yaml", result["created"])
