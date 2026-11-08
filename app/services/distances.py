"""How far each job is from your home (preferences.yaml → locations.home): the job list sorts by priority, then
distance. Straight-line kilometres from the job's town (OpenStreetMap, cached in data/geocode.json); fully remote
jobs count as 0 km. set_distance() only reads the cache (safe inside a transaction); resolve_job_distances() looks
up new towns between two short transactions (run before each analysis batch).
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select

log = logging.getLogger(__name__)


def _home() -> tuple[float, float] | None:
    from app.services.drive_time import home_point

    _, lat, lon = home_point()
    return (lat, lon) if lat is not None and lon is not None else None


def job_query(job) -> str | None:
    from app.services.drive_time import place_query

    return place_query(job.location, job.country)


def set_distance(job, home: tuple[float, float] | None = None, cache: dict | None = None) -> float | None:
    """From the cache only. Remote → 0 km. Unknown town / no home → None (sorted after the known ones)."""
    from app.services.drive_time import _load, straight_km

    home = home or _home()
    if home is None:
        return None
    if (job.remote_status or "") == "remote":
        job.distance_km = 0.0
        return 0.0
    query = job_query(job)
    if not query:
        return None
    hit = (cache if cache is not None else _load("geocode.json")).get(query.lower())
    if isinstance(hit, list):
        job.distance_km = round(straight_km(home, (hit[0], hit[1])), 1)
    return job.distance_km


def resolve_job_distances(limit: int = 150) -> dict[str, Any]:
    from app.db import session_scope
    from app.models import Job
    from app.services.drive_time import COUNTRY_SEARCH, _load, geocode

    home = _home()
    if home is None:
        return {"skipped": "no home set"}
    with session_scope() as s:
        rows = [(j.id, job_query(j), j.country) for j in s.scalars(
            select(Job).where(Job.active.is_(True), Job.distance_km.is_(None), Job.track.in_(("main", "intern"))))]
    cache = _load("geocode.json")
    todo: dict[str | None, list[str]] = {}
    for _, query, country in rows:
        if query and query.lower() not in cache:
            todo.setdefault(COUNTRY_SEARCH.get(country or "Germany", (None, None))[1], []).append(query)
    looked_up = 0
    for cc, queries in todo.items():
        if looked_up >= limit:
            break
        found = geocode(list(dict.fromkeys(queries)), limit=limit - looked_up, countrycodes=cc)  # network
        looked_up += len(found)
    cache = _load("geocode.json")
    updated = 0
    ids = [job_id for job_id, _, _ in rows]
    for i in range(0, len(ids), 200):
        with session_scope() as s:
            for job in s.scalars(select(Job).where(Job.id.in_(ids[i:i + 200]))):
                if set_distance(job, home, cache) is not None:
                    updated += 1
    return {"jobs_without_distance": len(rows), "looked_up": looked_up, "updated": updated}
