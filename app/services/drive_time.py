"""Driving time from home to a job's location (side jobs and internships must be reachable by car).

Places without coordinates (company job boards, LinkedIn: only a town name) are geocoded with OpenStreetMap
Nominatim — at most one request per second, every place cached permanently (data/geocode.json).
Uses OSRM (open-source road routing, router.project-osrm.org) in batches of up to 90 destinations, with a permanent
cache per place (data/drive_times.json) — each town is looked up once. If the router is unreachable, a cautious
estimate from the straight-line distance is used (road ≈ 1.35 × straight line at ≈ 65 km/h, +5 min) and marked so.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

OSRM = "https://router.project-osrm.org/table/v1/driving/"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
BATCH = 90
_lock = threading.Lock()


def _cache_path(name: str = "drive_times.json") -> Path:
    from app.services.application_agent import documents_dir

    return documents_dir().parent / name


def _load(name: str = "drive_times.json") -> dict:
    try:
        return json.loads(_cache_path(name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(cache: dict, name: str = "drive_times.json") -> None:
    try:
        _cache_path(name).write_text(json.dumps(cache, indent=0, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        log.info("Could not save the %s cache: %s", name, exc)


_COUNTRY_ONLY = re.compile(r"^(germany|deutschland|de|bundesweit|remote|homeoffice|home office|europe|emea|dach)$", re.I)


def place_query(location: str | None) -> str | None:
    """'Friedrichshafen, Baden-Württemberg, Germany (Hybrid)' → 'Friedrichshafen, Baden-Württemberg, Germany';
    a country alone ('Germany (Remote)') names no place → None."""
    text = re.sub(r"\([^)]*\)", " ", location or "")
    text = re.split(r"[;|/]| und | oder | or |\+", text)[0]
    text = re.sub(r"\s+", " ", text).strip(" ,-·")
    parts = [p.strip() for p in text.split(",") if p.strip() and not _COUNTRY_ONLY.match(p.strip())]
    if not parts:
        return None
    return ", ".join(parts) + ", Deutschland"


def cached_place(query: str | None) -> tuple[float, float] | None | bool:
    """Coordinates from the cache; None = known to be unfindable; False = not looked up yet."""
    if not query:
        return None
    hit = _load("geocode.json").get(query.lower(), False)
    return tuple(hit) if isinstance(hit, list) else hit


def geocode(queries: Iterable[str], http=None, limit: int = 60) -> dict[str, tuple[float, float] | None]:
    """Look up places that are not cached yet (≤ 1 request / second, at most `limit` per call). Network only —
    never call this inside a database transaction."""
    from app.core.http import get_http

    http = http or get_http()
    out: dict[str, tuple[float, float] | None] = {}
    with _lock:
        cache = _load("geocode.json")
        todo = [q for q in dict.fromkeys(queries) if q and q.lower() not in cache][:limit]
        for q in todo:
            hit = None
            for attempt in (q, q.split(",")[0] + ", Deutschland"):  # "Stuttgart, Friedrichshafen" → first town
                try:
                    rows = http.get_json(NOMINATIM, params={"q": attempt, "format": "jsonv2", "limit": 1, "countrycodes": "de"})
                except Exception as exc:  # noqa: BLE001 - best effort: retried at the next run
                    log.info("Geocoding %r failed: %s", attempt, exc)
                    rows = None
                    break
                finally:
                    time.sleep(1.1)  # Nominatim usage policy: at most one request per second
                if rows:
                    hit = [round(float(rows[0]["lat"]), 4), round(float(rows[0]["lon"]), 4)]
                    break
                if attempt == q.split(",")[0] + ", Deutschland":
                    break
            if rows is None:
                continue  # network trouble: not cached, tried again later
            cache[q.lower()] = hit
            out[q] = tuple(hit) if hit else None
        _save(cache, "geocode.json")
    for q in queries:
        if q and q not in out and isinstance(cache.get(q.lower()), list):
            out[q] = tuple(cache[q.lower()])
    return out


def countries_of_towns(towns, http=None, limit: int = 40) -> dict[str, str | None]:
    """Country of each bare town name ("Breda", "Zug", "Emstek") — OpenStreetMap, any country, cached permanently
    (data/town_countries.json), ≤ 1 request / second, at most `limit` new look-ups per call. Network only — never
    inside a database transaction. Unknown / not found → None."""
    from app.core.http import get_http
    from app.core.normalize import canonical_country

    http = http or get_http()
    towns = [t.strip() for t in dict.fromkeys(towns) if t and t.strip()]
    with _lock:
        cache = _load("town_countries.json")
        for town in [t for t in towns if t.lower() not in cache][:limit]:
            try:
                rows = http.get_json(NOMINATIM, params={"q": town, "format": "jsonv2", "limit": 1, "addressdetails": 1})
            except Exception as exc:  # noqa: BLE001 - best effort: tried again next time
                log.info("Town lookup %r failed: %s", town, exc)
                break
            finally:
                time.sleep(1.1)  # Nominatim usage policy
            code = ((rows or [{}])[0].get("address") or {}).get("country_code", "") if rows else ""
            cache[town.lower()] = canonical_country(code.upper()) if code else None
        _save(cache, "town_countries.json")
    return {t: cache.get(t.lower()) for t in towns}


def home_point(prefs=None) -> tuple[str, float | None, float | None]:
    """(town, lat, lon) of your home: locations.home (older files: side_jobs.home). Coordinates looked up from the
    town (cache only — the settings page geocodes it when you save) when not given. lat/lon None = unknown."""
    if prefs is None:
        from app.config_files import get_prefs

        prefs = get_prefs()
    loc, sp = prefs.locations, prefs.side_jobs
    town = loc.home or sp.home or ""
    lat = loc.home_lat if loc.home_lat is not None else sp.home_lat
    lon = loc.home_lon if loc.home_lon is not None else sp.home_lon
    if (lat is None or lon is None) and town:
        hit = cached_place(place_query(town))
        if hit:
            lat, lon = hit
    return town, lat, lon


def cached_minutes(home: tuple[float, float], place: tuple[float, float]) -> float | None:
    return _load().get(key(*home) + "->" + key(*place))


def key(lat: float, lon: float) -> str:
    return f"{lat:.3f},{lon:.3f}"  # ~100 m: one lookup per street / village


def straight_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def estimate_minutes(home: tuple[float, float], dest: tuple[float, float]) -> float:
    return round(straight_km(home, dest) * 1.35 / 65 * 60 + 5, 1)


def drive_minutes(home: tuple[float, float], places: Iterable[tuple[float, float]], http=None) -> dict[str, float]:
    """{key(lat, lon): minutes by car} for every place (cached; new places routed in batches)."""
    from app.core.http import get_http

    http = http or get_http()
    places = {key(*p): p for p in places if p and None not in p}
    with _lock:
        cache = _load()
        prefix = key(*home) + "->"
        missing = [(k, p) for k, p in places.items() if prefix + k not in cache]
        for i in range(0, len(missing), BATCH):
            chunk = missing[i:i + BATCH]
            coords = ";".join(f"{lon:.5f},{lat:.5f}" for lat, lon in [home] + [p for _, p in chunk])
            try:
                data = http.get_json(f"{OSRM}{coords}", params={"sources": 0, "annotations": "duration"})
                durations = (data.get("durations") or [[]])[0][1:]
                for (k, p), sec in zip(chunk, durations):
                    cache[prefix + k] = round(sec / 60, 1) if sec is not None else estimate_minutes(home, p)
            except Exception as exc:  # noqa: BLE001 - routing is best effort: estimate instead (not cached)
                log.info("Drive-time routing failed (%s) — using straight-line estimates", exc)
                for k, p in chunk:
                    places[k] = p
                break
        _save(cache)
    return {k: cache.get(prefix + k, estimate_minutes(home, p)) for k, p in places.items()}
