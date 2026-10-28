"""Bundesagentur für Arbeit — Jobsuche API (the public API behind arbeitsagentur.de/jobsuche).

ats_config:
  searches: list of searches, each a dict of API parameters plus options, e.g.
      {label: Teilzeit & Minijob in der Nähe, wo: "88045 Friedrichshafen", umkreis: 25, arbeitszeit: "tz;mj"}
      {label: Befristet in der Nähe, wo: "88045 Friedrichshafen", umkreis: 25, befristung: 1}
      {label: Remote IT Teilzeit, was: "Homeoffice IT", arbeitszeit: "tz;mj", require_remote: true, max_pages: 3}
    API parameters: was, wo, umkreis (km), arbeitszeit (vz;tz;snw;mj), befristung (1 befristet, 2 unbefristet),
    angebotsart (1 = Arbeit, default), veroeffentlichtseit (days, default 14), zeitarbeit (true/false)
    options: require_remote (keep only postings whose text says home office / remote — for nationwide searches),
             max_pages (100 per page, default 5)
  queries_from_preferences: true — one nationwide search per term in preferences.yaml → search.queries
      (main career search; `search_defaults` are added to each, e.g. {veroeffentlichtseit: 7, zeitarbeit: false})
  track: side  (part-time / temporary work: own filters and ranking — app/services/side_jobs.py)
         intern (internships, angebotsart 34: app/services/internships.py)
"""

from __future__ import annotations

import base64
import re
from datetime import datetime
from typing import Any

from app.core.text import clean_line, clean_text
from app.sources.base import JobPosting, JobSourceAdapter

API = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service"
HEADERS_KEY = "jobboerse-jobsuche"  # the public client id of arbeitsagentur.de's own job search
_OPTIONS = {"label", "require_remote", "max_pages"}
_HOMEOFFICE = re.compile(r"home.?office|remote|mobile[sn]? arbeiten|ortsunabhängig|telearbeit|100\s?% mobil", re.I)
# fully remote, not "Homeoffice an einem Tag pro Woche möglich"
_FULLY_REMOTE = re.compile(r"100\s?%\s*(remote|home.?office|mobil)|(voll|komplett|ausschließlich|vollständig|rein)\w*\s+"
                           r"(remote|im home.?office|von zu ?hause|mobil)|full(y)?[ -]remote|ortsunabhängig|remote[- ]first", re.I)


class ArbeitsagenturAdapter(JobSourceAdapter):
    name = "arbeitsagentur"
    # searches with a date window (veroeffentlichtseit) and page caps: a posting missing from one run is not closed
    partial_listing = True

    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        return self.http.get_json_with_headers(url, params=params, headers={"X-API-Key": HEADERS_KEY})

    def fetch_jobs(self) -> list[dict[str, Any]]:
        out, seen = [], set()
        searches = list(self.company.options.get("searches") or [])
        if self.company.options.get("queries_from_preferences"):
            from app.sources.workable_board import _queries

            defaults = {"veroeffentlichtseit": 7, "zeitarbeit": "false", "max_pages": 1,
                        **(self.company.options.get("search_defaults") or {})}
            searches += [{"label": q, "was": q, **defaults} for q in _queries()]
        for search in searches:
            params = {k: v for k, v in search.items() if k not in _OPTIONS}
            near_me = self.company.options.get("track") in ("side", "intern")
            if near_me and "wo" not in params and not search.get("require_remote"):  # "near me" searches: your home
                from app.config_files import get_prefs
                from app.services.drive_time import home_point

                town = home_point(get_prefs())[0]
                if town:
                    params["wo"] = town
            params.setdefault("angebotsart", 1)
            params.setdefault("veroeffentlichtseit", 14)
            for page in range(1, int(search.get("max_pages", 5)) + 1):
                data = self._get(f"{API}/pc/v6/jobs", {**params, "page": page, "size": 100})
                items = data.get("ergebnisliste") or []
                for item in items:
                    ref = item.get("referenznummer")
                    if ref and ref not in seen:
                        seen.add(ref)
                        out.append({**item, "_search": search.get("label") or "", "_require_remote": bool(search.get("require_remote"))})
                if len(items) < 100:
                    break
        if self.company.options.get("track") in ("side", "intern"):
            self._add_drive_times(out)
        return out

    def _add_drive_times(self, items: list[dict[str, Any]]) -> None:
        """Real driving time from home for every posting with coordinates (one batched routing call)."""
        from app.config_files import get_prefs
        from app.services.drive_time import drive_minutes, key

        from app.services.drive_time import home_point

        _, home_lat, home_lon = home_point(get_prefs())
        if home_lat is None or home_lon is None:
            return  # no home set: no drive times (the filters then judge by town)
        coords = {}
        for item in items:
            loc = (item.get("stellenlokationen") or [{}])[0]
            if loc.get("breite") is not None and loc.get("laenge") is not None:
                coords[item["referenznummer"]] = (float(loc["breite"]), float(loc["laenge"]))
        if not coords:
            return
        minutes = drive_minutes((home_lat, home_lon), coords.values(), self.http)
        for item in items:
            c = coords.get(item.get("referenznummer"))
            if c:
                item["_drive_min"] = minutes.get(key(*c))

    def external_id(self, raw: dict[str, Any]) -> str | None:
        return raw.get("referenznummer")

    def location_hint(self, raw: dict[str, Any]) -> str:
        loc = ((raw.get("stellenlokationen") or [{}])[0].get("adresse") or {})
        return ", ".join(x for x in (loc.get("ort"), "Germany" if loc.get("land") in (None, "DEUTSCHLAND") else loc.get("land")) if x)

    def fetch_details(self, raw: dict[str, Any]) -> dict[str, Any]:
        ref = base64.b64encode(raw["referenznummer"].encode()).decode()
        try:
            details = self._get(f"{API}/pc/v4/jobdetails/{ref}")
        except Exception:  # noqa: BLE001 - the listing alone is still a usable posting
            details = {}
        return {**raw, **{k: v for k, v in (details or {}).items() if v not in (None, "")}}

    def normalize_job(self, raw: dict[str, Any]) -> JobPosting:
        desc = clean_text(raw.get("stellenangebotsBeschreibung") or "")
        title = clean_line(raw.get("stellenangebotsTitel") or raw.get("hauptberuf") or "")
        remote = bool(_HOMEOFFICE.search(title) or _FULLY_REMOTE.search(f"{title}\n{desc}"))
        partly = not remote and bool(_HOMEOFFICE.search(desc))  # some home office days
        part_time = any(raw.get(k) for k in ("arbeitszeitTeilzeitVormittag", "arbeitszeitTeilzeitNachmittag",
                                             "arbeitszeitTeilzeitAbend", "arbeitszeitTeilzeitFlexibel"))
        minijob = bool(raw.get("istGeringfuegigeBeschaeftigung"))
        fixed = (raw.get("vertragsdauer") or "").upper().startswith("BEFRIST")
        kinds = [k for k, on in (("Minijob", minijob), ("Teilzeit", part_time), ("befristet", fixed), ("Remote", remote), ("Homeoffice teilweise", partly)) if on]
        start = (raw.get("eintrittszeitraum") or {}).get("von")
        posted = raw.get("datumErsteVeroeffentlichung")
        location = self.location_hint(raw)
        return JobPosting(
            source=self.name, source_type=self.source_type, external_job_id=raw.get("referenznummer"),
            url=raw.get("externeUrl") or f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{raw.get('referenznummer')}",
            title=title, company=clean_line(raw.get("firma")) or "—", location=location, country="Germany",
            remote_status="remote" if remote else ("hybrid" if partly else None), description=desc,
            employment_type=", ".join(kinds) or None, career_page_url="https://www.arbeitsagentur.de/jobsuche/",
            date_posted=datetime.fromisoformat(posted) if posted else None,
            raw={
                "search": raw.get("_search"), "occupation": raw.get("hauptberuf"), "distance_km": raw.get("entfernung"),
                "drive_minutes": raw.get("_drive_min"),
                "minijob": minijob, "part_time": part_time, "fixed_term": fixed, "homeoffice": remote, "homeoffice_partly": partly, "start": start,
                "career_changers": bool(raw.get("quereinstiegGeeignet")),
                "pay_stated": (raw.get("verguetungsangabe") or "KEINE_ANGABEN") != "KEINE_ANGABEN",
                "flexible_slots": sum(bool(raw.get(k)) for k in ("arbeitszeitTeilzeitVormittag", "arbeitszeitTeilzeitNachmittag",
                                                                 "arbeitszeitTeilzeitAbend")) >= 2 or bool(raw.get("arbeitszeitTeilzeitFlexibel")),
                "temp_agency": bool(raw.get("istArbeitnehmerUeberlassung")),
                # a nationwide "remote" search hit without home office: neither near home nor remote → rejected
                "nationwide_not_remote": bool(raw.get("_require_remote") and not remote),
            },
        )
