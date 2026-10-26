"""Find the job feeds (ATS boards) of many companies at once and keep those with live jobs in your target countries.

    .venv/Scripts/python scripts/discover_companies.py scripts/company_candidates.yaml data/discovery/found.json

For each candidate it
  1. probes the public job APIs (Greenhouse US/EU, Lever US/EU, Ashby, Workable, Recruitee, SmartRecruiters,
     Personio) with likely board names (or the `slug` hints in the candidate file),
  2. if nothing is found, searches the web for the company's careers page and reads which ATS it links to
     (also Workday: <tenant>.wd<N>.myworkdayjobs.com/<site>),
  3. counts the board's jobs per country and keeps it only if some are in your target countries.
A board must also plausibly belong to the company (its name / URLs match), so a generic slug ("noon") cannot
import someone else's jobs. Read-only: it only sends GET/search requests to public job APIs.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.normalize import infer_country  # noqa: E402

TARGETS = {"Germany", "UAE", "Saudi Arabia", "Luxembourg"}
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
client = httpx.Client(timeout=25, follow_redirects=True, headers={"User-Agent": UA, "Accept-Language": "en"})
_search_lock = threading.Lock()
_last_search = [0.0]


def slugs(name: str, hints: list[str]) -> list[str]:
    base = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    base = re.sub(r"\b(gmbh|ag|se|inc|ltd|llc|group|holding|technologies|technology|labs?|co|kg|sa|pjsc|psc|s\.a\.)\b", " ", base)
    words = re.findall(r"[a-z0-9]+", base)
    out = list(hints)
    if words:
        out += ["".join(words), "-".join(words), words[0]] if len(words) > 1 else [words[0]]
    return list(dict.fromkeys(s for s in out if s and len(s) >= 2))


def get(url: str, **kw):
    try:
        r = client.get(url, **kw)
        return r if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


def countries(locations: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for loc in locations:
        found = {infer_country(p.strip()) for p in re.split(r"[;|]|\s/\s", loc or "") if p.strip()} - {None}
        for c in found or {"?"}:
            out[c] = out.get(c, 0) + 1
    return out


# ------------------------------------------------------------------ one probe per ATS: (jobs, locations, owner text)

def probe_greenhouse(slug: str, eu: bool = False):
    host = "boards-api.eu.greenhouse.io" if eu else "boards-api.greenhouse.io"
    r = get(f"https://{host}/v1/boards/{slug}/jobs")
    if not r:
        return None
    jobs = r.json().get("jobs") or []
    meta = get(f"https://{host}/v1/boards/{slug}")
    owner = (meta.json().get("name") if meta else "") or ""
    return {"ats_type": "greenhouse", "identifier": slug, "options": {"region": "eu"} if eu else {},
            "jobs": len(jobs), "locations": [(j.get("location") or {}).get("name", "") for j in jobs],
            "titles": [j.get("title") for j in jobs], "owner": owner + " " + " ".join(j.get("absolute_url", "") for j in jobs[:3])}


def probe_lever(slug: str, eu: bool = False):
    host = "api.eu.lever.co" if eu else "api.lever.co"
    r = get(f"https://{host}/v0/postings/{slug}", params={"mode": "json"})
    if not r or not isinstance(r.json(), list):
        return None
    jobs = r.json()
    return {"ats_type": "lever", "identifier": slug, "options": {"region": "eu"} if eu else {}, "jobs": len(jobs),
            "locations": [((j.get("categories") or {}).get("location") or "") + "; " +
                          "; ".join((j.get("categories") or {}).get("allLocations") or []) for j in jobs],
            "titles": [j.get("text") for j in jobs], "owner": " ".join(j.get("hostedUrl", "") for j in jobs[:3])}


def probe_ashby(slug: str):
    r = get(f"https://api.ashbyhq.com/posting-api/job-board/{slug}")
    if not r:
        return None
    jobs = r.json().get("jobs") or []
    locs = []
    for j in jobs:
        addr = ((j.get("address") or {}).get("postalAddress") or {})
        locs.append("; ".join(x for x in [j.get("location"), addr.get("addressCountry"),
                                           *[s.get("location", "") for s in j.get("secondaryLocations") or []]] if x))
    return {"ats_type": "ashby", "identifier": slug, "options": {}, "jobs": len(jobs), "locations": locs,
            "titles": [j.get("title") for j in jobs], "owner": slug}


def probe_workable(slug: str):
    r = get(f"https://apply.workable.com/api/v1/widget/accounts/{slug}")
    if not r:
        return None
    d = r.json()
    jobs = d.get("jobs") or []
    locs = ["; ".join(", ".join(x for x in (l.get("city"), l.get("country")) if x) for l in (j.get("locations") or [{"city": j.get("city"), "country": j.get("country")}])) for j in jobs]
    return {"ats_type": "workable", "identifier": slug, "options": {}, "jobs": len(jobs), "locations": locs,
            "titles": [j.get("title") for j in jobs], "owner": d.get("name") or ""}


def probe_recruitee(slug: str):
    r = get(f"https://{slug}.recruitee.com/api/offers/")
    if not r:
        return None
    jobs = r.json().get("offers") or []
    locs = ["; ".join(", ".join(x for x in (l.get("city"), l.get("country")) if x) for l in (j.get("locations") or [])) or
            ", ".join(x for x in (j.get("city"), j.get("country")) if x) for j in jobs]
    return {"ats_type": "recruitee", "identifier": slug, "options": {}, "jobs": len(jobs), "locations": locs,
            "titles": [j.get("title") for j in jobs], "owner": " ".join(j.get("company_name") or "" for j in jobs[:2]) + " " + slug}


def probe_smartrecruiters(slug: str):
    r = get(f"https://api.smartrecruiters.com/v1/companies/{slug}/postings", params={"limit": 100})
    if not r:
        return None
    d = r.json()
    jobs = d.get("content") or []
    if not jobs:
        return None
    locs = [", ".join(x for x in ((j.get("location") or {}).get("city"), (j.get("location") or {}).get("country")) if x) for j in jobs]
    return {"ats_type": "smartrecruiters", "identifier": slug, "options": {}, "jobs": int(d.get("totalFound") or len(jobs)),
            "locations": locs, "titles": [j.get("name") for j in jobs], "owner": " ".join((j.get("company") or {}).get("name", "") for j in jobs[:2])}


def probe_personio(slug: str):
    for dom in ("jobs.personio.de", "jobs.personio.com"):
        r = get(f"https://{slug}.{dom}/xml")
        if r and "<position>" in r.text:
            titles = re.findall(r"<name>(.*?)</name>", r.text)
            offices = re.findall(r"<office>(.*?)</office>", r.text)
            return {"ats_type": "personio", "identifier": slug, "options": {"domain": dom}, "jobs": len(titles),
                    "locations": offices, "titles": titles, "owner": slug}
    return None


def probe_workday(identifier: str, searches: list[str]):
    host_part, _, site = identifier.partition("/")
    tenant, _, wd = host_part.partition(".")
    api = f"https://{tenant}.{wd}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs"
    titles, locs, total = [], [], 0
    for text in searches:
        try:
            r = client.post(api, json={"appliedFacets": {}, "limit": 20, "offset": 0, "searchText": text})
        except httpx.HTTPError:
            return None
        if r.status_code != 200:
            return None
        d = r.json()
        total += int(d.get("total") or 0)
        for p in d.get("jobPostings") or []:
            titles.append(p.get("title"))
            parts = (p.get("externalPath") or "").split("/")
            locs.append(parts[2].replace("-", " ") if len(parts) > 2 else text)
    return {"ats_type": "workday", "identifier": identifier, "options": {}, "jobs": total, "locations": locs,
            "titles": titles, "owner": tenant}


PROBES = [("greenhouse", lambda s: probe_greenhouse(s)), ("greenhouse-eu", lambda s: probe_greenhouse(s, True)),
          ("lever", lambda s: probe_lever(s)), ("lever-eu", lambda s: probe_lever(s, True)), ("ashby", probe_ashby),
          ("workable", probe_workable), ("recruitee", probe_recruitee), ("smartrecruiters", probe_smartrecruiters),
          ("personio", probe_personio)]


# ------------------------------------------------------------------ web search fallback

ATS_LINK = re.compile(
    r"https?://(?:[a-z0-9-]+\.)*(?:greenhouse\.io|lever\.co|ashbyhq\.com|workable\.com|recruitee\.com|smartrecruiters\.com|"
    r"jobs\.personio\.(?:de|com)|myworkdayjobs\.com)[^\s\"'<>)]*", re.I)


def search(query: str) -> list[str]:
    with _search_lock:  # one search at a time, politely spaced
        wait = _last_search[0] + 2.5 - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_search[0] = time.monotonic()
    urls = []
    r = get("https://html.duckduckgo.com/html/", params={"q": query})
    if r:
        for href in re.findall(r'class="result__a"[^>]*href="([^"]+)"', r.text):
            qs = parse_qs(urlsplit(href).query)
            urls.append(unquote(qs["uddg"][0]) if "uddg" in qs else href)
    if not urls:  # DuckDuckGo challenge page (HTTP 202) → Bing
        import base64
        import html as htmllib
        r = get("https://www.bing.com/search", params={"q": query, "setlang": "en"})
        for href in re.findall(r'<li class="b_algo".*?<h2[^>]*><a[^>]+href="([^"]+)"', r.text if r else "", re.S):
            u = (parse_qs(urlsplit(htmllib.unescape(href)).query).get("u") or [""])[0]
            if u.startswith("a1"):
                try:
                    urls.append(base64.urlsafe_b64decode(u[2:] + "=" * (-len(u[2:]) % 4)).decode("utf-8", "replace"))
                except ValueError:
                    pass
    return urls[:6]


def ats_from_url(url: str) -> tuple[str, str, dict] | None:
    parts = urlsplit(url)
    host, path = parts.netloc.lower(), [p for p in parts.path.split("/") if p]
    if "greenhouse.io" in host:
        tok = (parse_qs(parts.query).get("for") or [None])[0] or (path[0] if path and path[0] not in ("embed", "v1") else None)
        return ("greenhouse", tok, {"region": "eu"} if ".eu." in host else {}) if tok else None
    if "lever.co" in host and path:
        return "lever", path[0], {"region": "eu"} if ".eu." in host else {}
    if host == "jobs.ashbyhq.com" and path:
        return "ashby", path[0], {}
    if host == "apply.workable.com" and path:
        return "workable", path[0], {}
    m = re.match(r"^([a-z0-9-]+)\.recruitee\.com$", host)
    if m:
        return "recruitee", m.group(1), {}
    if "smartrecruiters.com" in host and path and host.startswith(("jobs.", "careers.")):
        return "smartrecruiters", path[0], {}
    m = re.match(r"^([a-z0-9-]+)\.jobs\.personio\.(de|com)$", host)
    if m:
        return "personio", m.group(1), {}
    m = re.match(r"^([a-z0-9-]+)\.(wd\d+)\.myworkdayjobs\.com$", host)
    if m and path:
        site = path[1] if len(path) > 1 and re.fullmatch(r"[a-z]{2}-[A-Z]{2}", path[0]) else path[0]
        return "workday", f"{m.group(1)}.{m.group(2)}/{site}", {}
    return None


def web_detect(name: str) -> list[tuple[str, str, dict]]:
    found: dict[tuple[str, str], dict] = {}
    sites = ("site:boards.greenhouse.io OR site:job-boards.greenhouse.io OR site:jobs.lever.co OR site:jobs.ashbyhq.com "
             "OR site:apply.workable.com OR site:myworkdayjobs.com OR site:recruitee.com OR site:jobs.smartrecruiters.com "
             "OR site:jobs.personio.de")
    for q in (f'"{name}" careers', f'"{name}" ({sites})'):
        urls = search(q)
        for u in urls:
            hit = ats_from_url(u)
            if hit:
                found.setdefault(hit[:2], hit[2])
                continue
            if any(x in u for x in ("linkedin.", "indeed.", "glassdoor.", "bayt.", "naukrigulf", "wikipedia")):
                continue
            page = get(u)
            if page:
                for link in ATS_LINK.findall(page.text[:400000]):
                    hit = ats_from_url(link)
                    if hit:
                        found.setdefault(hit[:2], hit[2])
    return [(k[0], k[1], v) for k, v in found.items()]


# ------------------------------------------------------------------ main

def plausible(name: str, result: dict, slug_from_hint: bool) -> bool:
    """The board belongs to this company: its name / URLs contain the company's main word (hints are trusted)."""
    if slug_from_hint:
        return True
    words = [w for w in re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower())
             if len(w) >= 3 and w not in ("the", "and", "group", "gmbh", "technologies", "bank")]
    hay = (result.get("owner") or "").lower() + " " + result["identifier"].lower()
    return any(w in hay for w in words[:2])


def discover(c: dict) -> dict:
    name, hints = c["name"], list(c.get("slug") or [])
    if isinstance(c.get("slug"), str):
        hints = [c["slug"]]
    best = None
    if c.get("workday"):
        res = probe_workday(c["workday"], sorted(TARGETS))
        if res:
            best = res
    for s in ([] if best else slugs(name, hints)):
        for _label, fn in PROBES:
            if c.get("ats") and not _label.startswith(c["ats"]):
                continue
            res = fn(s)
            if res and res["jobs"] > 0 and plausible(name, res, s in hints):
                if best is None or res["jobs"] > best["jobs"]:
                    best = res
        if best:
            break
    via = "probe"
    if best is None and not c.get("no_search"):
        for ats_type, ident, opts in web_detect(name):
            fn = {"greenhouse": lambda i: probe_greenhouse(i, opts.get("region") == "eu"),
                  "lever": lambda i: probe_lever(i, opts.get("region") == "eu"), "ashby": probe_ashby,
                  "workable": probe_workable, "recruitee": probe_recruitee, "smartrecruiters": probe_smartrecruiters,
                  "personio": probe_personio, "workday": lambda i: probe_workday(i, sorted(TARGETS))}[ats_type]
            res = fn(ident)
            if res and res["jobs"] > 0 and (ats_type == "workday" or plausible(name, res, False)):
                best, via = res, "web"
                break
    if best is None:
        return {**c, "found": False}
    by_country = countries(best["locations"])
    in_targets = {k: v for k, v in by_country.items() if k in TARGETS}
    sample = [t for t, loc in zip(best["titles"], best["locations"]) if countries([loc]).keys() & TARGETS][:6]
    return {**c, "found": True, "via": via, "ats_type": best["ats_type"], "identifier": best["identifier"],
            "options": best["options"], "jobs": best["jobs"], "by_country": by_country, "in_targets": in_targets,
            "sample": sample}


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    cands = yaml.safe_load(src.read_text(encoding="utf-8"))["candidates"]
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
    if only:
        cands = [c for c in cands if c["name"] in only]
    results = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(discover, c): c for c in cands}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as exc:  # noqa: BLE001
                r = {**futs[f], "found": False, "error": str(exc)[:200]}
            results.append(r)
            mark = f"{r.get('ats_type')}:{r.get('identifier')} {r.get('jobs')} jobs, targets {r.get('in_targets')}" if r.get("found") else "-"
            print(f"[{i}/{len(cands)}] {r['name']}: {mark}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sorted(results, key=lambda r: r["name"].lower()), indent=1, ensure_ascii=False), encoding="utf-8")
    hit = [r for r in results if r.get("found") and r.get("in_targets")]
    print(f"\n{len(hit)} of {len(cands)} companies have live jobs in your target countries -> {out}")


if __name__ == "__main__":
    main()
