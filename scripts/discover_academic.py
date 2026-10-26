"""Find the job pages of universities and research institutes and turn them into academic sources (PhD positions).

    .venv/Scripts/python scripts/discover_academic.py scripts/academic_candidates.yaml data/discovery/academic.json

For each institution it
  1. searches its own domain for the vacancies page (Bing/DuckDuckGo) and also tries common paths (/stellenangebote,
     /karriere/stellenangebote, /jobs, /career, ...),
  2. scores each candidate page by the links whose text reads like a doctoral / research position (the academic
     adapter's own anchor rule: "PhD", "Doktorand", "Wissenschaftliche*r Mitarbeiter*in", "Research Associate", ...),
  3. derives the job-link pattern from those links (the common URL prefix on the institution's host),
  4. verifies the result with the real academic adapter (it must find at least one posting link).
Pages rendered by JavaScript only have no links to find — those institutions stay covered by academics.de / EURAXESS.
Read-only: GET requests and searches only.
"""

from __future__ import annotations

import base64
import html as htmllib
import json
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import parse_qs, unquote, urljoin, urlsplit

import httpx
import yaml
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.sources.academic import DEFAULT_ANCHOR_PATTERN  # noqa: E402

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
client = httpx.Client(timeout=25, follow_redirects=True, headers={"User-Agent": UA, "Accept-Language": "de,en;q=0.8"})
ANCHOR = re.compile(DEFAULT_ANCHOR_PATTERN, re.IGNORECASE)
# a vacancy title, not a menu item ("Promotion", "PhD programme"): gender marker, "für/in/im" + topic, or a posting word
POSTING = re.compile(r"\((?:m|w|f|d|x|gn)\s*[/|,]|\b[mwfd]\s*/\s*[mwfd]\s*/\s*[mwdfx]\b|all genders|\*innen|\*in\b|"
                     r"promotionsstelle|doktorand\w*\s+(f\u00fcr|im|in|zum)|ph\.?\s?d\.?\s+(position|student|candidate|researcher|"
                     r"fellowship|scholarship)|doctoral\s+(candidate|researcher|position|student)|research\s+(associate|assistant)"
                     r"\s+(in|for|on)|mitarbeiter\w*\s*(\(|f\u00fcr|im|in|zur|zum)", re.IGNORECASE)
PORTALS = re.compile(r"interamt\.de|softgarden|dvinci|umantis|rexx-systems|onlyfy|personio|jobs\.b-ite|"
                     r"stellenportal|jobboerse|successfactors|recruitee|smartrecruiters|greenhouse|workday", re.I)
JOBISH = re.compile(r"stellen|karriere|career|jobs?\b|vacanc|ausschreibung|offene|opportunit|positions|recruit", re.I)
COMMON_PATHS = ["/stellenangebote", "/stellenangebote/", "/karriere/stellenangebote", "/jobs", "/jobs/", "/karriere",
                "/career", "/careers", "/de/karriere/stellenangebote", "/universitaet/stellenangebote", "/stellenmarkt",
                "/en/jobs", "/en/careers", "/en/about/careers", "/jobs-and-careers", "/vacancies"]
_search_lock = threading.Lock()
_last = [0.0]


def get(url: str):
    try:
        r = client.get(url)
    except httpx.HTTPError:
        return None
    if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
        return None
    if re.search(r"/(error|404)\b|[?;]404", str(r.url), re.I):  # error page served with status 200
        return None
    return r


def search(query: str) -> list[str]:
    with _search_lock:
        wait = _last[0] + 2.5 - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()
    urls = []
    r = get(f"https://www.bing.com/search?q={httpx.QueryParams({'q': query})['q']}&setlang=de")
    for href in re.findall(r'<li class="b_algo".*?<h2[^>]*><a[^>]+href="([^"]+)"', r.text if r else "", re.S):
        u = (parse_qs(urlsplit(htmllib.unescape(href)).query).get("u") or [""])[0]
        if u.startswith("a1"):
            try:
                urls.append(base64.urlsafe_b64decode(u[2:] + "=" * (-len(u[2:]) % 4)).decode("utf-8", "replace"))
            except ValueError:
                pass
    return urls[:8]


def on_domain(url: str, domain: str) -> bool:
    host = urlsplit(url).netloc.lower()
    return host == domain or host.endswith("." + domain)


def doctoral_links(page_url: str, html: str, domain: str) -> list[str]:
    out = []
    for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
        text = " ".join(a.get_text(" ").split()) or a.get("title", "")
        if len(text) < 25 or not ANCHOR.search(text) or not POSTING.search(text):
            continue
        href = urljoin(page_url, a["href"]).split("#")[0]
        if not href.startswith("http") or href.rstrip("/") == page_url.rstrip("/"):
            continue
        if on_domain(href, domain) or PORTALS.search(urlsplit(href).netloc):
            out.append(href)
    return list(dict.fromkeys(out))


def link_pattern(links: list[str]) -> str:
    """Common prefix of the posting links (host + path directory) as a regex; at least the host."""
    parts = [urlsplit(u) for u in links]
    host = Counter(p.netloc for p in parts).most_common(1)[0][0]
    paths = [p.path for p in parts if p.netloc == host]
    prefix = paths[0]
    for p in paths[1:]:
        while not p.startswith(prefix):
            prefix = prefix[:-1]
    prefix = prefix[: prefix.rfind("/") + 1] if "/" in prefix else "/"
    return re.escape(host + prefix).replace("\\/", "/") + ".+"


def candidate_pages(c: dict) -> list[str]:
    domain = c["domain"]
    urls = [c["careers_url"]] if c.get("careers_url") else []
    for q in (f"site:{domain} Stellenangebote wissenschaftliche Mitarbeiter", f"site:{domain} PhD position vacancies"):
        urls += [u for u in search(q) if on_domain(u, domain)]
    hosts = {f"https://www.{domain}", f"https://{domain}"}
    urls += [h + p for h in hosts for p in COMMON_PATHS]
    seen, out = set(), []
    for u in urls:
        key = u.rstrip("/").lower()
        if key not in seen:
            seen.add(key)
            out.append(u)
    # pages that look like job pages first
    return sorted(out, key=lambda u: 0 if JOBISH.search(u) else 1)[:24]


def discover(c: dict) -> dict:
    best = None
    for url in candidate_pages(c):
        r = get(url)
        if not r:
            continue
        links = doctoral_links(str(r.url), r.text, c["domain"])
        if len(links) >= 3 and (best is None or len(links) > len(best["links"])):
            best = {"url": str(r.url), "links": links}
            if len(links) >= 8:
                break
    if not best:
        return {**c, "found": False}
    pattern = link_pattern(best["links"])
    rx = re.compile(pattern)
    matched = [u for u in best["links"] if rx.search(u)]
    return {**c, "found": True, "careers_url": best["url"], "job_link_pattern": pattern, "links": len(matched),
            "sample": matched[:3]}


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    cands = yaml.safe_load(src.read_text(encoding="utf-8"))["candidates"]
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(discover, c): c for c in cands}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as exc:  # noqa: BLE001
                r = {**futs[f], "found": False, "error": str(exc)[:200]}
            results.append(r)
            print(f"[{i}/{len(cands)}] {r['name']}: " + (f"{r['links']} links · {r['careers_url']}" if r.get("found") else "-"), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sorted(results, key=lambda r: r["name"]), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{sum(1 for r in results if r.get('found'))} of {len(cands)} institutions have a readable job page -> {out}")


if __name__ == "__main__":
    main()
