"""Append the companies found by discover_companies.py to config/companies.yaml (only new ones, only with live jobs
in your target countries). Existing entries and your edits are never changed.

    .venv/Scripts/python scripts/merge_discovered.py data/discovery/found.json config/companies.yaml
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import yaml

BOARD_URL = {
    "greenhouse": "https://job-boards.greenhouse.io/{id}",
    "lever": "https://jobs.lever.co/{id}",
    "ashby": "https://jobs.ashbyhq.com/{id}",
    "workable": "https://apply.workable.com/{id}/",
    "recruitee": "https://{id}.recruitee.com/",
    "smartrecruiters": "https://jobs.smartrecruiters.com/{id}",
    "personio": "https://{id}.jobs.personio.de",
}
CITY = {"UAE": "Dubai", "Saudi Arabia": "Riyadh", "Luxembourg": "Luxembourg"}


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\b(gmbh|ag|se|inc|ltd|group|company)\b", "", s))


def careers_url(r: dict) -> str:
    if r["ats_type"] == "workday":
        tenant_wd, _, site = r["identifier"].partition("/")
        tenant, _, wd = tenant_wd.partition(".")
        return f"https://{tenant}.{wd}.myworkdayjobs.com/{site}"
    url = BOARD_URL[r["ats_type"]].format(id=r["identifier"])
    if r["ats_type"] in ("greenhouse", "lever") and (r.get("options") or {}).get("region") == "eu":
        url = url.replace("job-boards.greenhouse.io", "job-boards.eu.greenhouse.io").replace("jobs.lever.co", "jobs.eu.lever.co")
    return url


def priority(r: dict) -> str:
    gulf_lux = sum(v for k, v in r["in_targets"].items() if k != "Germany")
    if gulf_lux >= 3 or re.search(r"\bAI\b|robot|autonom", r.get("sector") or "", re.I):
        return "A"
    return "B" if sum(r["in_targets"].values()) >= 2 else "C"


def main() -> None:
    found = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    cfg_path = Path(sys.argv[2])
    text = cfg_path.read_text(encoding="utf-8")
    existing = yaml.safe_load(text)["companies"]
    known = {norm(c["company_name"]) for c in existing} | {norm(a) for c in existing for a in c.get("aliases") or []}
    known_ids = {(c.get("ats_type"), c.get("ats_identifier")) for c in existing}
    blocks, added = [], []
    for r in sorted(found, key=lambda r: (r.get("region") or "", r["name"].lower())):
        if not r.get("found") or not r.get("in_targets"):
            continue
        if norm(r["name"]) in known or (r["ats_type"], r["identifier"]) in known_ids:
            continue
        region = r.get("region") or "Germany"
        if region not in r["in_targets"]:
            # a Gulf / Luxembourg company whose "board" only has German jobs is someone else's board (a generic slug
            # like "property" or "sirius"); a German company with jobs only abroad is still fine
            if region != "Germany":
                print(f"skipped {r['name']}: board {r['ats_type']}:{r['identifier']} has no jobs in {region}")
                continue
        entry = {"company_name": r["name"], "country": region, "city": CITY.get(region), "sector": r.get("sector"),
                 "priority": priority(r), "careers_url": careers_url(r), "ats_type": r["ats_type"],
                 "ats_identifier": r["identifier"]}
        opts = dict(r.get("options") or {})
        if opts:
            entry["ats_config"] = opts
        entry = {k: v for k, v in entry.items() if v not in (None, "", {})}
        jobs = ", ".join(f"{k} {v}" for k, v in sorted(r["in_targets"].items()))
        blocks.append(f"  # discovered {r['ats_type']} board · live jobs: {jobs}\n"
                      + yaml.safe_dump([entry], allow_unicode=True, sort_keys=False, width=200).replace("- ", "  - ", 1)
                      .replace("\n  ", "\n    ").replace("\n", "\n", 1))
        added.append(r["name"])
        known.add(norm(r["name"]))
        known_ids.add((r["ats_type"], r["identifier"]))
    if not blocks:
        print("nothing new")
        return
    header = ("\n  # ---- Discovered by scripts/discover_companies.py: ATS boards with live jobs in your target countries "
              "(Germany, UAE, Saudi Arabia, Luxembourg) ----\n")
    cfg_path.write_text(text.rstrip("\n") + "\n" + header + "\n".join(blocks), encoding="utf-8")
    yaml.safe_load(cfg_path.read_text(encoding="utf-8"))  # still valid
    print(f"added {len(added)}: {', '.join(added)}")


if __name__ == "__main__":
    main()
