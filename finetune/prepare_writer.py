"""Step 3 — writer training set (chat JSONL) from the distilled letters + the filtered public letters.

    docker compose exec -T -w /app -e PYTHONPATH=/app app python -m finetune.prepare_writer

Distilled letters (data/finetune/distill.jsonl, keep=true) are the core signal: your profile, real postings, the
writer's exact prompt, and the validated final letter. They are weighted x2.
Public letters (ShashiVish/cover-letter-dataset) add English variety, but only those that pass the same gates the
pipeline applies: none of the banned stock phrases, no technology or number the applicant's inputs don't contain,
80-450 words. They are converted into the writer's real prompt format (profile block + posting block + task).
Output: data/finetune/writer_train.jsonl, writer_valid.jsonl  ({"messages": [system, user, assistant]})
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

DATA = Path(os.environ.get("JOBSEARCHER_DATA", "/data"))
FT = DATA / "finetune"


def public_examples(limit: int = 200) -> list[dict]:
    sys.path.insert(0, "/app")
    from app.config_files import CandidateProfile, get_prefs
    from app.llm.letter_prompts import AVOID_PHRASES_EN, choose_variant, writer_system_en, writer_user
    from app.services.evidence import TECH_TERMS

    w = get_prefs().writing
    out = []
    for line in (FT / "raw" / "coverletters.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        letter = (r.get("Cover Letter") or "").strip()
        inputs = " ".join(str(r.get(k) or "") for k in ("Job Title", "Preferred Qualifications", "Hiring Company",
                                                         "Past Working Experience", "Current Working Experience",
                                                         "Skillsets", "Qualifications"))
        low = letter.lower()
        if any(p.lower() in low for p in AVOID_PHRASES_EN):
            continue
        if not 80 <= len(letter.split()) <= 450:
            continue
        if [t for t in TECH_TERMS if re.search(rf"(?<!\w){re.escape(t.lower())}(?!\w)", low) and t.lower() not in inputs.lower()]:
            continue  # names a technology the applicant's inputs don't have
        if [n for n in re.findall(r"\b\d+\b", letter) if n not in inputs]:
            continue  # invented number (years, percentages ...)
        name = (r.get("Applicant Name") or "").strip() or "Alex Example"
        profile = CandidateProfile(
            headline=r.get("Job Title") or "", summary="",
            experience=[{"employer": x, "role": "", "period": "", "highlights": []}
                        for x in (r.get("Current Working Experience"), r.get("Past Working Experience")) if x],
            skills={"direct_professional": [s.strip() for s in str(r.get("Skillsets") or "").split(",") if s.strip()]},
            education=[{"degree": r.get("Qualifications") or "", "institution": ""}],
        )
        job = SimpleNamespace(title=r.get("Job Title") or "", company=r.get("Hiring Company") or "", location="",
                              country="", employment_type=None, salary_min=None, salary_max=None, currency=None,
                              description=f"We are hiring a {r.get('Job Title')}.",
                              requirements=f"Preferred qualifications: {r.get('Preferred Qualifications') or ''}",
                              responsibilities="")
        body = re.sub(r"^\s*dear [^\n,]*,\s*", "", letter, flags=re.I)
        body = re.sub(r"\n\s*(sincerely|best regards|kind regards|regards)[\s\S]*$", "", body, flags=re.I).strip()
        target = f"Dear Hiring Team,\n\n{body}\n\nKind regards\n{name}"
        system = writer_system_en(w, False, False, False, choose_variant(name + job.title, False), False)
        user = writer_user(profile, job, w, "en")
        assistant = json.dumps({"angle": "", "selected_evidence": [], "subject": f"Application for the position of {job.title}",
                                "letter": target}, ensure_ascii=False)
        out.append({"messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                                 {"role": "assistant", "content": assistant}], "source": "public"})
        if len(out) >= limit:
            break
    return out


def distilled_examples() -> list[dict]:
    path = FT / "distill.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("keep") and r.get("system") and r.get("user"):
            out.append({"messages": [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]},
                                     {"role": "assistant", "content": r["assistant"]}],
                        "source": f"distilled-{r['language']}", "job_id": r["job_id"]})
    return out


def main() -> int:
    random.seed(7)
    distilled, public = distilled_examples(), public_examples()
    rows = distilled * 2 + public
    random.shuffle(rows)
    n_valid = max(4, len(rows) // 10)
    valid, train = rows[:n_valid], rows[n_valid:]
    for name, part in (("writer_train.jsonl", train), ("writer_valid.jsonl", valid)):
        with (FT / name).open("w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    print(f"distilled kept: {len(distilled)} (x2) · public passing the gates: {len(public)} · "
          f"train {len(train)} · valid {len(valid)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
