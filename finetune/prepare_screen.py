"""Step 4 — screener training set (Qwen3-8B, the model that screens every new job).

    docker compose exec -T -w /app -e PYTHONPATH=/app app python -m finetune.prepare_screen

Two sources, both in the screener's exact prompt format:
  - your own relevance labels (GREAT / MAYBE / BAD from the Calibrate page), fold A only (even job ids);
    fold B (odd ids) is never trained on — evaluate.py measures the screener on it
  - cnamuangtoun/resume-job-description-fit (Good / Potential / No Fit): 500 balanced pairs, the résumé in place of
    your profile
The base screener answers each prompt (teacher); its relevance verdict is then corrected to the human label
(score and likely_relevant), so every other field keeps the format the app expects. Runs on the scratch database.
Output: data/finetune/screen_train.jsonl, screen_valid.jsonl, screen_eval.json
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

DATA = Path(os.environ.get("JOBSEARCHER_DATA", "/data"))
FT = DATA / "finetune"
SCRATCH = FT / "distill.db"

LABEL_RULES = {  # label → (min score, max score, likely_relevant, reason)
    "GREAT": (8, 10, True, "Strong overlap between the candidate's experience and the role's core requirements."),
    "Good Fit": (8, 10, True, "Strong overlap between the candidate's experience and the role's core requirements."),
    "MAYBE": (4, 6, True, "Partial overlap: some core requirements match, others would need to be learned."),
    "Potential Fit": (4, 6, True, "Partial overlap: some core requirements match, others would need to be learned."),
    "BAD": (0, 3, False, "The role's core requirements are outside the candidate's experience and goals."),
    "No Fit": (0, 2, False, "The role's core requirements are outside the candidate's experience."),
}


def correct(teacher: dict, label: str) -> dict:
    lo, hi, likely, reason = LABEL_RULES[label]
    out = dict(teacher)
    out["relevance_score"] = max(lo, min(hi, int(teacher.get("relevance_score") or lo)))
    out["likely_relevant"] = likely
    if bool(teacher.get("likely_relevant")) != likely:
        out["reason"] = reason  # the teacher's reasoning pointed the other way
    return out


def main() -> int:
    os.environ["DATABASE_URL"] = f"sqlite:///{SCRATCH}"
    os.environ["SCHEDULER_ENABLED"] = "false"
    sys.path.insert(0, "/app")
    import contextlib

    from sqlalchemy import select

    from app.config_files import CandidateProfile, get_prefs, get_profile
    from app.db import session_scope
    from app.enums import RelevanceLabel
    from app.llm.factory import Role, get_llm
    from app.llm.prompts import SCREEN_PROMPT_VERSION, screen_system_prompt, screen_user_prompt
    from app.llm.schemas import JobScreening
    from app.models import Job
    from app.services.llm_calls import llm_call

    random.seed(11)
    prefs, profile = get_prefs(), get_profile()
    target = get_llm().target(Role.SCREEN)
    max_chars = prefs.llm.max_description_chars_triage
    items: list[tuple[str, str, str]] = []  # (system, user, label)

    with session_scope() as s:
        labelled = s.scalars(select(Job).where(Job.relevance_label != RelevanceLabel.UNRATED)).all()
        fold_a = [j for j in labelled if j.id % 2 == 0]
        fold_b = [j.id for j in labelled if j.id % 2 == 1]
        system_own = screen_system_prompt(profile, prefs)
        for j in fold_a:
            items.append((system_own, screen_user_prompt(j, max_chars), str(j.relevance_label.value if hasattr(j.relevance_label, "value") else j.relevance_label)))
    (FT / "screen_eval.json").write_text(json.dumps({"job_ids": sorted(fold_b)}, indent=1))

    pairs = [json.loads(line) for line in (FT / "raw" / "jobfit.jsonl").read_text(encoding="utf-8").splitlines()]
    by_label: dict[str, list] = {}
    for p in pairs:
        by_label.setdefault(p["label"], []).append(p)
    sample = (random.sample(by_label["Good Fit"], 150) + random.sample(by_label["Potential Fit"], 150)
              + random.sample(by_label["No Fit"], 200))
    for p in sample:
        cand = CandidateProfile(summary=(p["resume_text"] or "")[:3000])
        job = SimpleNamespace(title="", company="", location="", country="", employment_type=None, salary_min=None,
                              salary_max=None, currency=None, description=(p["job_description_text"] or "")[:4000],
                              requirements="", responsibilities="")
        items.append((screen_system_prompt(cand, prefs), screen_user_prompt(job, max_chars), p["label"]))

    out_rows = []
    print(f"own labels fold A: {len(fold_a)} · eval fold B: {len(fold_b)} · public pairs: {len(sample)}", flush=True)
    for i, (system, user, label) in enumerate(items, 1):
        with session_scope() as s, contextlib.suppress(Exception):
            res = llm_call(s, target, kind="screen", job_id=None, prompt_version=SCREEN_PROMPT_VERSION + "-ft-teacher",
                           profile_hash="-", system=system, user=user, schema=JobScreening)
            target_json = correct(res.data, label)
            out_rows.append({"messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                                          {"role": "assistant", "content": json.dumps(target_json, ensure_ascii=False)}]})
        if i % 50 == 0:
            print(f"  teacher {i}/{len(items)}", flush=True)
    random.shuffle(out_rows)
    n_valid = max(10, len(out_rows) // 20)
    for name, part in (("screen_train.jsonl", out_rows[n_valid:]), ("screen_valid.jsonl", out_rows[:n_valid])):
        with (FT / name).open("w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"screener examples: train {len(out_rows) - n_valid} · valid {n_valid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
