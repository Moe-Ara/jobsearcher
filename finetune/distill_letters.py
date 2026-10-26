"""Step 2 — self-distillation: training letters from the full pipeline, on a scratch copy of the database.

    docker compose exec -T -w /app -e PYTHONPATH=/app app python -m finetune.distill_letters --n 40

For each training job the complete letter pipeline runs (writer → German editor → fact check → correction →
outline / salutation / mobility guarantees). Only letters that come out clean (no unsupported, ambiguous, placeholder,
salutation or tone warnings) are kept. Each record pairs the WRITER's exact prompt with the validated FINAL letter,
so a fine-tuned writer learns to produce in one pass what currently takes an editor and correction rounds.

A fixed, held-out set of jobs (data/finetune/eval_jobs.json) is never used for training — evaluate.py uses it.
Your real database is never written: everything happens in data/finetune/distill.db (a fresh backup copy).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

DATA = Path(os.environ.get("JOBSEARCHER_DATA", "/data"))
FT = DATA / "finetune"
SCRATCH = FT / "distill.db"
CLEAN_BLOCKERS = {"unsupported", "ambiguous", "placeholder", "salutation", "tone", "number"}


def scratch_copy() -> None:
    """Consistent copy of the live DB via the SQLite backup API (safe while the app runs)."""
    FT.mkdir(parents=True, exist_ok=True)
    if SCRATCH.exists():
        return
    src = sqlite3.connect(DATA / "jobsearcher.db")
    dst = sqlite3.connect(SCRATCH)
    src.backup(dst)
    dst.close()
    src.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="training jobs to try")
    ap.add_argument("--eval-n", type=int, default=12, help="held-out jobs (fixed on the first run)")
    ap.add_argument("--out", default=str(FT / "distill.jsonl"))
    args = ap.parse_args()

    scratch_copy()
    os.environ["DATABASE_URL"] = f"sqlite:///{SCRATCH}"  # before any app import: never the live DB
    os.environ["SCHEDULER_ENABLED"] = "false"
    sys.path.insert(0, "/app")
    from sqlalchemy import select

    from app.db import session_scope
    from app.enums import AnalysisStatus, FilterStatus
    from app.llm.factory import get_llm
    from app.models import ApplicationDocument, Job
    from app.services import letters, research

    with session_scope() as s:
        jobs = s.scalars(select(Job).where(Job.active.is_(True), Job.filter_status != FilterStatus.REJECTED,
                                          Job.analysis_status == AnalysisStatus.DONE, Job.description.is_not(None))
                         .order_by(Job.fit_score.desc())).all()
        pool = [(j.id, letters.detect_language(j)[0], j.company or "") for j in jobs if len(j.description or "") > 600]

    eval_path = FT / "eval_jobs.json"
    if eval_path.exists():
        eval_ids = set(json.loads(eval_path.read_text())["job_ids"])
    else:  # every 5th job from the top 150: spread over fit scores, ~2/3 German like the real mix
        candidates = pool[:150:5]
        de = [p for p in candidates if p[1] == "de"][: args.eval_n * 2 // 3]
        en = [p for p in candidates if p[1] == "en"][: args.eval_n - len(de)]
        eval_ids = {p[0] for p in de + en}
        eval_path.write_text(json.dumps({"job_ids": sorted(eval_ids)}, indent=1))
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(line)["job_id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [p for p in pool if p[0] not in eval_ids and p[0] not in done][: max(0, args.n - len(done))]
    print(f"eval jobs: {len(eval_ids)} · already distilled: {len(done)} · to run: {len(todo)}", flush=True)

    captured: dict = {}
    original = letters.llm_call

    def capture(session, target, **kw):
        res = original(session, target, **kw)
        if kw.get("kind") == "letter_draft":
            captured.update(system=kw["system"], user=kw["user"], data=res.data)
        return res

    letters.llm_call = capture
    # Background work: no GPU priority — your own letter / application runs always go first (requests wait for them).
    import contextlib

    letters.interactive = contextlib.nullcontext
    setup = get_llm()
    kept = 0
    for job_id, lang, company in todo:
        captured.clear()
        started = time.monotonic()
        hit = research.cached(company) if company else None
        try:
            group = letters.generate_letter(job_id, language=lang, setup=setup, research=research.research_block(hit),
                                            personalization_review=False)
        except Exception as exc:  # noqa: BLE001 - keep going
            print(f"job {job_id}: failed ({exc})", flush=True)
            continue
        with session_scope() as s:
            final = s.scalars(select(ApplicationDocument).where(ApplicationDocument.group_id == group,
                                                                ApplicationDocument.stage == "final")).first()
            report = final.validation or {}
            blockers = sorted({w.get("type") for w in report.get("warnings", [])} & CLEAN_BLOCKERS)
            record = {"job_id": job_id, "language": lang, "seconds": round(time.monotonic() - started),
                      "status": report.get("status"), "blockers": blockers, "keep": not blockers,
                      "system": captured.get("system"), "user": captured.get("user"),
                      "assistant": json.dumps({"angle": (captured.get("data") or {}).get("angle", ""),
                                               "selected_evidence": (captured.get("data") or {}).get("selected_evidence", []),
                                               "subject": final.subject or "", "letter": final.content}, ensure_ascii=False)}
        kept += record["keep"]
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"job {job_id} [{lang}] {record['seconds']}s keep={record['keep']} {blockers}", flush=True)
    print(f"done: kept {kept} of {len(todo)} new letters", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
