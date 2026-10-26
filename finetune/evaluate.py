"""Step 7 — A/B evaluation: base vs fine-tuned, on held-out data, with the BASE model as the only judge.

    docker compose exec -T -w /app -e PYTHONPATH=/app app python -m finetune.evaluate --task writer
    docker compose exec -T -w /app -e PYTHONPATH=/app app python -m finetune.evaluate --task screen

writer  For every held-out job (eval_jobs.json, never trained on) both writers draft the letter in the real
        pipeline (no editor, so the writer itself is compared); the base validator fact-checks both. Per letter:
        unsupported / ambiguous claims in the DRAFT, tone problems, outline parts the system had to add, stock
        phrases, length outside the target, writer seconds. Lower penalty is better.
screen  Your own labels, fold B (never trained on): AUC of relevance_score for GREAT vs BAD, accuracy of
        likely_relevant, mean score per label.
Decision rule (written to data/finetune/report_<task>.json): the fine-tuned model is KEPT only if it is better on
the main metric and not worse on the safety metric (unsupported claims / BAD-job acceptance). Otherwise the base stays.
Runs on the scratch database only.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

DATA = Path(os.environ.get("JOBSEARCHER_DATA", "/data"))
FT = DATA / "finetune"
SCRATCH = FT / "distill.db"
FT_MODELS = {"writer": "jobsearcher-writer-ft:latest", "screen": "jobsearcher-screen-ft:latest"}


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def eval_screen(models: dict[str, str]) -> dict:
    from sqlalchemy import select

    from app.config_files import get_prefs, get_profile
    from app.db import session_scope
    from app.llm.factory import Role, get_llm
    from app.llm.prompts import screen_system_prompt, screen_user_prompt
    from app.llm.schemas import JobScreening
    from app.models import Job
    from app.services.llm_calls import llm_call

    ids = json.loads((FT / "screen_eval.json").read_text())["job_ids"]
    base_target = get_llm().target(Role.SCREEN)
    prefs, profile = get_prefs(), get_profile()
    system = screen_system_prompt(profile, prefs)
    results = {}
    for name, model in models.items():
        target = replace(base_target, model=model, fallback=None)
        scores = {"GREAT": [], "MAYBE": [], "BAD": []}
        correct, n, seconds = 0, 0, 0.0
        with session_scope() as s:
            for job in s.scalars(select(Job).where(Job.id.in_(ids))):
                label = str(getattr(job.relevance_label, "value", job.relevance_label))
                t = time.monotonic()
                try:
                    res = llm_call(s, target, kind="screen", job_id=job.id, prompt_version=f"eval-{name}",
                                   profile_hash="-", system=system, user=screen_user_prompt(job, prefs.llm.max_description_chars_triage),
                                   schema=JobScreening, use_cache=False)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {name} job {job.id}: {exc}", flush=True)
                    continue
                seconds += time.monotonic() - t
                score = float(res.data.get("relevance_score") or 0)
                scores.setdefault(label, []).append(score)
                if label in ("GREAT", "BAD"):
                    n += 1
                    correct += bool(res.data.get("likely_relevant")) == (label == "GREAT")
        results[name] = {
            "auc_great_vs_bad": round(auc(scores["GREAT"], scores["BAD"]), 3),
            "accuracy_likely_relevant": round(correct / n, 3) if n else None,
            "bad_accepted": sum(1 for x in scores["BAD"] if x >= get_prefs().llm.min_screen_relevance),
            "mean_score": {k: round(statistics.mean(v), 2) for k, v in scores.items() if v},
            "n": {k: len(v) for k, v in scores.items()}, "seconds_per_job": round(seconds / max(1, sum(map(len, scores.values()))), 2),
        }
        print(name, json.dumps(results[name]), flush=True)
    base, ft = results["base"], results["finetuned"]
    keep = (ft["auc_great_vs_bad"] >= base["auc_great_vs_bad"] + 0.02
            and ft["bad_accepted"] <= base["bad_accepted"]
            and (ft["accuracy_likely_relevant"] or 0) >= (base["accuracy_likely_relevant"] or 0))
    return {"results": results, "keep_finetuned": keep,
            "rule": "AUC +0.02 or more, no more BAD jobs passed to deep analysis, accuracy not lower"}


def letter_metrics(report: dict, text: str, lang: str) -> dict:
    from app.llm.letter_prompts import AVOID_PHRASES_DE, AVOID_PHRASES_EN
    from app.services.letters import word_count

    initial = (report.get("rounds") or [{}])[0].get("claims", [])
    unsupported = sum(1 for c in initial if c.get("status") == "UNSUPPORTED")
    ambiguous = sum(1 for c in initial if c.get("status") == "AMBIGUOUS")
    warnings = report.get("warnings", [])
    tone = sum(1 for w in warnings if w.get("type") == "tone")
    outline = [a for a in report.get("outline_added", []) if a != "closing"]
    salutation = sum(1 for w in warnings if w.get("type") == "salutation")
    low = text.lower()
    stock = sum(1 for p in (AVOID_PHRASES_DE + AVOID_PHRASES_EN) if p.lower() in low)
    words = word_count(text)
    length_off = int(not 170 <= words <= 420)
    penalty = 3 * unsupported + 1.5 * ambiguous + tone + len(outline) + 2 * salutation + 0.5 * stock + length_off
    return {"unsupported": unsupported, "ambiguous": ambiguous, "tone": tone, "outline_added": len(outline),
            "invented_salutation": salutation, "stock_phrases": stock, "words": words, "penalty": penalty}


def eval_writer(models: dict[str, str]) -> dict:
    from sqlalchemy import select

    from app.db import session_scope
    from app.llm.factory import Role, get_llm
    from app.models import ApplicationDocument, JobAnalysis
    from app.services import letters

    letters.interactive = __import__("contextlib").nullcontext  # background work: user runs go first
    ids = json.loads((FT / "eval_jobs.json").read_text())["job_ids"]
    base_setup = get_llm()
    judge = base_setup.target(Role.VALIDATOR)
    results = {}
    for name, model in models.items():
        setup = replace(base_setup, targets=dict(base_setup.targets))
        setup.targets[Role.WRITER] = replace(base_setup.target(Role.WRITER), model=model, fallback=None)
        setup.targets[Role.VALIDATOR] = judge  # the base model judges both writers (never self-grading)
        rows = []
        for job_id in ids:
            t = time.monotonic()
            try:
                group = letters.generate_letter(job_id, setup=setup, use_editor=False, use_humanizer=False,
                                                personalization_review=False)
            except Exception as exc:  # noqa: BLE001
                print(f"  {name} job {job_id}: {exc}", flush=True)
                continue
            with session_scope() as s:
                final = s.scalars(select(ApplicationDocument).where(ApplicationDocument.group_id == group,
                                                                    ApplicationDocument.stage == "final")).first()
                draft_rec = s.scalars(select(JobAnalysis).where(JobAnalysis.job_id == job_id, JobAnalysis.kind == "letter_draft")
                                      .order_by(JobAnalysis.id.desc())).first()
                m = letter_metrics(final.validation or {}, final.content, final.language)
                m.update(job_id=job_id, language=final.language, total_seconds=round(time.monotonic() - t),
                         writer_seconds=round((draft_rec.duration_ms or 0) / 1000, 1) if draft_rec else None)
            rows.append(m)
            print(f"  {name} job {job_id}: {json.dumps(m)}", flush=True)
        agg = {k: round(statistics.mean(r[k] for r in rows), 2) for k in
               ("penalty", "unsupported", "ambiguous", "tone", "outline_added", "invented_salutation", "stock_phrases",
                "words", "total_seconds")} if rows else {}
        agg["writer_seconds"] = round(statistics.mean(r["writer_seconds"] for r in rows if r["writer_seconds"] is not None), 1) if rows else None
        results[name] = {"mean": agg, "letters": rows}
        print(name, json.dumps(agg), flush=True)
    base, ft = results["base"]["mean"], results["finetuned"]["mean"]
    keep = bool(base and ft and ft["penalty"] <= base["penalty"] - 0.5 and ft["unsupported"] <= base["unsupported"])
    return {"results": results, "keep_finetuned": keep,
            "rule": "mean penalty at least 0.5 lower and no more unsupported claims per draft"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=FT_MODELS, required=True)
    args = ap.parse_args()
    os.environ["DATABASE_URL"] = f"sqlite:///{SCRATCH}"
    os.environ["SCHEDULER_ENABLED"] = "false"
    sys.path.insert(0, "/app")
    from app.llm.factory import Role, get_llm

    base_model = get_llm().target(Role.WRITER if args.task == "writer" else Role.SCREEN).model
    models = {"base": base_model, "finetuned": FT_MODELS[args.task]}
    report = eval_writer(models) if args.task == "writer" else eval_screen(models)
    report["models"] = models
    (FT / f"report_{args.task}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("KEEP fine-tuned" if report["keep_finetuned"] else "KEEP base", "—", report["rule"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
