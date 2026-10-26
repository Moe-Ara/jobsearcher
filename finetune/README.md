# Fine-tuning the local models

A complete, repeatable pipeline: datasets → training data → QLoRA training → Ollama import → A/B evaluation →
keep the fine-tuned model **only if it measurably beats the base model**. Nothing here touches your real database:
data generation and evaluation run on `data/finetune/distill.db`, a backup copy.

| Step | Where | Command |
|---|---|---|
| 1. Public datasets → JSONL | host | `python finetune/raw_to_jsonl.py` |
| 2. Self-distilled letters | app container | `python -m finetune.distill_letters --n 40` |
| 3. Writer training set | app container | `python -m finetune.prepare_writer` |
| 4. Screener training set | app container | `python -m finetune.prepare_screen` |
| 5. QLoRA training | `unsloth/unsloth:core`, GPU, **Ollama stopped** | `python finetune/train.py --task writer` / `--task screen` |
| 6. Import into Ollama | host | `python finetune/export_ollama.py --task writer` |
| 7. A/B evaluation | app container | `python -m finetune.evaluate --task writer` / `--task screen` |

App-container commands: `docker compose exec -T -w /app -e PYTHONPATH=/app app <command>`.

## Datasets

- **ShashiVish/cover-letter-dataset** (Hugging Face, 1,159 English letters; no licence declared — personal, local
  use only, don't redistribute models trained on it). 81 % contain stock phrases the pipeline bans and many claim
  experience not in the applicant's inputs, so only letters passing the pipeline's own gates are used.
- **cnamuangtoun/resume-job-description-fit** (Hugging Face, 7,999 résumé/job pairs: No / Potential / Good Fit) — a
  balanced sample of 500 for the screener.
- **Self-distillation** (the main writer signal): the full pipeline (writer → German editor → fact check →
  corrections → outline guarantees) on real postings from your database; only letters that come out clean are
  kept, paired with the writer's exact prompt. No public German cover-letter dataset exists.
- **Your relevance labels** (Calibrate page): fold A (even job ids) trains the screener, fold B is the held-out test.

## Decision rules (evaluate.py)

- **Writer**: held-out jobs, base model as the only fact-check judge. Kept if the mean penalty (3× unsupported
  claim, 1.5× ambiguous, tone problems, outline parts the system had to add, invented salutation, stock phrases,
  length) is at least 0.5 lower **and** there are no more unsupported claims per draft.
- **Screener**: AUC (GREAT vs BAD) at least +0.02, no more BAD jobs passed to deep analysis, accuracy not lower.

To switch a model on, set `LOCAL_WRITER_MODEL=jobsearcher-writer-ft:latest` (and keep
`LOCAL_VALIDATOR_MODEL=hf.co/Qwen/Qwen3-14B-GGUF:Q5_K_M`, so the base model keeps fact-checking) or
`LOCAL_SCREEN_MODEL=jobsearcher-screen-ft:latest` in `.env`. To switch back, restore the base names.
