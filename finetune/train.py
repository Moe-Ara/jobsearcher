"""Step 5 — QLoRA fine-tuning with Unsloth (runs in the unsloth/unsloth:core GPU container, Ollama stopped).

    docker run --rm --gpus all -v %CD%:/work -w /work unsloth/unsloth:core \
        python finetune/train.py --task writer      (Qwen3-14B)
        python finetune/train.py --task screen      (Qwen3-8B)

LoRA rank 16 on all attention + MLP projections of the 4-bit base, loss on the assistant answers only, Qwen3
chat format with thinking disabled (exactly how the app calls the models). Afterwards the adapter is merged and
exported as GGUF Q5_K_M (same quantisation as the production base models) for Ollama.
Output: data/finetune/out/<task>/{lora/, gguf/*.gguf, metrics.json}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

TASKS = {
    "writer": {"base": "unsloth/Qwen3-14B-unsloth-bnb-4bit", "data": "writer", "epochs": 2, "lr": 1e-4, "max_seq": 12288},
    "screen": {"base": "unsloth/Qwen3-8B-unsloth-bnb-4bit", "data": "screen", "epochs": 2, "lr": 1e-4, "max_seq": 4608},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--root", default="data/finetune")
    ap.add_argument("--skip-gguf", action="store_true")
    ap.add_argument("--base", help="override the base model (e.g. a tiny model for a dry run)")
    ap.add_argument("--max-seq", type=int, help="override the max sequence length")
    ap.add_argument("--epochs", type=float, help="override the number of epochs")
    ap.add_argument("--out-name", help="output folder name (default: the task)")
    args = ap.parse_args()
    cfg = dict(TASKS[args.task])
    for key, value in (("base", args.base), ("max_seq", args.max_seq), ("epochs", args.epochs)):
        if value:
            cfg[key] = value
    root = Path(args.root)
    out = root / "out" / (args.out_name or args.task)
    out.mkdir(parents=True, exist_ok=True)

    from unsloth import FastLanguageModel  # noqa: I001 - unsloth must be imported before transformers/trl
    from unsloth.chat_templates import train_on_responses_only
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(cfg["base"], max_seq_length=cfg["max_seq"], load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=3407,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    def to_text(batch):
        return {"text": [tokenizer.apply_chat_template(m, tokenize=False, enable_thinking=False) for m in batch["messages"]]}

    ds = load_dataset("json", data_files={"train": str(root / f"{cfg['data']}_train.jsonl"),
                                          "validation": str(root / f"{cfg['data']}_valid.jsonl")})
    ds = ds.map(to_text, batched=True, remove_columns=ds["train"].column_names)
    before = {k: len(v) for k, v in ds.items()}
    ds = ds.filter(lambda ex: len(tokenizer(ex["text"]).input_ids) <= cfg["max_seq"])
    print("examples (before/after length filter):", before, {k: len(v) for k, v in ds.items()}, flush=True)

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=ds["train"], eval_dataset=ds["validation"],
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=cfg["max_seq"], per_device_train_batch_size=1,
            gradient_accumulation_steps=8, num_train_epochs=cfg["epochs"], learning_rate=cfg["lr"],
            warmup_ratio=0.05, lr_scheduler_type="cosine", logging_steps=5, eval_strategy="epoch",
            optim="adamw_8bit", weight_decay=0.01, bf16=True, output_dir=str(out / "checkpoints"),
            save_strategy="no", report_to="none", seed=3407,
        ),
    )
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                      response_part="<|im_start|>assistant\n")
    started = time.time()
    stats = trainer.train()
    metrics = {"task": args.task, "base": cfg["base"], "train_seconds": round(time.time() - started),
               "train_loss": stats.training_loss, "eval": trainer.evaluate(), "examples": {k: len(v) for k, v in ds.items()}}
    print(json.dumps(metrics, indent=1), flush=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    model.save_pretrained(str(out / "lora"))
    tokenizer.save_pretrained(str(out / "lora"))
    if not args.skip_gguf:
        export_gguf(model, tokenizer, out)
    return 0


LLAMA_CPP = Path("/opt/unsloth/llama.cpp")


def export_gguf(model, tokenizer, out: Path) -> Path:
    """Merge LoRA into 16-bit weights → GGUF bf16 (llama.cpp converter) → Q5_K_M. Done explicitly: Unsloth's one-call
    export fails silently in this image (the converter can't import llama.cpp's bundled gguf module)."""
    import os
    import shutil
    import subprocess

    merged = out / "merged"
    model.save_pretrained_merged(str(merged), tokenizer, save_method="merged_16bit")
    bf16, q5 = out / "model-bf16.gguf", out / "model-q5_k_m.gguf"
    env = {**os.environ, "PYTHONPATH": str(LLAMA_CPP / "gguf-py")}
    subprocess.run(["python", str(LLAMA_CPP / "convert_hf_to_gguf.py"), str(merged), "--outfile", str(bf16),
                    "--outtype", "bf16"], check=True, env=env)
    subprocess.run([str(LLAMA_CPP / "build" / "bin" / "llama-quantize"), str(bf16), str(q5), "Q5_K_M"], check=True)
    bf16.unlink()  # tens of GB for a 14B model — only the quantised file is needed
    shutil.rmtree(merged, ignore_errors=True)
    print("GGUF:", q5, f"{q5.stat().st_size / 1e9:.1f} GB", flush=True)
    return q5


if __name__ == "__main__":
    raise SystemExit(main())
