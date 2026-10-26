"""Step 6 — import a fine-tuned GGUF into Ollama under its own name (the base model stays untouched).

    python finetune/export_ollama.py --task writer      (host; uses the docker CLI)

The new model gets the base model's exact Modelfile (chat template, stop tokens, parameters) — only the weights
differ — so the app talks to it exactly like to the base. Names: jobsearcher-writer-ft / jobsearcher-screen-ft.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = {"writer": "hf.co/Qwen/Qwen3-14B-GGUF:Q5_K_M", "screen": "hf.co/Qwen/Qwen3-8B-GGUF:Q5_K_M"}


def sh(*cmd: str, **kw) -> str:
    return subprocess.run(list(cmd), check=True, capture_output=True, text=True, encoding="utf-8", **kw).stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=BASE, required=True)
    args = ap.parse_args()
    ggufs = sorted((ROOT / "data" / "finetune" / "out" / args.task).rglob("*.gguf"), key=lambda p: p.stat().st_size)
    ggufs = [g for g in ggufs if "q5_k_m" in g.name.lower()] or ggufs
    if not ggufs:
        print("no GGUF found — run train.py first", file=sys.stderr)
        return 1
    gguf = ggufs[-1]
    container = sh("docker", "compose", "ps", "-q", "ollama", cwd=ROOT).strip()
    base_file = sh("docker", "exec", container, "ollama", "show", "--modelfile", BASE[args.task])
    body = [ln for ln in base_file.splitlines() if not ln.startswith("FROM ") and not ln.startswith("#")]
    name = f"jobsearcher-{args.task}-ft"
    modelfile = "\n".join([f"FROM /root/.ollama/ft/{name}.gguf", *body]) + "\n"
    tmp = ROOT / "data" / "finetune" / "out" / args.task / "Modelfile"
    tmp.write_text(modelfile, encoding="utf-8")
    sh("docker", "exec", container, "mkdir", "-p", "/root/.ollama/ft")
    print(f"copying {gguf.name} ({gguf.stat().st_size / 1e9:.1f} GB) into the ollama volume…", flush=True)
    sh("docker", "cp", str(gguf), f"{container}:/root/.ollama/ft/{name}.gguf")
    sh("docker", "cp", str(tmp), f"{container}:/root/.ollama/ft/{name}.Modelfile")
    print(sh("docker", "exec", container, "ollama", "create", name, "-f", f"/root/.ollama/ft/{name}.Modelfile"))
    sh("docker", "exec", container, "rm", "-f", f"/root/.ollama/ft/{name}.gguf")  # ollama keeps its own blob copy
    print(f"created Ollama model {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
