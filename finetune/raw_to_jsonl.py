"""Step 1 — convert the downloaded public datasets (Parquet) to JSONL.

    python finetune/raw_to_jsonl.py            (host or training container; needs pandas + pyarrow)

Inputs  data/finetune/raw/coverletters_{train,test}.parquet   ShashiVish/cover-letter-dataset (1,162 English letters)
        data/finetune/raw/jobfit_{train,test}.parquet         cnamuangtoun/resume-job-description-fit (~8k pairs)
Outputs data/finetune/raw/coverletters.jsonl, jobfit.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "finetune" / "raw"


def convert(prefix: str, out: str) -> int:
    frames = [pd.read_parquet(p) for p in sorted(RAW.glob(f"{prefix}_*.parquet"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    with (RAW / out).open("w", encoding="utf-8") as fh:
        for row in df.to_dict(orient="records"):
            fh.write(json.dumps({k: (None if pd.isna(v) else v) for k, v in row.items()}, ensure_ascii=False) + "\n")
    return len(df)


if __name__ == "__main__":
    print("cover letters:", convert("coverletters", "coverletters.jsonl"))
    print("job fit pairs:", convert("jobfit", "jobfit.jsonl"))
