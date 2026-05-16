#!/usr/bin/env python3
"""HF データセットの中身を数行だけ確認。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grim.bootstrap import ensure_torch, reexec_in_venv_if_needed

reexec_in_venv_if_needed()
ensure_torch()

from grim.data.hf_dataset import load_hf_text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", help="例: fn-aka-mur/wiki40b_ja")
    p.add_argument("--split", default="train")
    p.add_argument("--text-column", default=None)
    p.add_argument("--max-samples", type=int, default=3)
    p.add_argument("--streaming", action="store_true", default=True)
    args = p.parse_args()

    text, col = load_hf_text(
        args.dataset,
        split=args.split,
        text_column=args.text_column,
        max_samples=args.max_samples,
        streaming=args.streaming,
    )
    print(f"column={col}  chars={len(text):,}")
    for i, block in enumerate(text.split("\n")[: args.max_samples], 1):
        preview = block[:200].replace("\n", " ")
        print(f"--- [{i}] ---")
        print(preview)


if __name__ == "__main__":
    main()
