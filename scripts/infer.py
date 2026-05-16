#!/usr/bin/env python3
"""次トークン予測の評価（自然言語）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grim.bootstrap import ensure_torch, reexec_in_venv_if_needed

reexec_in_venv_if_needed()
ensure_torch()

import torch

from grim.config import GRIMConfig
from grim.data.text import TextCorpus, get_lm_loaders
from grim.model import GRIM
from grim.training import evaluate_lm, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRIM LM evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data", type=str, default=str(ROOT / "data" / "sample_corpus.txt"))
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config: GRIMConfig = ckpt.get("config", GRIMConfig(task_mode="lm"))
    config.device = str(device)

    corpus = TextCorpus(args.data)
    _, val_loader, _ = get_lm_loaders(corpus, seq_len=config.seq_len, batch_size=16)

    model = GRIM(config)
    load_checkpoint(model, ckpt_path, device)
    acc, ppl = evaluate_lm(model, val_loader, device)
    print(f"val_token_acc={acc:.4f}  val_ppl≈{ppl:.2f}")


if __name__ == "__main__":
    main()
