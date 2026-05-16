#!/usr/bin/env python3
"""NLP スモークテスト（次トークン予測 + 生成）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from grim.bootstrap import ensure_torch, reexec_in_venv_if_needed

reexec_in_venv_if_needed()
ensure_torch()

import torch

from grim.config import GRIMConfig
from grim.data.text import TextCorpus, get_lm_loaders
from grim.geometry import check_unitarity
from grim.model import GRIM


def main() -> None:
    corpus = TextCorpus(ROOT / "data" / "sample_corpus.txt")
    config = GRIMConfig(task_mode="lm", V=max(corpus.vocab.size, 64), device="cpu")
    config.apply_fast_preset()
    train_loader, _, _ = get_lm_loaders(corpus, seq_len=config.seq_len, batch_size=config.batch_size)
    model = GRIM(config)

    x, y = next(iter(train_loader))
    out = model.forward_train_lm(x, y)
    assert check_unitarity(out["psi0"])
    assert check_unitarity(out["psi_T"])
    print(
        "forward_train_lm OK",
        f"loss={out['loss'].item():.4f}",
        f"fm={out['loss_fm'].item():.4f}",
        f"lm={out['loss_obs'].item():.4f}",
    )

    pred = model.predict_next_token(x)
    print("predict_next_token OK", "targets=", y[:8].tolist(), "preds=", pred[:8].tolist())

    prompt = torch.tensor([corpus.vocab.encode("GRIM", max_len=config.M_max)], dtype=torch.long)
    gen = model.generate(prompt, max_new_tokens=20)
    print("generate OK", corpus.vocab.decode(gen))
    print("GRIM NLP demo passed.")


if __name__ == "__main__":
    main()
