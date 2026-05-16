#!/usr/bin/env python3
"""GRIM 自己回帰テキスト生成（第4.3節）。"""

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
from grim.data.text import CharVocab, TextCorpus, get_lm_loaders
from grim.model import GRIM
from grim.training import load_checkpoint, train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRIM text generation")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--prompt", type=str, default="GRIMは")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--min-new-tokens", type=int, default=1)
    p.add_argument("--train-first", action="store_true")
    p.add_argument("--data", type=str, default=str(ROOT / "data" / "sample_corpus.txt"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--show-ids", action="store_true", help="生成トークン ID を表示")
    p.add_argument("--fast", action="store_true", help="高速モード（Euler・小型）")
    p.add_argument("--greedy", action="store_true", help="greedy（繰り返しやすい）")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    return p.parse_args()


def load_vocab_from_ckpt(ckpt: dict, corpus: TextCorpus) -> CharVocab:
    if "vocab" in ckpt:
        return CharVocab.from_state(ckpt["vocab"])
    print("注意: checkpoint に語彙がありません。コーパスから再構築します（学習時と同じ --data を使ってください）。")
    return corpus.vocab


def explain_empty(raw_ids: list[int], vocab: CharVocab) -> None:
    if not raw_ids:
        print("原因: 0 トークン生成（学習不足、または max-new-tokens=0）")
        return
    names = {0: "<pad>", 1: "<eos>", 2: "<unk>"}
    labels = [names.get(i, vocab.id2char.get(i, f"id={i}")) for i in raw_ids[:10]]
    print(f"生成トークン(先頭): {raw_ids[:10]}  ->  {labels}")
    if all(i <= 2 for i in raw_ids):
        print(
            "原因: 特殊トークン(PAD/EOS/UNK)のみ。対処:\n"
            "  1) 十分学習する: python scripts/train.py --epochs 50\n"
            "  2) プロンプトをコーパスに含まれる語で始める（例: 意識は）"
        )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    corpus = TextCorpus(args.data)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    if args.train_first or not ckpt_path.exists():
        config = GRIMConfig(task_mode="lm", V=max(corpus.vocab.size, 64), device=str(device))
        config.apply_fast_preset() if args.fast else None
        config.epochs = args.epochs
        config.batch_size = 8
        if not args.fast:
            config.D = 256
            config.M_max = 64
            config.seq_len = 64
        model = GRIM(config).to(device)
        model.init_history()
        train_loader, val_loader, vocab = get_lm_loaders(
            corpus, seq_len=config.seq_len, batch_size=8
        )
        print("言語モデルを学習中...")
        train(
            model,
            train_loader,
            val_loader,
            config,
            checkpoint_dir=ROOT / "checkpoints",
            vocab=vocab,
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt.get("config", GRIMConfig(task_mode="lm"))
        if not isinstance(config, GRIMConfig):
            config = GRIMConfig(task_mode="lm", **config)
        config.device = str(device)
        if args.fast:
            config.apply_fast_preset()
        model = GRIM(config).to(device)
        load_checkpoint(model, ckpt_path, device)
        model.init_history()
        vocab = load_vocab_from_ckpt(ckpt, corpus)

    # 学習時と同じ語彙サイズであること
    if model.config.V < vocab.size:
        print(f"警告: モデル V={model.config.V} < 語彙 {vocab.size}。学習し直すか同じ checkpoint を使ってください。")

    prompt_ids_list = vocab.encode_prompt(args.prompt, max_len=config.M_max)
    if not prompt_ids_list:
        print("エラー: プロンプトが空、または語彙に無い文字のみです。")
        sys.exit(1)

    unknown = [ch for ch in args.prompt if vocab.char2id.get(ch, vocab.UNK) == vocab.UNK]
    if unknown:
        print(f"警告: 語彙に無い文字があります（<unk> になります）: {unknown}")

    prompt_ids = torch.tensor([prompt_ids_list], dtype=torch.long, device=device)
    generated = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        greedy=args.greedy,
    )
    text = vocab.decode(generated)

    print(f"prompt: {args.prompt!r}")
    print(f"generated: {text!r}")
    if args.show_ids or not text:
        print(f"raw_token_ids ({len(generated)}): {generated}")
    if not text:
        explain_empty(generated, vocab)


if __name__ == "__main__":
    main()
