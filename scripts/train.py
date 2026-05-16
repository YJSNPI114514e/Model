#!/usr/bin/env python3
"""GRIM 自然言語学習（ローカルファイル / Hugging Face datasets）。"""

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
from grim.training import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRIM language model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # ローカル UTF-8 ファイル
  python scripts/train.py --data data/sample_corpus.txt --fast

  # Hugging Face（wiki40b 日本語）
  python scripts/train.py --dataset fn-aka-mur/wiki40b_ja --fast --max-samples 50000

  # ストリーミング + 件数制限（メモリ節約）
  python scripts/train.py --dataset fn-aka-mur/wiki40b_ja --streaming --max-samples 100000 --fast
        """,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--data",
        type=str,
        default=None,
        help="UTF-8 テキストファイル",
    )
    src.add_argument(
        "--dataset",
        type=str,
        default=None,
        help='Hugging Face データセット名（例: fn-aka-mur/wiki40b_ja）',
    )
    p.add_argument("--split", type=str, default="train", help="データセット split")
    p.add_argument("--text-column", type=str, default=None, help="テキスト列名（自動推定可）")
    p.add_argument("--dataset-config", type=str, default=None, help="load_dataset の config 名")
    p.add_argument("--max-samples", type=int, default=None, help="使用する最大行数")
    p.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="HF データセットの最大文字数（既定: 150000）",
    )
    p.add_argument("--data-files", type=str, default=None, help="datasets ライブラリ用データファイルパス")
    p.add_argument("--streaming", action="store_true", help="ストリーミング読み込み")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--D", type=int, default=256)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--no-natural-grad", action="store_true")
    p.add_argument("--fast", action="store_true", help="小型モデル + Euler ODE（高速）")
    p.add_argument("--amp", action="store_true", help="GPU 混合精度（cuda 時のみ）")
    p.add_argument("--resume", action="store_true", help="既存のチェックポイントから再開")
    return p.parse_args()


def resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def print_device_info(device: str) -> None:
    print(f"device={device}")
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")
    elif device.startswith("cuda"):
        print("  警告: --device cuda ですが torch.cuda.is_available()=False")
        print("  → .\\setup_env_gpu.ps1 を実行するか、--device cpu を使ってください")


def load_corpus(args: argparse.Namespace, vocab: CharVocab | None = None) -> TextCorpus:
    if args.dataset:
        args.dataset = args.dataset.strip().strip("'\"")
        from grim.data.hf_dataset import load_hf_corpus

        streaming = args.streaming
        if args.max_samples and not streaming:
            print("注意: 大規模データセットのため --streaming を自動で有効にします。")
            streaming = True
        if not args.max_samples and not streaming:
            print(
                "警告: 全件読み込みは時間・容量が大きいです。"
                " まずは --max-samples 50000 --streaming を推奨します。"
            )

        print(f"Loading Hugging Face dataset: {args.dataset} (split={args.split}) ...", flush=True)
        corpus, col = load_hf_corpus(
            args.dataset,
            split=args.split,
            text_column=args.text_column,
            max_samples=args.max_samples,
            max_chars=args.max_chars if args.max_chars is not None else 150_000,
            streaming=streaming,
            dataset_config=args.dataset_config,
            cache_dir=ROOT / "data" / "cache",
            vocab=vocab,
            data_files=args.data_files,
        )
        chars = len(corpus.text)
        print(
            f"  column={col}  chars={chars:,}  vocab={corpus.vocab.size}  "
            f"max_chars={args.max_chars}",
            flush=True,
        )
        print("  encoding corpus for training (freeing raw text) ...", flush=True)
        corpus.prepare_for_training()
        print(f"  token_ids={len(corpus.token_ids):,}", flush=True)
        return corpus

    path = args.data or str(ROOT / "data" / "sample_corpus.txt")
    path = path.strip().strip("'\"")
    print(f"Loading local file: {path}")
    return TextCorpus(path, vocab=vocab)


def main() -> None:
    import traceback

    args = parse_args()
    try:
        _main(args)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None


def _main(args: argparse.Namespace) -> None:
    device_name = resolve_device(args.device)
    device = torch.device(device_name)
    print_device_info(device_name)

    ckpt_path = ROOT / args.checkpoint_dir / "best.pt"
    ckpt = None
    vocab = None
    if args.resume:
        if ckpt_path.exists():
            print(f"Resuming from checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "vocab" in ckpt:
                from grim.data.text import CharVocab
                vocab = CharVocab.from_state(ckpt["vocab"])
        else:
            print(f"Warning: --resume specified but {ckpt_path} not found. Starting from scratch.")

    corpus = load_corpus(args, vocab=vocab)
    
    # Use config from checkpoint if available, otherwise create new
    if ckpt and "config" in ckpt:
        config = ckpt["config"]
        config.device = device_name
        config.epochs = args.epochs # Allow changing epochs
        config.batch_size = args.batch_size
        print(f"Using existing config from checkpoint (D={config.D}, V={config.V})")
    else:
        config = GRIMConfig(
            task_mode="lm",
            D=args.D or 256,
            V=max(corpus.vocab.size, 64),
            M_max=args.seq_len,
            seq_len=args.seq_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device_name,
            K=min(10, args.D or 256),
        )
        if args.fast:
            user_d = args.D
            config.apply_fast_preset()
            if user_d != 256:
                config.D = user_d
    
    if args.no_natural_grad:
        config.use_natural_grad = False

    use_cuda = device_name.startswith("cuda")
    train_loader, val_loader, vocab = get_lm_loaders(
        corpus,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        pin_memory=use_cuda,
    )
    config.V = max(config.V, vocab.size)

    model = GRIM(config)
    if ckpt:
        model.load_state_dict(ckpt["model"])
        print("Successfully loaded model weights.")

    src = args.dataset or corpus.source
    print(
        f"NLP/LM  source={src}  vocab={vocab.size}  seq_len={config.seq_len}  D={config.D}  "
        f"ode={config.ode_solver}({config.euler_steps})  fast={args.fast}  amp={args.amp and use_cuda}"
    )
    payload_extra = {"dataset": args.dataset, "data": args.data, "corpus_source": corpus.source}
    from grim.training import train
    train(
        model,
        train_loader,
        val_loader,
        config,
        checkpoint_dir=ROOT / args.checkpoint_dir,
        vocab=vocab,
        extra_ckpt=payload_extra,
        use_amp=args.amp and use_cuda,
    )


if __name__ == "__main__":
    main()
