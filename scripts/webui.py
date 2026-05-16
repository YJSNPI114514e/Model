#!/usr/bin/env python3
"""GRIM Web UI for text generation and training."""

from __future__ import annotations

import io
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grim.bootstrap import ensure_torch
from grim.config import GRIMConfig
from grim.data.text import CharVocab, TextCorpus, get_lm_loaders
from grim.model import GRIM
from grim.training import train, load_checkpoint

try:
    import gradio as gr
except ImportError as exc:
    raise ImportError(
        "gradio is required for this web UI. Install it with: pip install gradio"
    ) from exc


def resolve_device(device: str | None) -> str:
    if device and device.strip():
        return device.strip()
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_corpus(
    data_path: str | None,
    dataset_name: str | None,
    dataset_config: str | None,
    split: str,
    text_column: str | None,
    max_samples: int | None,
    max_chars: int | None,
    streaming: bool,
    vocab: CharVocab | None = None,
) -> TextCorpus:
    if dataset_name:
        from grim.data.hf_dataset import load_hf_corpus

        if max_samples and not streaming:
            streaming = True
        corpus, _ = load_hf_corpus(
            dataset_name,
            split=split,
            text_column=text_column,
            max_samples=max_samples,
            max_chars=max_chars if max_chars is not None else 150_000,
            streaming=streaming,
            dataset_config=dataset_config,
            cache_dir=ROOT / "data" / "cache",
        )
        corpus.prepare_for_training()
        return corpus

    if data_path and Path(data_path).exists():
        return TextCorpus(Path(data_path), vocab=vocab)
    return TextCorpus(ROOT / "data" / "sample_corpus.txt", vocab=vocab)


def generate_text(
    checkpoint: str,
    prompt: str,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float | None,
    top_k: int | None,
    greedy: bool,
    data_path: str | None,
    fast: bool,
    device: str | None,
) -> str:
    import torch
    import traceback

    try:
        ensure_torch()
        device_name = resolve_device(device)
        device = torch.device(device_name)
        max_new_tokens = int(max_new_tokens)
        min_new_tokens = int(min_new_tokens)
        top_k = int(top_k) if top_k is not None and top_k != 0 else None
        corpus = load_corpus(
            data_path,
            None,
            None,
            "train",
            None,
            None,
            None,
            False,
        )
        ckpt_path = Path(checkpoint or "checkpoints/best.pt")
        if not ckpt_path.is_absolute():
            ckpt_path = ROOT / ckpt_path

        if not ckpt_path.exists():
            return f"Checkpoint not found: {ckpt_path}\nPlease train a model first."

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt.get("config", GRIMConfig(task_mode="lm"))
        if not isinstance(config, GRIMConfig):
            config = GRIMConfig(task_mode="lm", **config)
        config.device = str(device)
        if fast:
            config.apply_fast_preset()

        model = GRIM(config).to(device)
        load_checkpoint(model, ckpt_path, device)
        model.init_history()

        vocab = None
        if "vocab" in ckpt:
            from grim.data.text import CharVocab

            vocab = CharVocab.from_state(ckpt["vocab"])
        else:
            vocab = corpus.vocab

        prompt_ids = vocab.encode_prompt(prompt or "", max_len=config.M_max)
        if not prompt_ids:
            return "プロンプトを入力してください。"

        unknown = [ch for ch in prompt if vocab.char2id.get(ch, vocab.UNK) == vocab.UNK]
        generated = model.generate(
            torch.tensor([prompt_ids], dtype=torch.long, device=device),
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            greedy=greedy,
        )
        text = vocab.decode(generated)
        warning = ""
        if unknown:
            warning = f"\n[警告] 未知文字が含まれています: {unknown}\n"
        return f"prompt: {prompt}\n---\n{text}{warning}"
    except Exception:
        return f"エラーが発生しました:\n{traceback.format_exc()}"


def train_model(
    data_path: str,
    dataset_name: str,
    dataset_config: str,
    split: str,
    text_column: str,
    max_samples: int,
    max_chars: int,
    streaming: bool,
    epochs: int,
    batch_size: int,
    seq_len: int,
    D: int,
    fast: bool,
    amp: bool,
    device: str,
    checkpoint_dir: str,
    no_natural_grad: bool,
    resume: bool,
) -> str:
    import torch

    ensure_torch()
    device_name = resolve_device(device)
    max_samples = int(max_samples) if max_samples not in (None, "", 0) else None
    max_chars = int(max_chars) if max_chars not in (None, "", 0) else None
    epochs = int(epochs)
    batch_size = int(batch_size)
    seq_len = int(seq_len)
    D = int(D)
    out = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(out):
            device = torch.device(device_name)
            ckpt_path = Path(checkpoint_dir) / "best.pt"
            if not ckpt_path.is_absolute():
                ckpt_path = ROOT / ckpt_path
            
            ckpt = None
            vocab = None
            if resume:
                if ckpt_path.exists():
                    print(f"Resuming from checkpoint: {ckpt_path}")
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                    if "vocab" in ckpt:
                        vocab = CharVocab.from_state(ckpt["vocab"])
                else:
                    print(f"Warning: Resume requested but {ckpt_path} not found. Starting from scratch.")

            corpus = load_corpus(
                data_path,
                dataset_name,
                dataset_config,
                split,
                text_column,
                max_samples,
                max_chars,
                streaming,
                vocab=vocab,
            )
            
            if ckpt and "config" in ckpt:
                config = ckpt["config"]
                config.device = device_name
                config.epochs = epochs
                config.batch_size = batch_size
                print(f"Using config from checkpoint (D={config.D}, V={config.V})")
            else:
                config = GRIMConfig(
                    task_mode="lm",
                    D=D,
                    V=max(corpus.vocab.size, 64),
                    M_max=seq_len,
                    seq_len=seq_len,
                    epochs=epochs,
                    batch_size=batch_size,
                    device=device_name,
                    K=min(10, D),
                )
                if fast:
                    config.apply_fast_preset()
                    config.D = D
            
            if no_natural_grad:
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
                print("Loaded model weights.")
                
            print(f"Training on: {corpus.source}")
            print(f"device={device_name}")
            if use_cuda and torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")

            train(
                model,
                train_loader,
                val_loader,
                config,
                checkpoint_dir=Path(checkpoint_dir),
                vocab=vocab,
                extra_ckpt={"dataset": dataset_name, "data": data_path, "corpus_source": corpus.source},
                use_amp=amp and use_cuda,
            )
    except Exception:
        print(traceback.format_exc(), file=out)
    return out.getvalue()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="GRIM Web UI") as demo:
        gr.Markdown("# GRIM Web UI - 文章生成と学習")

        with gr.Tab("文章生成"):
            gen_checkpoint = gr.Textbox(label="Checkpoint path", value="checkpoints/best.pt")
            gen_data_path = gr.Textbox(label="ローカル学習データ (オプション)", value="data/sample_corpus.txt")
            gen_prompt = gr.Textbox(label="プロンプト", value="GRIMは", lines=2)
            with gr.Row():
                gen_max_tokens = gr.Number(label="max_new_tokens", value=64, precision=0)
                gen_min_tokens = gr.Number(label="min_new_tokens", value=1, precision=0)
            with gr.Row():
                gen_temp = gr.Number(label="temperature", value=1.0)
                gen_top_k = gr.Number(label="top_k", value=10, precision=0)
            gen_greedy = gr.Checkbox(label="greedy", value=False)
            gen_fast = gr.Checkbox(label="fast preset", value=False)
            gen_device = gr.Textbox(label="device", value="", placeholder="cuda or cpu or leave empty")
            gen_button = gr.Button("生成する")
            gen_output = gr.Textbox(label="生成結果", lines=10)
            gen_button.click(
                generate_text,
                inputs=[
                    gen_checkpoint,
                    gen_prompt,
                    gen_max_tokens,
                    gen_min_tokens,
                    gen_temp,
                    gen_top_k,
                    gen_greedy,
                    gen_data_path,
                    gen_fast,
                    gen_device,
                ],
                outputs=[gen_output],
            )

        with gr.Tab("学習"):
            train_data_path = gr.Textbox(label="ローカル UTF-8 データファイル", value="data/sample_corpus.txt")
            train_dataset = gr.Textbox(label="Hugging Face dataset", value="")
            train_dataset_config = gr.Textbox(label="dataset config", value="")
            train_split = gr.Textbox(label="split", value="train")
            train_text_column = gr.Textbox(label="text column", value="")
            with gr.Row():
                train_max_samples = gr.Number(label="max_samples", value=10000, precision=0)
                train_max_chars = gr.Number(label="max_chars", value=150000, precision=0)
            train_streaming = gr.Checkbox(label="streaming", value=True)
            with gr.Row():
                train_epochs = gr.Number(label="epochs", value=10, precision=0)
                train_batch = gr.Number(label="batch_size", value=16, precision=0)
            with gr.Row():
                train_seq_len = gr.Number(label="seq_len", value=64, precision=0)
                train_D = gr.Number(label="D", value=256, precision=0)
            train_fast = gr.Checkbox(label="fast preset", value=False)
            train_amp = gr.Checkbox(label="AMP (cuda only)", value=False)
            train_no_natural = gr.Checkbox(label="no natural grad", value=False)
            train_device = gr.Textbox(label="device", value="", placeholder="cuda or cpu or leave empty")
            train_ckpt_dir = gr.Textbox(label="checkpoint dir", value="checkpoints")
            train_resume = gr.Checkbox(label="既存のチェックポイントから再開", value=False)
            train_button = gr.Button("学習を開始")
            train_output = gr.Textbox(label="ログ出力", lines=20)
            train_button.click(
                train_model,
                inputs=[
                    train_data_path,
                    train_dataset,
                    train_dataset_config,
                    train_split,
                    train_text_column,
                    train_max_samples,
                    train_max_chars,
                    train_streaming,
                    train_epochs,
                    train_batch,
                    train_seq_len,
                    train_D,
                    train_fast,
                    train_amp,
                    train_device,
                    train_ckpt_dir,
                    train_no_natural,
                    train_resume,
                ],
                outputs=[train_output],
            )

    return demo


def main() -> None:
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()
