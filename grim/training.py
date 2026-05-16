"""訓練ループ（第4.1節 Algorithm）。"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from grim.config import GRIMConfig
from grim.data.text import CharVocab
from grim.model import GRIM
from grim.natural_grad import KFACNaturalGradient


@torch.no_grad()
def evaluate_lm(model: GRIM, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """次トークン精度とパープレキシティ（近似）。"""
    model.eval()
    correct = total = 0
    total_nll = 0.0
    for batch in loader:
        if len(batch) == 3:
            x, y, mask = batch
        else:
            x, y = batch
            mask = None
        x, y = x.to(device), y.to(device)
        if mask is not None:
            mask = mask.to(device)
        pred = model.predict_next_token(x, mask)
        correct += (pred == y).sum().item()
        total += y.numel()
        psi0 = model.tokenize(x, mask)
        h_emb = model.summarize_history(x.shape[0])
        psi_T = model.integrate(psi0, h_emb)
        total_nll += model.language_modeling_loss(psi_T, y).item() * y.numel()
    acc = correct / max(total, 1)
    ppl = float(torch.exp(torch.tensor(total_nll / max(total, 1))))
    return acc, ppl


@torch.no_grad()
def evaluate_classify(model: GRIM, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        if len(batch) == 3:
            x, y, mask = batch
        else:
            x, y = batch
            mask = None
        x, y = x.to(device), y.to(device)
        if mask is not None:
            mask = mask.to(device)
        pred, _, _ = model.classify(x, mask)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def evaluate(model: GRIM, loader: DataLoader, device: torch.device) -> float:
    if model.config.task_mode == "lm":
        acc, _ = evaluate_lm(model, loader, device)
        return acc
    return evaluate_classify(model, loader, device)


def train_epoch(
    model: GRIM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: GRIMConfig,
    kfac: KFACNaturalGradient | None,
    global_step: int,
    use_amp: bool = False,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for batch in tqdm(loader, desc="train", leave=False):
        if len(batch) == 3:
            x, y, mask = batch
        else:
            x, y = batch
            mask = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if mask is not None:
            mask = mask.to(device, non_blocking=True)

        do_meta = (global_step + 1) % config.k3_interval == 0
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model.forward_train_batch(x, y, mask)
            loss = out["loss"]
        if not torch.isfinite(loss):
            continue
        if use_amp:
            scaler.scale(loss).backward(retain_graph=do_meta)
        else:
            loss.backward(retain_graph=do_meta)

        if kfac is not None and config.use_natural_grad:
            kfac.precondition(model.parameters())

        if use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if model.config.task_mode == "classify":
            model.reorthogonalize_obs()

        if do_meta:
            for p in model.meta.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            meta_loss = model.meta.kl_to_history()
            meta_loss.backward()
            model.meta.apply_natural_grad_step(config.meta_lr)
            model.meta.momentum_update()

        with torch.no_grad():
            for psi in out["psi_T"]:
                model.history.decay()
                model.history.push(psi)

        global_step += 1
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1), global_step


def train(
    model: GRIM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: GRIMConfig,
    checkpoint_dir: str | Path = "checkpoints",
    vocab: CharVocab | None = None,
    extra_ckpt: dict | None = None,
    use_amp: bool = False,
) -> GRIM:
    device = torch.device(config.device)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA が使えません。.\\setup_env_gpu.ps1 で CUDA 版 PyTorch を入れるか、--device cpu を指定してください。"
        )
    model.to(device)
    model.init_history()
    if use_amp and not device.type == "cuda":
        use_amp = False

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    kfac = None
    if config.use_natural_grad:
        kfac = KFACNaturalGradient(model, damping=config.kfac_damping)
        kfac.register()

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_metric = 0.0
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        avg_loss, global_step = train_epoch(
            model, train_loader, optimizer, device, config, kfac, global_step, use_amp=use_amp
        )
        if config.task_mode == "lm":
            acc, ppl = evaluate_lm(model, val_loader, device)
            print(
                f"epoch {epoch}/{config.epochs}  loss={avg_loss:.4f}  "
                f"val_token_acc={acc:.4f}  val_ppl~{ppl:.2f}"
            )
            metric = acc
        else:
            acc = evaluate_classify(model, val_loader, device)
            print(f"epoch {epoch}/{config.epochs}  loss={avg_loss:.4f}  val_acc={acc:.4f}")
            metric = acc

        if metric >= best_metric:
            best_metric = metric
            path = ckpt_dir / "best.pt"
            payload: dict = {
                "model": model.state_dict(),
                "config": config,
                "meta": model.meta.as_dict(),
                "epoch": epoch,
                "val_metric": metric,
            }
            if config.task_mode == "lm":
                payload["val_ppl"] = ppl
            if vocab is not None:
                payload["vocab"] = vocab.to_state()
            if extra_ckpt:
                payload.update(extra_ckpt)
            torch.save(payload, path)
            print(f"  saved checkpoint -> {path}")

    if kfac is not None:
        kfac.remove()

    return model


def load_checkpoint(model: GRIM, path: str | Path, device: torch.device | None = None) -> GRIM:
    device = device or next(model.parameters()).device
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model
