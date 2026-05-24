"""訓練ループ（sekkeisyo.txt CORRECT TRAINING LOOP 準拠）。"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from grim.config import GRIMConfig
from grim.data.text import CharVocab
from grim.model import GRIM
from grim.natural_grad import KFACNaturalGradient


@torch.no_grad()
def evaluate_lm(model: GRIM, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    """次トークン精度、パープレキシティ、P(y_true) 平均。"""
    model.eval()
    
    # Save training history, clear it for evaluation, restore it afterward
    saved_history = list(model.history._entries)
    model.history.clear()
    
    correct = total = 0
    total_nll = 0.0
    total_p_true = 0.0
    try:
        for batch in loader:
            if len(batch) == 3:
                x, y, mask = batch
            else:
                x, y = batch
                mask = None
            x, y = x.to(device), y.to(device)
            if mask is not None:
                mask = mask.to(device)
            psi0 = model.tokenize(x, mask)
            h_emb = model.summarize_history(x.shape[0])
            psi_T = model.integrate(psi0, h_emb, use_amp=False)

            # Born Rule 確率
            probs = model.generation.born_probs(psi_T)  # [B, V]
            pred = probs.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()

            # P(y_true): 正解トークンの平均確率
            p_true = probs[torch.arange(y.shape[0], device=device), y]  # [B]
            total_p_true += p_true.sum().item()

            # NLL for perplexity
            log_probs = torch.log(probs.clamp_min(1e-8))
            total_nll += F.nll_loss(log_probs, y, reduction="sum").item()
    finally:
        # Restore training history
        model.history.clear()
        for e in saved_history:
            model.history._entries.append(e)

    acc = correct / max(total, 1)
    ppl = float(torch.exp(torch.tensor(total_nll / max(total, 1))))
    avg_p_true = total_p_true / max(total, 1)
    return acc, ppl, avg_p_true


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
        acc, _, _ = evaluate_lm(model, loader, device)
        return acc
    return evaluate_classify(model, loader, device)


def _collect_k2_params(model: GRIM) -> list[torch.nn.Parameter]:
    """
    sekkeisyo PARAMETER GROUPS:
    K2_PARAMETERS — meta.* を除く全パラメータ
    """
    return [p for n, p in model.named_parameters() if not n.startswith("meta.")]


def _collect_k3_params(model: GRIM) -> list[torch.nn.Parameter]:
    """
    sekkeisyo PARAMETER GROUPS:
    K3_PARAMETERS — meta.* のみ
    """
    return [p for n, p in model.named_parameters() if n.startswith("meta.")]


def train_epoch(
    model: GRIM,
    loader: DataLoader,
    k2_optimizer: torch.optim.Optimizer,
    k3_optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: GRIMConfig,
    kfac: KFACNaturalGradient | None,
    global_step: int,
    use_amp: bool = False,
    use_bf16: bool = False,
    grad_accum_steps: int = 1,
) -> tuple[float, float, float, float, int]:
    model.train()
    total_loss = 0.0
    total_fm = 0.0
    total_obs = 0.0
    train_acc_sum = 0
    train_acc_total = 0
    n_batches = 0
    
    # AMP スケーラー（FP16 のみ。BF16 は不要）
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not use_bf16)

    # sekkeisyo: K2 params only for gradient clipping
    k2_params = _collect_k2_params(model)

    # 勾配累積用のゼロクリア初期化
    k2_optimizer.zero_grad(set_to_none=True)
    if (global_step + 1) % config.k3_interval == 0:
        k3_optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        if len(batch) == 3:
            x, y, mask = batch
        else:
            x, y = batch
            mask = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if mask is not None:
            mask = mask.to(device, non_blocking=True)

        do_meta = ((global_step + 1) % config.k3_interval == 0) and (batch_idx % grad_accum_steps == 0)

        # --- K=2: NATURAL GRADIENT UPDATE ---
        # sekkeisyo: K2_optimizer updates K2 params only
        # 勾配累積：grad_accum_steps > 1 の場合、zero_grad を呼ばない
        
        # 精度に応じた autocast コンテキスト
        if use_bf16:
            dtype = torch.bfloat16
        elif use_amp:
            dtype = torch.float16
        else:
            dtype = torch.float32
            
        with torch.amp.autocast("cuda", enabled=(use_amp or use_bf16), dtype=dtype):
            out = model.forward_train_batch(x, y, mask)
            # Flow Matching 損失を無効化し、total_loss = obs_loss のみとする
            loss = out["loss_obs"]
        
        # 勾配累積のためのスケーリング
        loss_scaled = loss / grad_accum_steps
        
        if not torch.isfinite(loss_scaled):
            global_step += 1
            continue

        if use_amp and not use_bf16:
            scaler.scale(loss_scaled).backward(retain_graph=do_meta)
        else:
            loss_scaled.backward(retain_graph=do_meta)

        # 勾配累積：grad_accum_steps ステップに一度だけ更新
        if (batch_idx + 1) % grad_accum_steps == 0 or batch_idx == len(loader) - 1:
            # sekkeisyo: KFAC preconditions K2 params ONLY (not meta)
            if kfac is not None and config.use_natural_grad:
                kfac.precondition(iter(k2_params))

            if use_amp and not use_bf16:
                scaler.unscale_(k2_optimizer)
            # sekkeisyo: clip K2 params only
            torch.nn.utils.clip_grad_norm_(k2_params, config.grad_clip)
            if use_amp and not use_bf16:
                scaler.step(k2_optimizer)
                scaler.update()
            else:
                k2_optimizer.step()
            
            # 次のバッチのためにゼロクリア
            k2_optimizer.zero_grad(set_to_none=True)

        if model.config.task_mode == "classify":
            model.reorthogonalize_obs()

        # Track train token accuracy
        with torch.no_grad():
            psi_T = out["psi_T"].detach()
            if model.config.task_mode == "lm":
                probs = model.generation.born_probs(psi_T)
                pred = probs.argmax(dim=-1)
                train_acc_sum += (pred == y).sum().item()
                train_acc_total += y.numel()
            else:
                probs = model.observation.born_probs(psi_T)
                pred = probs.argmax(dim=-1)
                train_acc_sum += (pred == y).sum().item()
                train_acc_total += y.numel()

        # --- HISTORY UPDATE ---
        # sekkeisyo: history_buffer.push(psi_T.detach())
        #            then decay all weights
        with torch.no_grad():
            if model.config.task_mode != "lm":
                if psi_T.dim() == 2 and psi_T.shape[0] > 0:
                    model.history.push(psi_T[0])
                model.history.decay()

        # --- K=3: META UPDATE (every k3_interval steps) ---
        # メタ損失: 全体の weighted loss を通して fm_weight/obs_weight に勾配を流す
        if do_meta:
            k3_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                meta_out = model.forward_train_batch(x, y, mask)
            # meta_out["loss"] = softplus(fm_w)*L_fm + softplus(obs_w)*L_obs
            # → fm_weight, obs_weight に勾配が流れる
            meta_total = meta_out["loss"] + F.softplus(model.meta.meta_beta) * model.meta.kl_to_history()
            meta_total.backward()
            k3_optimizer.step()
            model.meta.momentum_update()

        global_step += 1
        total_loss += loss.item()
        total_fm += 0.0  # Flow Matching 損失を無効化
        total_obs += out["loss_obs"].item()
        n_batches += 1

    d = max(n_batches, 1)
    avg_train_acc = train_acc_sum / max(train_acc_total, 1)
    return total_loss / d, total_fm / d, total_obs / d, avg_train_acc, global_step


def train(
    model: GRIM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: GRIMConfig,
    checkpoint_dir: str | Path = "checkpoints",
    vocab: CharVocab | None = None,
    extra_ckpt: dict | None = None,
    use_amp: bool = False,
    use_bf16: bool = False,
    grad_accum_steps: int = 1,
) -> GRIM:
    device = torch.device(config.device)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA が使えません。.\\setup_env_gpu.ps1 で CUDA 版 PyTorch を入れるか、--device cpu を指定してください。"
        )
    model.to(device)
    model.init_history()
    
    # AMP と BF16 の排他制御
    if use_bf16 and device.type == "cuda":
        use_amp = False  # BF16 を優先
        print("Using BF16 precision (Ampere+ GPUs)")
    elif use_amp and device.type != "cuda":
        use_amp = False
        use_bf16 = False
    
    # sekkeisyo VIOLATION 2: Split into K2 and K3 optimizers
    k2_params = _collect_k2_params(model)
    k3_params = _collect_k3_params(model)

    # sekkeisyo: K2 uses Natural Gradient (KFAC approximation)
    k2_optimizer = torch.optim.Adam(k2_params, lr=config.lr)
    
    # sekkeisyo: K3 uses separate Meta Gradient optimizer
    # K3 parameters might be empty if meta module has no trainable params
    if k3_params:
        k3_optimizer = torch.optim.Adam(k3_params, lr=config.meta_lr)
    else:
        # Create a dummy optimizer that does nothing
        k3_optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=config.meta_lr)
        # Remove the dummy parameter from the optimizer's param_groups to avoid updating it
        k3_optimizer.param_groups[0]['params'] = []

    kfac = None
    if config.use_natural_grad:
        kfac = KFACNaturalGradient(model, damping=config.kfac_damping)
        kfac.register()

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_metric = 0.0
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        avg_loss, avg_fm, avg_obs, avg_train_acc, global_step = train_epoch(
            model, train_loader, k2_optimizer, k3_optimizer,
            device, config, kfac, global_step, 
            use_amp=use_amp, use_bf16=use_bf16, grad_accum_steps=grad_accum_steps,
        )
        if config.task_mode == "lm":
            acc, ppl, avg_p_true = evaluate_lm(model, val_loader, device)
            meta_w = model.meta.as_dict()
            random_p = 1.0 / config.V
            print(
                f"epoch {epoch}/{config.epochs}  loss={avg_loss:.4f}  "
                f"train_acc={avg_train_acc:.6f}  "
                f"val_token_acc={acc:.6f}  val_ppl~{ppl:.2f}  "
                f"P(y_true)={avg_p_true:.6f} (random={random_p:.6f})  "
                f"fm_w=0.000  obs_w={meta_w['obs_weight']:.3f}"
            )
            metric = acc
        else:
            acc = evaluate_classify(model, val_loader, device)
            print(
                f"epoch {epoch}/{config.epochs}  loss={avg_loss:.4f}  "
                f"train_acc={avg_train_acc:.4f}  val_acc={acc:.4f}"
            )
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
