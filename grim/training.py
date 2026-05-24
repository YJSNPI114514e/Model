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


def _get_psi_T_numpy(psi_T: torch.Tensor) -> "np.ndarray":
    """torch.Tensor から NumPy 配列へ変換（複素数対応）。"""
    import numpy as np
    return psi_T.detach().cpu().numpy()


def _evaluate_val_loss(
    model: GRIM,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    """検証損失と平均ψ_T を計算。
    
    Returns:
        (avg_val_loss, mean_psi_T) のタプル。
    """
    import numpy as np
    
    model.eval()
    total_loss = 0.0
    total_samples = 0
    psi_T_list = []
    
    # Save and clear history
    saved_history = []
    for layer in [model.history.short_term, model.history.mid_term, model.history.long_term]:
        saved_history.extend(list(layer))
    model.history.clear()
    
    try:
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    # LM モード：(x, y, is_doc_start) の形式
                    if model.config.task_mode == "lm" and batch[2].dtype == torch.bool:
                        x, y, is_doc_start = batch
                        mask = None
                    else:
                        x, y, mask = batch
                        is_doc_start = None
                else:
                    x, y = batch
                    mask = None
                    is_doc_start = None
                x, y = x.to(device), y.to(device)
                if mask is not None:
                    mask = mask.to(device)
                
                # 文書境界で履歴をクリア（評価時も同様）
                if is_doc_start is not None and model.config.task_mode == "lm":
                    doc_start_indices = is_doc_start.nonzero(as_tuple=True)[0]
                    if len(doc_start_indices) > 0:
                        model.history.clear()
                
                psi0 = model.tokenize(x, mask)
                h_emb = model.summarize_history(x.shape[0])
                psi_T = model.integrate(psi0, h_emb, use_amp=False)
                
                # Born Rule 確率
                probs = model.generation.born_probs(psi_T)
                
                # NLL 損失
                log_probs = torch.log(probs.clamp_min(1e-8))
                loss = F.nll_loss(log_probs, y, reduction="sum")
                total_loss += loss.item()
                total_samples += y.numel()
                
                # ψ_T を保存（バッチ平均）
                psi_T_np = _get_psi_T_numpy(psi_T.mean(dim=0))
                psi_T_list.append(psi_T_np)
    finally:
        # Restore history
        model.history.clear()
        for e in saved_history:
            model.history._entries.append(e)
    
    avg_loss = total_loss / max(total_samples, 1)
    
    # 平均ψ_T を計算
    import numpy as np
    mean_psi_T = np.mean(np.array(psi_T_list), axis=0) if psi_T_list else np.zeros(1)
    
    model.train()
    return avg_loss, mean_psi_T


@torch.no_grad()
def evaluate_lm(model: GRIM, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    """次トークン精度、パープレキシティ、P(y_true) 平均。"""
    model.eval()
    
    # Save training history by collecting from all layers, clear it for evaluation, restore it afterward
    saved_history = []
    for layer in [model.history.short_term, model.history.mid_term, model.history.long_term]:
        saved_history.extend(list(layer))
    model.history.clear()
    
    correct = total = 0
    total_nll = 0.0
    total_p_true = 0.0
    try:
        for batch in loader:
            if len(batch) == 3:
                # LM モード：(x, y, is_doc_start) の形式
                if model.config.task_mode == "lm" and batch[2].dtype == torch.bool:
                    x, y, is_doc_start = batch
                    mask = None
                else:
                    x, y, mask = batch
                    is_doc_start = None
            else:
                x, y = batch
                mask = None
                is_doc_start = None
            x, y = x.to(device), y.to(device)
            if mask is not None:
                mask = mask.to(device)
            
            # 文書境界で履歴をクリア（評価時も同様）
            if is_doc_start is not None and model.config.task_mode == "lm":
                doc_start_indices = is_doc_start.nonzero(as_tuple=True)[0]
                if len(doc_start_indices) > 0:
                    model.history.clear()
            
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
    K2_PARAMETERS — meta.* と flow_field.raw_* を除く全パラメータ
    flow_field.raw_* は K-3 層のパラメータ（カーネルリッジ回帰で更新）
    """
    return [p for n, p in model.named_parameters() 
            if not n.startswith("meta.") and not n.startswith("flow_field.raw_")]


def _collect_k3_params(model: GRIM) -> list[torch.nn.Parameter]:
    """
    sekkeisyo PARAMETER GROUPS:
    K3_PARAMETERS — meta.* と flow_field.raw_* （ベクトル場のハイパーパラメータ）
    これらのパラメータはカーネルリッジ回帰またはメタ勾配で更新される
    """
    return [p for n, p in model.named_parameters() 
            if n.startswith("meta.") or n.startswith("flow_field.raw_")]


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
        # バッチの形式に応じて処理
        # LM モード：(x, y, is_doc_start) の 3 つ、または (x, y) の 2 つ
        # Classify モード：(x, y, mask) の 3 つ、または (x, y) の 2 つ
        if len(batch) == 3:
            # 3 つ目の要素が is_doc_start (bool テンソル) か mask かを判別
            if model.config.task_mode == "lm" and batch[2].dtype == torch.bool:
                x, y, is_doc_start = batch
                mask = None
            else:
                x, y, mask = batch
                is_doc_start = None
        else:
            x, y = batch
            mask = None
            is_doc_start = None
            
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        
        # 文書境界フラグがある場合、該当サンプルの履歴をクリア
        if is_doc_start is not None and model.config.task_mode == "lm":
            with torch.no_grad():
                # is_doc_start が True のサンプルのインデックスを取得
                doc_start_indices = is_doc_start.nonzero(as_tuple=True)[0]
                if len(doc_start_indices) > 0:
                    # 文書先頭のサンプルがある場合、履歴をクリア
                    # バッチ全体で共通の履歴を使うため、1 つでも文書先頭があればクリア
                    model.history.clear()

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
        # 改良：LM モードでも履歴を更新する（文書境界リセットによりリークは防止）
        with torch.no_grad():
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
    
    # K=3 カーネルリッジ回帰メタ学習器（オプション）
    k3_krr = None
    if config.use_k3_kernel_ridge:
        from grim.k3_meta_learner import K3MetaLearner
        k3_krr = K3MetaLearner(
            gamma=config.k3_krr_gamma,
            max_buffer_size=config.k3_krr_max_buffer,
            update_interval=config.k3_krr_update_interval,
            smoothing_factor=config.k3_krr_smoothing,
        )
        print(f"K=3 Kernel Ridge Regression enabled (gamma={config.k3_krr_gamma}, interval={config.k3_krr_update_interval})")

    for epoch in range(1, config.epochs + 1):
        avg_loss, avg_fm, avg_obs, avg_train_acc, global_step = train_epoch(
            model, train_loader, k2_optimizer, k3_optimizer,
            device, config, kfac, global_step, 
            use_amp=use_amp, use_bf16=use_bf16, grad_accum_steps=grad_accum_steps,
        )
        if config.task_mode == "lm":
            acc, ppl, avg_p_true = evaluate_lm(model, val_loader, device)
            
            # K=3 カーネルリッジ回帰によるメタパラメータ更新
            if k3_krr is not None and k3_krr.should_update(epoch):
                val_loss, psi_T_mean = _evaluate_val_loss(model, val_loader, device)
                
                # メタパラメータの現在値を取得
                meta_dict = model.meta.as_dict()
                lam = meta_dict.get('meta_lambda', 0.01)
                beta = meta_dict.get('meta_beta', 0.01)
                
                # データを蓄積
                k3_krr.accumulate(psi_T_mean, val_loss, lam, beta)
                
                # 十分なデータがあれば更新
                if len(k3_krr.psi_buffer) >= 10:
                    lam_new, beta_new, weights = k3_krr.update(lam, beta)
                    
                    # 移動平均で急激な変化を防止
                    lam_smoothed, beta_smoothed = k3_krr.smooth_update(lam, beta, lam_new, beta_new)
                    
                    # モデルのメタパラメータを更新
                    with torch.no_grad():
                        # softplus の逆関数を通してパラメータを設定
                        import math
                        def inv_softplus(x: float) -> float:
                            return math.log(math.exp(max(x, 1e-8)) - 1.0)
                        
                        model.meta.meta_lambda.copy_(torch.tensor(inv_softplus(lam_smoothed)))
                        model.meta.meta_beta.copy_(torch.tensor(inv_softplus(beta_smoothed)))
                    
                    print(f"  K3-KRR update: lambda={lam_smoothed:.6f}, beta={beta_smoothed:.6f}")
            
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
                "config": config,  # チェックポイントに設定を保存
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
    """チェックポイントからモデルをロード。
    
    チェックポイントに含まれる設定パラメータを自動的に検出し、
    モデルの設定をそれに合わせて調整してから重みを読み込みます。
    """
    from grim.model import GRIM
    from grim.config import GRIMConfig
    
    device = device or next(model.parameters()).device
    ckpt = torch.load(path, map_location=device, weights_only=False)
    
    # チェックポイントに設定情報が含まれているか確認
    if "config" in ckpt:
        # チェックポイントから設定を読み取り、モデルを再構築
        ckpt_config = ckpt["config"]
        print(f"Loading checkpoint with config: D={ckpt_config.D}, D_h={ckpt_config.D_h}, flow_hidden={ckpt_config.flow_hidden}")
        
        # モデルを再構築（同じアーキテクチャを持つ新しいインスタンスを作成）
        new_model = GRIM(ckpt_config)
        new_model.to(device)
        new_model.load_state_dict(ckpt["model"])
        new_model.eval()
        return new_model
    else:
        # 設定情報がない場合、既存のモデル構造で読み込みを試みる
        # エラーが発生する可能性あり
        try:
            model.load_state_dict(ckpt["model"])
        except RuntimeError as e:
            if "size mismatch" in str(e):
                # サイズミマッチの場合、チェックポイントから次元を推測
                state_dict = ckpt["model"]
                
                # tokenizer.emb_re の形状から D を推測
                if "tokenizer.emb_re.weight" in state_dict:
                    emb_shape = state_dict["tokenizer.emb_re.weight"].shape
                    inferred_D = emb_shape[1]  # [V, D]
                    inferred_D_h = state_dict.get("history_embedder.proj_re.weight", torch.zeros(1,1)).shape[1] // 4 if "history_embedder.proj_re.weight" in state_dict else inferred_D // 2
                    
                    print(f"Size mismatch detected. Inferring from checkpoint: D={inferred_D}, D_h≈{inferred_D_h}")
                    print("To fix this properly, please create a GRIMConfig with matching parameters and rebuild the model.")
                raise e
        
        model.to(device)
        model.eval()
        return model
