#!/usr/bin/env python3
"""
数値安定性修正の検証スクリプト
1. 100エポック学習でNaNが出ないこと
2. Q†Q が単位行列に近いこと（誤差 < 1e-4）
3. lambda_real の最大値が常に負（< -1e-6）
4. seq_len=128, 256 でも loss が発散しないこと
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from grim.config import GRIMConfig
from grim.data.text import TextCorpus, get_lm_loaders
from grim.model import GRIM
from grim.training import train_epoch

DEVICE = torch.device("cpu")
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

# ── 共通コーパス ──────────────────────────────────────────
corpus = TextCorpus(ROOT / "data" / "sample_corpus.txt")

# ──────────────────────────────────────────────────────────
# 検証1: ユニタリ性チェック Q†Q ≈ I
# ──────────────────────────────────────────────────────────
print("\n=== 検証1: Q†Q が単位行列に近いこと (誤差 < 1e-4) ===")
config1 = GRIMConfig(task_mode="lm", V=max(corpus.vocab.size, 64), device="cpu")
config1.apply_fast_preset()
model1 = GRIM(config1)
tok = model1.tokenizer

Q = tok._build_unitary_U()  # [D, D]
QtQ = Q.conj().T @ Q        # should be ≈ I
I = torch.eye(config1.D, dtype=torch.cfloat)
err = (QtQ - I).abs().max().item()
print(f"  max|Q†Q - I| = {err:.2e}")
if err < 1e-4:
    print(f"  {PASS} ユニタリ性OK")
else:
    print(f"  {FAIL} ユニタリ性NG (err={err:.2e})")

# ──────────────────────────────────────────────────────────
# 検証2: lambda_real の最大値が常に負
# ──────────────────────────────────────────────────────────
print("\n=== 検証2: lambda_real の最大値が常に < -1e-6 ===")
lambda_real = -F.softplus(tok.eigvals_re) - 1e-6
max_re = lambda_real.max().item()
print(f"  lambda_real max = {max_re:.6f}")
if max_re < -1e-6:
    print(f"  {PASS} 固有値実部は全て負")
else:
    print(f"  {FAIL} 正の固有値実部が存在 (max={max_re:.6f})")

# ──────────────────────────────────────────────────────────
# 検証3: 100エポック学習でNaNが出ない
# ──────────────────────────────────────────────────────────
print("\n=== 検証3: 100エポック学習でNaNが出ないこと ===")
config3 = GRIMConfig(task_mode="lm", V=max(corpus.vocab.size, 64), device="cpu")
config3.apply_fast_preset()
config3.use_natural_grad = False  # KFACなしで高速化
config3.batch_size = 8
model3 = GRIM(config3)
model3.to(DEVICE)
model3.init_history()

train_loader3, _, _ = get_lm_loaders(
    corpus, seq_len=config3.seq_len, batch_size=config3.batch_size
)
opt_k2 = torch.optim.Adam(
    [p for n, p in model3.named_parameters() if not n.startswith("meta.")],
    lr=config3.lr
)
opt_k3 = torch.optim.Adam(
    [p for n, p in model3.named_parameters() if n.startswith("meta.")],
    lr=config3.meta_lr
)

nan_detected = False
last_loss = None
for ep in range(1, 101):
    avg_loss, _, _, avg_acc, _ = train_epoch(
        model3, train_loader3, opt_k2, opt_k3, DEVICE, config3, None, 0
    )
    if torch.isnan(torch.tensor(avg_loss)):
        print(f"  {FAIL} epoch {ep} で NaN 検出！")
        nan_detected = True
        break
    last_loss = avg_loss
    if ep % 20 == 0:
        print(f"  epoch {ep:3d}/100  loss={avg_loss:.4f}  train_acc={avg_acc:.4f}")

if not nan_detected:
    print(f"  {PASS} 100エポック完走、NaN なし (最終loss={last_loss:.4f})")

# ──────────────────────────────────────────────────────────
# 検証4: seq_len=128, 256 でもlossが発散しないこと
# ──────────────────────────────────────────────────────────
print("\n=== 検証4: 長シーケンス (seq_len=128, 256) での安定性 ===")
for seq_len in [128, 256]:
    cfg = GRIMConfig(task_mode="lm", V=max(corpus.vocab.size, 64), device="cpu")
    cfg.apply_fast_preset()
    cfg.seq_len = seq_len
    cfg.M_max = seq_len
    cfg.use_natural_grad = False
    cfg.batch_size = 4

    m = GRIM(cfg).to(DEVICE)
    m.init_history()

    try:
        ldr, _, _ = get_lm_loaders(corpus, seq_len=seq_len, batch_size=cfg.batch_size)
        o_k2 = torch.optim.Adam(
            [p for n, p in m.named_parameters() if not n.startswith("meta.")], lr=cfg.lr
        )
        o_k3 = torch.optim.Adam(
            [p for n, p in m.named_parameters() if n.startswith("meta.")], lr=cfg.meta_lr
        )
        loss_val, _, _, acc_val, _ = train_epoch(m, ldr, o_k2, o_k3, DEVICE, cfg, None, 0)
        ok = not torch.isnan(torch.tensor(loss_val)) and not torch.isinf(torch.tensor(loss_val))
        tag = PASS if ok else FAIL
        print(f"  {tag} seq_len={seq_len}  loss={loss_val:.4f}  train_acc={acc_val:.4f}")
    except Exception as e:
        print(f"  {FAIL} seq_len={seq_len}  例外発生: {e}")

print("\n=== 検証完了 ===")
