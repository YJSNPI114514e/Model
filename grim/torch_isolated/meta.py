"""K=3 メタ学習層（sekkeisyo.txt 準拠）。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _inv_softplus(x: float) -> float:
    """softplus(result) ≈ x となる初期値を計算。"""
    return math.log(math.exp(x) - 1.0)


class MetaParams(nn.Module):
    """
    K3 メタパラメータ群 (sekkeisyo.txt K3_PARAMETERS)
    全重みは softplus() を通して使用し、常に正を保証する。
    """

    def __init__(self, beta: float = 0.01) -> None:
        super().__init__()
        self.beta = beta
        # sekkeisyo K3_PARAMETERS: 初期値は softplus 適用後の値で指定
        self.fm_weight = nn.Parameter(torch.tensor(_inv_softplus(0.5)))      # softplus ≈ 0.5
        self.obs_weight = nn.Parameter(torch.tensor(_inv_softplus(1.0)))     # softplus ≈ 1.0
        self.meta_lambda = nn.Parameter(torch.tensor(_inv_softplus(0.01)))   # softplus ≈ 0.01
        self.meta_mu = nn.Parameter(torch.tensor(_inv_softplus(0.001)))      # softplus ≈ 0.001
        self.meta_sigma = nn.Parameter(torch.tensor(_inv_softplus(0.1)))     # softplus ≈ 0.1
        self.meta_beta = nn.Parameter(torch.tensor(_inv_softplus(0.1)))      # softplus ≈ 0.1
        self.register_buffer("w_hist", torch.zeros(6))

    def effective_weights(self) -> dict[str, Tensor]:
        """softplus 適用後の実効値を返す。"""
        return {
            "fm_weight": F.softplus(self.fm_weight),
            "obs_weight": F.softplus(self.obs_weight),
            "meta_lambda": F.softplus(self.meta_lambda),
            "meta_mu": F.softplus(self.meta_mu),
            "meta_sigma": F.softplus(self.meta_sigma),
            "meta_beta": F.softplus(self.meta_beta),
        }

    def as_dict(self) -> dict[str, float]:
        return {k: float(v.detach()) for k, v in self.effective_weights().items()}

    def kl_to_history(self) -> Tensor:
        w = torch.stack([
            self.fm_weight, self.obs_weight, self.meta_lambda,
            self.meta_mu, self.meta_sigma, self.meta_beta,
        ])
        w_pos = torch.softmax(w, dim=0)
        hist = torch.softmax(self.w_hist, dim=0).clamp_min(1e-8)
        return torch.sum(w_pos * (torch.log(w_pos.clamp_min(1e-8)) - torch.log(hist)))

    def compute_meta_loss(self, obs_loss: Tensor) -> Tensor:
        """sekkeisyo: meta_loss = mean(L_obs) + beta * KL(meta_weights || historical)"""
        return obs_loss + F.softplus(self.meta_beta) * self.kl_to_history()

    @torch.no_grad()
    def momentum_update(self, momentum: float = 0.9) -> None:
        w = torch.stack([
            self.fm_weight, self.obs_weight, self.meta_lambda,
            self.meta_mu, self.meta_sigma, self.meta_beta,
        ])
        self.w_hist.mul_(momentum).add_(w * (1.0 - momentum))

    def apply_natural_grad_step(self, lr: float, eps: float = 1e-4) -> None:
        """G(w)⁻¹∇F の対角近似（backward 後に .grad を使用）"""
        for p in self.parameters():
            if p.grad is None:
                continue
            fisher = p.grad.detach() ** 2 + eps
            p.data.add_(-lr * p.grad / fisher)
            p.grad = None
