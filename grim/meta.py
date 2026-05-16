"""K=3 メタ学習層（第3.3節）。"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MetaParams(nn.Module):
    """
    w: 損失重み・温度・帯域幅など
    F(w) = E[L_FM(θ;w)] + β·KL(w||w_hist)
    """

    def __init__(self, beta: float = 0.01) -> None:
        super().__init__()
        self.beta = beta
        self.fm_weight = nn.Parameter(torch.tensor(1.0))
        self.obs_weight = nn.Parameter(torch.tensor(0.5))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("w_hist", torch.zeros(3))

    def as_dict(self) -> dict[str, float]:
        return {
            "fm_weight": float(self.fm_weight.detach()),
            "obs_weight": float(self.obs_weight.detach()),
            "temperature": float(self.temperature.detach()),
        }

    def kl_to_history(self) -> Tensor:
        w = torch.stack([self.fm_weight, self.obs_weight, self.temperature])
        w_pos = torch.softmax(w, dim=0)
        hist = torch.softmax(self.w_hist, dim=0).clamp_min(1e-8)
        return torch.sum(w_pos * (torch.log(w_pos.clamp_min(1e-8)) - torch.log(hist)))

    def meta_loss(self, fm_loss: Tensor, obs_loss: Tensor) -> Tensor:
        return self.fm_weight * fm_loss + self.obs_weight * obs_loss + self.beta * self.kl_to_history()

    @torch.no_grad()
    def momentum_update(self, momentum: float = 0.9) -> None:
        w = torch.stack([self.fm_weight, self.obs_weight, self.temperature])
        self.w_hist.mul_(momentum).add_(w * (1.0 - momentum))

    def apply_natural_grad_step(self, lr: float, eps: float = 1e-4) -> None:
        """G(w)⁻¹∇F の対角近似（backward 後に .grad を使用）"""
        for p in (self.fm_weight, self.obs_weight, self.temperature):
            if p.grad is None:
                continue
            fisher = p.grad.detach() ** 2 + eps
            p.data.add_(-lr * p.grad / fisher)
            p.grad = None
