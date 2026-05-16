"""学習可能ベクトル場 ComplexMLP（sekkeisyo.txt COMPONENT 2 準拠）。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from grim.geometry import complex_inner, tangent_project


class ComplexLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int) -> None:
        super().__init__()
        scale = 1.0 / math.sqrt(in_f)
        self.w_re = nn.Parameter(torch.randn(out_f, in_f) * scale)
        self.w_im = nn.Parameter(torch.randn(out_f, in_f) * scale)
        self.b_re = nn.Parameter(torch.zeros(out_f))
        self.b_im = nn.Parameter(torch.zeros(out_f))

    def forward(self, x: Tensor) -> Tensor:
        w = torch.complex(self.w_re, self.w_im)
        b = torch.complex(self.b_re, self.b_im)
        return x @ w.T + b


def sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    """
    sekkeisyo COMPONENT 2: t_emb = sinusoidal_embedding(t, D)
    正弦波位置埋め込み（Transformer 式だが attention ではない）。
    t: [B, 1] or [B]  →  [B, dim]
    """
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
    )
    args = t * freqs.unsqueeze(0)  # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim] or [B, dim-1]
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(t)], dim=-1)
    return emb


class EnergyVectorField(nn.Module):
    """
    E(psi) = -log|⟨psi|psi_0⟩|² + λ·‖psi‖² + μ·MLP(psi, h_emb)
    v = dψ/dt = -∇E
    """

    def __init__(self, dim: int, hidden: int, history_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.lam = nn.Parameter(torch.tensor(-2.0))  # softplus(-2) ≈ 0.126
        self.mu = nn.Parameter(torch.tensor(-2.0))
        
        self.history_mlp = nn.Sequential(
            nn.Linear(dim * 2 + history_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def energy(self, psi: Tensor, psi0: Tensor, h_emb: Tensor) -> Tensor:
        # -log|⟨psi|psi_0⟩|²
        overlap = complex_inner(psi, psi0, dim=-1)
        overlap_sq = torch.abs(overlap)**2
        e_base = -torch.log(overlap_sq.clamp_min(1e-8))
        
        # ‖psi‖²
        e_norm = torch.sum(torch.abs(psi)**2, dim=-1)
        
        B = psi.shape[0]
        if h_emb.shape[0] != B:
            h_emb = h_emb.expand(B, -1)
            
        feat = torch.cat([psi.real, psi.imag, h_emb], dim=-1)
        e_hist = self.history_mlp(feat).squeeze(-1)
        
        import torch.nn.functional as F
        return e_base + F.softplus(self.lam) * e_norm + F.softplus(self.mu) * e_hist

    def forward(
        self,
        psi: Tensor,
        psi0: Tensor,
        h_emb: Tensor,
        t: Tensor,
    ) -> Tensor:
        # 勾配流 dψ/dt = -∇E
        with torch.enable_grad():
            psi_r = psi.real.detach().requires_grad_(True)
            psi_i = psi.imag.detach().requires_grad_(True)
            psi_c = torch.complex(psi_r, psi_i)
            
            E = self.energy(psi_c, psi0, h_emb).sum()
            
            grad_r, grad_i = torch.autograd.grad(E, [psi_r, psi_i], create_graph=True)
            v = torch.complex(-grad_r, -grad_i)
            
        return tangent_project(v, psi)

