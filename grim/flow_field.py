"""学習可能ベクトル場 ComplexMLP（sekkeisyo.txt COMPONENT 2 準拠）。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def inv_softplus(y: float) -> float:
    return math.log(math.exp(y) - 1.0)


class EnergyVectorField(nn.Module):
    """
    E(psi) = E_world + E_self + E_cont + E_explore
    v = dψ/dt = -∇E
    """

    def __init__(
        self,
        dim: int,
        hidden: int,
        history_dim: int,
        tokenizer: nn.Module,
        history_getter: callable,
        generation: nn.Module,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.tokenizer = tokenizer
        self.history_getter = history_getter
        self.generation = generation

        # Learnable scalars (softplus parameters)
        self.raw_lam = nn.Parameter(torch.tensor(-4.6))    # lam: softplus(-4.6) ≈ 0.01
        self.raw_mu = nn.Parameter(torch.tensor(-4.6))     # mu: softplus(-4.6) ≈ 0.01
        self.raw_sigma = nn.Parameter(torch.tensor(0.0))    # sigma: softplus(0.0) ≈ 0.693
        self.raw_beta = nn.Parameter(torch.tensor(-4.6))    # beta: softplus(-4.6) ≈ 0.01

    def energy(self, psi: Tensor, psi0: Tensor) -> Tensor:
        # 1. World potential (E_world)
        overlap = complex_inner(psi, psi0, dim=-1)
        overlap_sq = torch.abs(overlap)**2
        e_world = -torch.log(overlap_sq.clamp_min(1e-8))

        # 2. Self-energy (E_self)
        e_norm = torch.sum(torch.abs(psi)**2, dim=-1)
        lam = F.softplus(self.raw_lam)
        e_self = lam * e_norm

        # 3. Continuation potential (E_cont)
        history = self.history_getter()
        entries = history._entries if history is not None else []
        if not entries:
            e_cont = torch.zeros(psi.shape[0], device=psi.device, dtype=psi.real.dtype)
        else:
            psis = torch.stack([e.psi.squeeze(0) for e in entries], dim=0)  # [H_len, D]
            weights = torch.tensor([e.weight for e in entries], device=psi.device, dtype=psi.real.dtype)

            # overlaps: [B, H_len]
            overlaps = torch.mm(psi, psis.conj().T)
            abs_overlap = torch.abs(overlaps).clamp(0.0, 1.0)
            d = torch.acos(abs_overlap)  # Fubini-Study distance [B, H_len]

            sigma = F.softplus(self.raw_sigma).clamp_min(1e-8)
            kernel = torch.exp(- (d**2) / (2 * sigma**2))  # [B, H_len]

            mu = F.softplus(self.raw_mu)
            weighted_sum = torch.sum(weights.unsqueeze(0) * kernel, dim=-1)  # [B]
            e_cont = - (mu / len(entries)) * weighted_sum

        # 4. Exploration potential (E_explore)
        probs = self.generation.born_probs(psi)  # [B, V]
        beta = F.softplus(self.raw_beta)
        e_explore = - beta * torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

        return e_world + e_self + e_cont + e_explore

    def forward(
        self,
        psi: Tensor,
        psi0: Tensor,
        h_emb: Tensor,  # kept for signature compatibility
        t: Tensor,      # kept for signature compatibility
    ) -> Tensor:
        # dψ/dt = -∇E
        with torch.enable_grad():
            psi_r = psi.real.detach().requires_grad_(True)
            psi_i = psi.imag.detach().requires_grad_(True)
            psi_c = torch.complex(psi_r, psi_i)

            E = self.energy(psi_c, psi0).sum()

            grad_r, grad_i = torch.autograd.grad(E, [psi_r, psi_i], create_graph=True)
            v = torch.complex(-grad_r, -grad_i)

        return tangent_project(v, psi)

