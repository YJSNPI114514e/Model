"""フビニ・スタディ計量・測地線・接空間射影（第2.1–2.4節）。"""

from __future__ import annotations

import torch
from torch import Tensor


def complex_inner(a: Tensor, b: Tensor, dim: int = -1) -> Tensor:
    """⟨a|b⟩ = Σ conj(a) * b"""
    return torch.sum(a.conj() * b, dim=dim)


def normalize_state(psi: Tensor, eps: float = 1e-8) -> Tensor:
    """‖|ψ⟩‖ = 1"""
    n = torch.linalg.vector_norm(psi, dim=-1, keepdim=True).clamp_min(eps)
    return psi / n


def tangent_project(v: Tensor, psi: Tensor) -> Tensor:
    """Π_tangent(v) = v - ⟨ψ|v⟩|ψ⟩（複素内積）"""
    radial = complex_inner(psi, v, dim=-1).unsqueeze(-1) * psi
    return v - radial


def fs_angle(psi0: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """θ = arccos|⟨ψ₀|target⟩|"""
    overlap = torch.abs(complex_inner(psi0, target, dim=-1)).clamp(0.0, 1.0 - eps)
    return torch.arccos(overlap)


def geodesic_interp(
    psi0: Tensor,
    target: Tensor,
    t: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    |ψ_t⟩ = sin((1-t)θ)/sinθ · |ψ₀⟩ + sin(tθ)/sinθ · |target⟩
    t: [B] or scalar
    """
    theta = fs_angle(psi0, target, eps=eps)
    sin_theta = torch.sin(theta).clamp_min(eps)

    if t.dim() == 0:
        t = t.view(1)
    if t.dim() == 1:
        t = t.unsqueeze(-1)

    c0 = torch.sin((1.0 - t) * theta.unsqueeze(-1)) / sin_theta.unsqueeze(-1)
    c1 = torch.sin(t * theta.unsqueeze(-1)) / sin_theta.unsqueeze(-1)
    psi_t = c0 * psi0 + c1 * target
    return normalize_state(psi_t)


def geodesic_velocity_target(
    psi0: Tensor,
    target: Tensor,
    t: Tensor,
    psi_t: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """
    v_target = (-θ cos((1-t)θ)/sinθ)|ψ₀⟩ + (θ cos(tθ)/sinθ)|o_y⟩
    接空間へ射影。
    """
    theta = fs_angle(psi0, target, eps=eps)
    sin_theta = torch.sin(theta).clamp_min(eps)

    if t.dim() == 0:
        t = t.view(1)
    if t.dim() == 1:
        t = t.unsqueeze(-1)

    coef0 = -theta.unsqueeze(-1) * torch.cos((1.0 - t) * theta.unsqueeze(-1)) / sin_theta.unsqueeze(-1)
    coef1 = theta.unsqueeze(-1) * torch.cos(t * theta.unsqueeze(-1)) / sin_theta.unsqueeze(-1)
    v = coef0 * psi0 + coef1 * target

    if psi_t is None:
        psi_t = geodesic_interp(psi0, target, t.squeeze(-1) if t.shape[-1] == 1 else t, eps=eps)
    return tangent_project(v, psi_t)


def fs_norm_sq(v: Tensor) -> Tensor:
    """‖·‖²_FS（実ユークリッドノルムで近似）"""
    return torch.sum(torch.abs(v) ** 2, dim=-1)


def check_unitarity(psi: Tensor, atol: float = 1e-5) -> bool:
    n = torch.linalg.vector_norm(psi, dim=-1)
    return bool(torch.allclose(n, torch.ones_like(n), atol=atol))
