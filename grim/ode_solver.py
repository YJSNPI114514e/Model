"""ODE積分 d|ψ⟩/dt = v_t（第4.2節, Dopri5）。"""

from __future__ import annotations

import torch
from torch import Tensor
from torchdiffeq import odeint

from grim.geometry import normalize_state, tangent_project


def _to_real_flat(psi: Tensor) -> Tensor:
    return torch.view_as_real(psi).reshape(-1)


def _from_real_flat(flat: Tensor, shape: tuple[int, ...]) -> Tensor:
    B, D = shape
    return torch.view_as_complex(flat.view(B, D, 2))


def integrate_flow(
    flow_field,
    psi0: Tensor,
    h_emb: Tensor,
    t_span: tuple[float, float] = (0.0, 1.0),
    method: str = "dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> Tensor:
    """|ψ_T⟩ = ODE_Solve(|ψ₀⟩, flow_field, t: 0→1)"""
    psi0 = normalize_state(psi0)
    B, D = psi0.shape
    y0 = _to_real_flat(psi0)

    def dynamics(t_scalar, y_flat):
        psi = _from_real_flat(y_flat, (B, D))
        psi = normalize_state(psi)
        ts = t_scalar.item() if isinstance(t_scalar, torch.Tensor) else float(t_scalar)
        t_batch = torch.full((B,), ts, device=psi.device, dtype=torch.float32)
        v = flow_field(psi, psi0, h_emb, t_batch)
        v = tangent_project(v, psi)
        return _to_real_flat(v)

    t_eval = torch.tensor([t_span[0], t_span[1]], device=psi0.device, dtype=torch.float64)
    solution = odeint(dynamics, y0, t_eval, method=method, rtol=rtol, atol=atol)
    return normalize_state(_from_real_flat(solution[-1], (B, D)))


def integrate_flow_euler(
    flow_field,
    psi0: Tensor,
    h_emb: Tensor,
    steps: int = 32,
) -> Tensor:
    dt = 1.0 / steps
    psi = normalize_state(psi0)
    for i in range(steps):
        t = torch.full((psi.shape[0],), i * dt, device=psi.device, dtype=torch.float32)
        v = flow_field(psi, psi0, h_emb, t)
        psi = normalize_state(psi + dt * v)
    return psi
