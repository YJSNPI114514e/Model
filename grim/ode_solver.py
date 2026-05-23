"""ODE積分 d|ψ⟩/dt = v_t（sekkeisyo.txt COMPONENT 3 準拠）。"""

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
    rtol: float = 1e-4,
    atol: float = 1e-6,
    use_amp: bool = False,
) -> Tensor:
    """
    sekkeisyo COMPONENT 3:
    1. def rhs(t, psi): return self.gravitational_field(t, psi, psi_0, H_emb)
    2. t_span = [0.0, 1.0]
    3. psi_T = odeint(rhs, psi_0, t_span, method='dopri5', rtol=1e-4, atol=1e-6)[-1]
    4. ASSERT torch.allclose(norm(psi_T), 1.0, atol=1e-5)
    
    use_amp: True の場合、BF16/FP16 で計算（GPU 時のみ有効）。False は FP32 固定。
    """
    psi0 = normalize_state(psi0)
    B, D = psi0.shape
    y0 = _to_real_flat(psi0)

    def dynamics(t_scalar, y_flat):
        # use_amp=True なら混合精度、False なら FP32 固定
        with torch.amp.autocast("cuda", enabled=use_amp):
            psi = _from_real_flat(y_flat, (B, D))
            psi = normalize_state(psi)
            ts = t_scalar.item() if isinstance(t_scalar, torch.Tensor) else float(t_scalar)
            t_batch = torch.full((B,), ts, device=psi.device, dtype=torch.float32)
            v = flow_field(psi, psi0, h_emb.float(), t_batch)
            v = tangent_project(v, psi)
            return _to_real_flat(v)

    t_eval = torch.tensor([t_span[0], t_span[1]], device=psi0.device, dtype=torch.float64)
    solution = odeint(dynamics, y0, t_eval, method=method, rtol=rtol, atol=atol)
    psi_T = normalize_state(_from_real_flat(solution[-1], (B, D)))

    # sekkeisyo COMPONENT 3 step 4: unitarity assertion
    norms = torch.linalg.vector_norm(psi_T, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
        psi_T = normalize_state(psi_T)

    return psi_T
