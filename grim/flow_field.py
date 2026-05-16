"""学習可能ベクトル場 ComplexMLP（sekkeisyo.txt COMPONENT 2 準拠）。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from grim.geometry import tangent_project


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


class FlowVectorField(nn.Module):
    """
    v_pred = Π_tangent(MLP(ψ.real, ψ.imag, ψ₀.real, ψ₀.imag, H_emb, t_emb))
    
    sekkeisyo COMPONENT 2:
    - Input: real-valued concat [psi.real, psi.imag, psi0.real, psi0.imag, H_emb, t_emb]
    - Output: complex [B, D], projected onto tangent space
    """

    def __init__(self, dim: int, hidden: int, history_dim: int, n_layers: int = 3) -> None:
        super().__init__()
        self.dim = dim
        self.history_dim = history_dim
        # sekkeisyo: concat = [psi.real(D), psi.imag(D), psi0.real(D), psi0.imag(D), H_emb(D_h), t_emb(D)]
        in_dim = dim * 4 + history_dim + dim

        layers: list[nn.Module] = [ComplexLinear(in_dim, hidden)]
        for _ in range(n_layers - 2):
            layers.append(ComplexLinear(hidden, hidden))
        layers.append(ComplexLinear(hidden, dim))
        self.net = nn.ModuleList(layers)

    @staticmethod
    def _c_relu(z: Tensor) -> Tensor:
        return torch.complex(torch.relu(z.real), torch.relu(z.imag))

    def _build_features(
        self,
        psi: Tensor,
        psi0: Tensor,
        h_emb: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        sekkeisyo COMPONENT 2:
        concat = cat([psi.real, psi.imag, psi0.real, psi0.imag, H_emb, t_emb], dim=-1)
        """
        B = psi.shape[0]
        device = psi.device

        if t.dim() == 0:
            t = t.expand(B)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        if h_emb.shape[0] != B:
            h_emb = h_emb.expand(B, -1)
        if h_emb.shape[-1] != self.history_dim:
            if h_emb.shape[-1] < self.history_dim:
                h_emb = torch.nn.functional.pad(h_emb, (0, self.history_dim - h_emb.shape[-1]))
            else:
                h_emb = h_emb[..., : self.history_dim]

        # sekkeisyo: sinusoidal_embedding(t, D)
        t_emb = sinusoidal_embedding(t.squeeze(-1), self.dim)

        # All real-valued features
        real_features = torch.cat([
            psi.real, psi.imag,
            psi0.real, psi0.imag,
            h_emb,
            t_emb,
        ], dim=-1)

        # Convert to complex for ComplexMLP input
        return torch.complex(real_features, torch.zeros_like(real_features))

    def forward(
        self,
        psi: Tensor,
        psi0: Tensor,
        h_emb: Tensor,
        t: Tensor,
    ) -> Tensor:
        x = self._build_features(psi, psi0, h_emb, t)
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i < len(self.net) - 1:
                x = self._c_relu(x)
        # sekkeisyo COMPONENT 2: FORCE tangent space projection
        return tangent_project(x, psi)
