"""学習可能ベクトル場 ComplexMLP（第2.3.3節）。"""

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


class FlowVectorField(nn.Module):
    """v_pred = Π_tangent(MLP(|ψ⟩, |ψ₀⟩, H_emb, t))"""

    def __init__(self, dim: int, hidden: int, history_dim: int, n_layers: int = 3) -> None:
        super().__init__()
        self.dim = dim
        self.history_dim = history_dim
        in_dim = dim * 2 + history_dim + 2

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
        B = psi.shape[0]
        device, dtype = psi.device, psi.dtype

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

        t_feat = torch.cat([t, torch.sin(math.pi * t)], dim=-1)
        h_c = torch.complex(h_emb, torch.zeros(B, self.history_dim, device=device))
        t_c = torch.complex(t_feat, torch.zeros(B, 2, device=device))
        return torch.cat([psi, psi0, h_c, t_c], dim=-1)

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
        return tangent_project(x, psi)
