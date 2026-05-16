"""複素RKHSトークナイザー（第2.2節）。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from grim.geometry import complex_inner, normalize_state


class ComplexTokenizer(nn.Module):
    """
    |ψ₀⟩ = Z^{-1/2} Σ_j α_j e^{iφ_j} |e_{t_j}⟩
    α_j = softmax(w_α · Re(|e_{t_j}⟩))
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        max_len: int,
        w_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len
        self.w_alpha = w_alpha

        re = torch.randn(vocab_size, dim) / math.sqrt(dim)
        im = torch.randn(vocab_size, dim) / math.sqrt(dim)
        self.embeddings = nn.Parameter(torch.complex(re, im))

        self.w_phi = nn.Parameter(torch.tensor(1.0))
        self.b_phi = nn.Parameter(torch.tensor(0.0))

    def phase(self, positions: Tensor) -> Tensor:
        """φ_j = w_φ · j / M_max + b_φ"""
        j = positions.to(dtype=self.w_phi.dtype, device=positions.device)
        return self.w_phi * j / self.max_len + self.b_phi

    def token_states(self, token_ids: Tensor) -> Tensor:
        """|e_t⟩ for each token [B, L, D]"""
        return self.embeddings[token_ids]

    def forward(self, token_ids: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        token_ids: [B, L]
        returns: |ψ₀⟩ [B, D]
        """
        emb = self.token_states(token_ids)
        B, L, D = emb.shape
        pos = torch.arange(L, device=token_ids.device).view(1, L).expand(B, L)
        phase = self.phase(pos)
        rotated = emb * torch.exp(1j * phase).unsqueeze(-1)

        scores = self.w_alpha * rotated.real.sum(dim=-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)

        psi = torch.sum(alpha.unsqueeze(-1) * rotated, dim=1)
        z = torch.sum(torch.abs(alpha) ** 2, dim=-1, keepdim=True).clamp_min(1e-8)
        psi = psi / torch.sqrt(z)
        return normalize_state(psi)

    def inject(self, psi: Tensor, token_id: Tensor) -> Tensor:
        """生成時: 新トークンを状態に重ねる簡易注入"""
        if token_id.dim() == 0:
            token_id = token_id.view(1)
        emb = self.embeddings[token_id]
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        mixed = psi + 0.5 * emb
        return normalize_state(mixed)
