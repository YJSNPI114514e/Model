"""複素RKHSトークナイザー（第2.2節）。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from grim.geometry import complex_inner, normalize_state


class ComplexTokenizer(nn.Module):
    """
    |ψ₀⟩ = Z^{-1/2} Σ_j α_j e^{iφ_j} T^j |e_{t_j}⟩
    α_j = softmax(w_α · Re(T^j |e_{t_j}⟩))
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

        self.emb_re = nn.Parameter(torch.randn(vocab_size, dim) / math.sqrt(dim))
        self.emb_im = nn.Parameter(torch.randn(vocab_size, dim) / math.sqrt(dim))

        self.w_phi = nn.Parameter(torch.tensor(1.0))
        self.b_phi = nn.Parameter(torch.tensor(0.0))

        # Unitary transition matrix parameters
        self.raw_A = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))
        self.raw_B = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))

    @property
    def embeddings(self) -> Tensor:
        return torch.complex(self.emb_re, self.emb_im)

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

        # Construct skew-Hermitian matrix X = A + iB
        A = self.raw_A - self.raw_A.T
        B = self.raw_B + self.raw_B.T
        X = torch.complex(A, B)

        # Positions: j = 0, ..., L-1
        j = torch.arange(L, dtype=self.raw_A.dtype, device=token_ids.device).view(L, 1, 1)
        X_all = j * X  # [L, D, D]

        # T^j = exp(j * X)
        T_all = torch.linalg.matrix_exp(X_all)  # [L, D, D]

        # Apply T^j to emb
        rotated = torch.einsum("lxy,bly->blx", T_all, emb)  # [B, L, D]

        # Apply phase rotation
        pos = torch.arange(L, device=token_ids.device).view(1, L).expand(B, L)
        phase = self.phase(pos)
        rotated = rotated * torch.exp(1j * phase).unsqueeze(-1)

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
