"""複素RKHSトークナイザー（第2.2節）。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # 固有値（複素数）の実部と虚部
        # 数値安定性のため、eigvals_re は負にバイアスする初期化
        self.eigvals_re = nn.Parameter(-0.1 + 0.02 * torch.randn(dim))
        self.eigvals_im = nn.Parameter(torch.zeros(dim))

        # 固有ベクトル行列 U の実部と虚部（ユニタリではない非正規行列を許容）
        self.U_re = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))
        self.U_im = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))

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

    def _build_unitary_U(self) -> Tensor:
        """
        U_re, U_im から複素行列を組み立て、QR分解でユニタリ化した Q を返す。
        Q†Q = I が保証されるため、逆行列は共役転置 Q.conj().T で代替可能。
        torch.linalg.qr は微分可能。
        """
        U_raw = torch.complex(self.U_re, self.U_im)  # [D, D]
        Q, _ = torch.linalg.qr(U_raw)               # Q: [D, D] ユニタリ
        return Q

    def _get_T_power(self, j: int | float) -> Tensor:
        """
        指定された j に対する T^j ∈ C^{D×D} を返す。
        T = U diag(λ_1,...,λ_D) U†  （U はユニタリ）
        T^j = U diag(λ_1^j,...,λ_D^j) U†
        """
        # [修正2] 固有値実部を -softplus で常に負値に強制（発散防止）
        lambda_real = -F.softplus(self.eigvals_re) - 1e-6  # 常に < 0
        lambda_pow = torch.exp(
            j * torch.complex(lambda_real, self.eigvals_im)
        )  # [D]

        # [修正1] QR分解でユニタリ化。逆行列は共役転置で代替
        Q = self._build_unitary_U()  # [D, D]
        Q_inv = Q.conj().T           # [D, D]

        # T^j = Q @ diag(λ^j) @ Q†
        return Q @ (lambda_pow.unsqueeze(-1) * Q_inv)

    def forward(self, token_ids: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        token_ids: [B, L]
        returns: |ψ₀⟩ [B, D]
        """
        emb = self.token_states(token_ids)
        B, L, D = emb.shape

        # Positions: j = 0, ..., L-1
        j_indices = torch.arange(L, dtype=self.U_re.dtype, device=token_ids.device)

        # [修正2] 固有値実部を -softplus で常に負値に強制（j が大きくなっても発散しない）
        lambda_real = -F.softplus(self.eigvals_re) - 1e-6  # [D]、常に < 0
        # Lambda_pow[j, k] = exp(j * (lambda_real[k] + i * eigvals_im[k]))
        lambda_pow_all = torch.exp(
            j_indices.unsqueeze(-1) * torch.complex(lambda_real, self.eigvals_im).unsqueeze(0)
        )  # [L, D]

        # [修正1] QR分解でユニタリ化。U_inv = Q† = Q.conj().T（逆行列計算不要）
        Q = self._build_unitary_U()   # [D, D]
        Q_inv = Q.conj().T            # [D, D]

        # T^j = Q @ diag(Λ^j) @ Q†
        T_all = Q.unsqueeze(0) @ (lambda_pow_all.unsqueeze(-1) * Q_inv.unsqueeze(0))  # [L, D, D]

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
