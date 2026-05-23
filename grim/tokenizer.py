"""複素 RKHS トークナイザー（第 2.2 節）。状態空間モデル (SSM) 版。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from grim.geometry import complex_inner, normalize_state


class ComplexTokenizer(nn.Module):
    """
    |ψ₀⟩ = Z^{-1/2} Σ_j α_j e^{iφ_j} · |ψ_j⟩
    ここで |ψ_j⟩ = Â_j |ψ_{j-1}⟩ + B̂_j |e_j⟩
    Â_j: 選択的ユニタリ遷移作用素（入力依存）
    B̂_j: 選択的入力作用素
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

        # トークン埋め込み
        self.emb_re = nn.Parameter(torch.randn(vocab_size, dim) / math.sqrt(dim))
        self.emb_im = nn.Parameter(torch.randn(vocab_size, dim) / math.sqrt(dim))

        # 位相パラメータ
        self.w_phi = nn.Parameter(torch.tensor(1.0))
        self.b_phi = nn.Parameter(torch.tensor(0.0))

        # SSM パラメータ：基底遷移行列 A_base (歪エルミート化用)
        self.A_base_re = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))
        self.A_base_im = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))

        # SSM パラメータ：入力作用素 B
        self.B_re = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))
        self.B_im = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))

        # SSM パラメータ：状態依存補正の射影ベクトル delta_raw
        self.delta_proj_re = nn.Parameter(torch.randn(dim, 1) / math.sqrt(dim))
        self.delta_proj_im = nn.Parameter(torch.randn(dim, 1) / math.sqrt(dim))

        # 注意重み用
        self.omega_attn = nn.Parameter(torch.randn(dim) / math.sqrt(dim))

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

    def _build_skew_hermitian(self, X_re: Tensor, X_im: Tensor) -> Tensor:
        """
        実部・虚部から複素行列を組み立て、歪エルミート化 (X - X†) する。
        結果は exp(X) でユニタリ行列になる。
        """
        X = torch.complex(X_re, X_im)
        if X.dim() == 3:
            X_dagger = X.conj().transpose(-2, -1)
        else:
            X_dagger = X.conj().T
        return X - X_dagger

    def _compute_selective_unitary(self, psi_prev: Tensor) -> Tensor:
        """
        状態 ψ_prev に基づいて選択的ユニタリ作用素 Â_j を計算する。
        Â_j = exp(X_base + gate * delta)
        """
        B = psi_prev.shape[0]
        D = self.dim

        # 基底歪エルミート行列
        X_base = self._build_skew_hermitian(self.A_base_re, self.A_base_im)

        # 状態依存スコア：score = |⟨ψ_prev|delta_raw⟩|^2
        delta_raw = torch.complex(self.delta_proj_re, self.delta_proj_im)  # [D, 1]
        # ⟨ψ|δ⟩ = ψ^H · δ  [B, D] @ [D, 1] -> [B, 1]
        inner_prod = torch.matmul(psi_prev.conj(), delta_raw).squeeze(-1)  # [B]
        score = torch.abs(inner_prod) ** 2

        # ゲート：sigmoid(score - 0.5) → [0, 1]
        gate = torch.sigmoid(score - 0.5)

        # 状態依存補正項 delta (歪エルミート)
        delta_raw_matrix = delta_raw @ delta_raw.conj().T  # [D, D] (PSD, エルミート)
        delta = self._build_skew_hermitian(delta_raw_matrix.real, delta_raw_matrix.imag)

        # X_j = X_base + gate * delta
        X_j = X_base.unsqueeze(0) + gate.view(B, 1, 1) * delta.unsqueeze(0)

        # 行列指数関数でユニタリ行列に
        A_j = torch.matrix_exp(X_j)

        return A_j

    def forward(self, token_ids: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        token_ids: [B, L]
        returns: |ψ₀⟩ [B, D]
        """
        emb = self.token_states(token_ids)
        B, L, D = emb.shape
        device = token_ids.device

        # 入力作用素 B_mat
        B_mat = torch.complex(self.B_re, self.B_im)

        # 状態空間モデルによる逐次的な文脈蓄積
        psi = torch.zeros(B, D, dtype=torch.complex64, device=device)
        psi_states = []

        for j in range(L):
            e_j = emb[:, j, :]

            # 選択的遷移作用素の計算
            A_j = self._compute_selective_unitary(psi)

            # 状態遷移：|ψ_j⟩ = Â_j |ψ_{j-1}⟩ + B̂ |e_j⟩
            psi = torch.einsum('bxy,by->bx', A_j, psi) + torch.einsum('xy,by->bx', B_mat, e_j)
            psi_states.append(psi)

        psi_states = torch.stack(psi_states, dim=1)

        # 位相と注意重み
        pos = torch.arange(L, device=device).view(1, L).expand(B, L)
        phi = self.phase(pos)
        
        scores = torch.einsum('d,bld->bl', self.omega_attn, psi_states.real)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)

        # 重ね合わせ：|ψ₀⟩ = Σ_j α_j e^{iφ_j} |ψ_j⟩
        psi_0 = torch.sum(
            alpha.unsqueeze(-1) * torch.exp(1j * phi.unsqueeze(-1)) * psi_states,
            dim=1
        )

        return normalize_state(psi_0)

    def inject(self, psi: Tensor, token_id: Tensor) -> Tensor:
        """生成時：新トークンを状態に重ねる簡易注入"""
        if token_id.dim() == 0:
            token_id = token_id.view(1)
        emb = self.embeddings[token_id]
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        mixed = psi + 0.5 * emb
        return normalize_state(mixed)

    def verify_unitarity(self, batch_size: int = 4) -> dict:
        """
        Â_j のユニタリ性を検証する。
        戻り値：{max_error, mean_error, is_unitary}
        """
        device = next(self.parameters()).device
        dummy_psi = torch.randn(batch_size, self.dim, dtype=torch.complex64, device=device)
        
        A_j = self._compute_selective_unitary(dummy_psi)
        
        # A_j^† @ A_j が単位行列に近いか確認
        A_j_dagger = A_j.conj().transpose(-2, -1)
        identity = torch.eye(self.dim, dtype=torch.complex64, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        product = torch.einsum('bxy,byz->bxz', A_j_dagger, A_j)
        
        error = torch.abs(product - identity)
        max_error = error.max().item()
        mean_error = error.mean().item()
        
        return {
            "max_error": max_error,
            "mean_error": mean_error,
            "is_unitary": max_error < 1e-4
        }
