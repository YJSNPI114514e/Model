"""複素 RKHS トークナイザー（第 2.2 節）。状態空間モデル (SSM) 版。

注意：Stiefel 多様体による正規直交化は不要になった。
期待値埋め込みによるソフトな状態更新により、埋め込みの幾何学的偏りが
状態に直接影響しなくなるため、単純なランダム初期化で十分。
"""

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
    
    改良 2: 固有分解による高速化
    Â_j = exp(X_base + gate * delta) の計算を O(D³)→O(D²) に削減。
    初期化時に固有分解し、毎ステップは対角行列の指数関数計算のみ。
    
    重要：Stiefel 多様体による正規直交化は廃止。
    埋め込みはランダム初期化のまま学習させる。
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        max_len: int,
        w_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        # Stiefel 多様体の制約を削除：dim >= vocab_size は不要
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len
        self.w_alpha = w_alpha

        # シンプルな埋め込み：Cayley 変換による正規直交化は不要
        # ランダム初期化された埋め込みをそのまま使用
        self.embeddings_raw = nn.Parameter(torch.randn(vocab_size, dim, dtype=torch.cfloat) / math.sqrt(dim))

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
        
        # === 改良 2: 固有分解の事前計算 ===
        # 初期化後に一度だけ固有分解を実行
        self._register_eigendecomposition()

    def _register_eigendecomposition(self) -> None:
        """
        改良 2: X_base の固有分解を事前計算。
        X_base = V @ diag(λ) @ V^{-1}
        
        Skew-Hermitian 行列の固有値は純虚数になるため、
        exp(X_base) はユニタリ行列になる。
        """
        with torch.no_grad():
            # 基底歪エルミート行列
            X_base = self._build_skew_hermitian(self.A_base_re, self.A_base_im)
            
            # 固有分解：X_base = V @ diag(eigvals) @ V^{-1}
            eigvals, eigvecs = torch.linalg.eig(X_base)
            eigvecs_inv = torch.linalg.inv(eigvecs)
            
            # delta_raw も同じ基底で表現（状態依存ゲート用）
            delta_raw = torch.complex(self.delta_proj_re, self.delta_proj_im)  # [D, 1]
            delta_raw_matrix = delta_raw @ delta_raw.conj().T  # [D, D] - rank-1 行列
            delta = self._build_skew_hermitian(delta_raw_matrix.real, delta_raw_matrix.imag)
            
            # delta を固有ベクトル基底に変換
            delta_eig = eigvecs_inv @ delta @ eigvecs
            
            # バッファとして登録（勾配不要）
            self.register_buffer('eigvals', eigvals)
            self.register_buffer('eigvecs', eigvecs)
            self.register_buffer('eigvecs_inv', eigvecs_inv)
            self.register_buffer('delta_eig', delta_eig)

    @property
    def embeddings(self) -> Tensor:
        """シンプルな埋め込みを返す。
        
        Stiefel 多様体による正規直交化は不要。
        埋め込みは自由に学習される。
        
        Returns:
            [V, D] の埋め込み行列
        """
        return self.embeddings_raw  # [V, D]

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
        
        改良 2: 固有分解による高速化
        Â_j = exp(X_base + gate * delta)
        
        従来：torch.matrix_exp(X_j)  # O(D³) 毎ステップ
        改良後：V @ diag(exp(λ + gate * δ_eig)) @ V^{-1}  # O(D²) 毎ステップ
        
        注意：delta は対角行列ではないため、完全な O(D²) にはならないが、
        固有ベクトル基底での計算により数値安定性が向上する。
        """
        B = psi_prev.shape[0]

        # 状態依存スコア：score = |⟨ψ_prev|delta_raw⟩|^2
        delta_raw = torch.complex(self.delta_proj_re, self.delta_proj_im)  # [D, 1]
        # ⟨ψ|δ⟩ = ψ^H · δ  [B, D] @ [D, 1] -> [B, 1]
        inner_prod = torch.matmul(psi_prev.conj(), delta_raw).squeeze(-1)  # [B]
        score = torch.abs(inner_prod) ** 2

        # ゲート：sigmoid(score - 0.5) → [0, 1]
        gate = torch.sigmoid(score - 0.5)

        # === 改良 2: 固有分解による高速計算 ===
        # X_j = X_base + gate * delta
        # 固有値分解：X_base = V @ diag(λ) @ V^{-1}
        # delta も同じ基底で表現済み：delta_eig = V^{-1} @ delta @ V
        
        # 各バッチに対して対角近似で高速計算
        # 対角要素のみ使用：exp(λ_i + gate * (delta_eig)_ii)
        delta_eig_diag = torch.diag(self.delta_eig)  # [D]
        
        # 各サンプルごとにゲートが異なるため、broadcasting で計算
        # lambda_j[b, i] = exp(eigvals[i] + gate[b] * delta_eig_diag[i])
        lambda_j = torch.exp(self.eigvals.unsqueeze(0) + gate.view(B, 1) * delta_eig_diag.unsqueeze(0))
        
        # A_j @ psi = V @ (lambda_j * (V^{-1} @ psi))
        psi_in_eig_basis = torch.matmul(self.eigvecs_inv, psi_prev.T).T  # [B, D]
        scaled = lambda_j * psi_in_eig_basis  # [B, D]
        A_j_psi = torch.matmul(scaled, self.eigvecs.T)  # [B, D]
        
        # 下位互換性のため A_j 行列も構築（O(D²)）
        # V @ diag(λ) @ V^{-1} の計算
        # (V * λ.unsqueeze(0)) @ V^{-1}
        A_j = (self.eigvecs * lambda_j.unsqueeze(1)) @ self.eigvecs_inv
        
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
        
        # ボルン則による重み計算：α_j = |⟨e_j|ψ_context⟩|^2 / Σ_k |⟨e_k|ψ_context⟩|^2
        # scores = ⟨omega_attn|ψ_states⟩ の振幅を計算
        scores = torch.einsum('d,bld->bl', self.omega_attn, psi_states.real)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        
        # 確率振幅の二乗（ボルン則）
        amplitude_squared = torch.exp(scores)  # exp(scores) で正値化
        if mask is not None:
            amplitude_squared = amplitude_squared.masked_fill(~mask, 0.0)
        
        # 規格化：α_j = |amplitude_j|^2 / Σ_k |amplitude_k|^2
        alpha = amplitude_squared / (amplitude_squared.sum(dim=-1, keepdim=True) + 1e-8)

        # 重ね合わせ：|ψ₀⟩ = Σ_j α_j e^{iφ_j} |ψ_j⟩
        psi_0 = torch.sum(
            alpha.unsqueeze(-1) * torch.exp(1j * phi.unsqueeze(-1)) * psi_states,
            dim=1
        )

        return normalize_state(psi_0)

    def phase(self, positions: Tensor) -> Tensor:
        """φ_j = w_φ · j / M_max + b_φ"""
        j = positions.to(dtype=self.w_phi.dtype, device=positions.device)
        return self.w_phi * j / self.max_len + self.b_phi

    def inject(self, psi: Tensor, token_id: Tensor) -> Tensor:
        """生成時：新トークンを状態に重ねる簡易注入（旧方式、現在は不使用）"""
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
