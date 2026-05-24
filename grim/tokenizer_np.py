"""複素 RKHS トークナイザー（第 2.2 節）NumPy 版。状態空間モデル (SSM) 版。

推論・生成専用。逆伝播は含まない。
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray

from grim.geometry_np import complex_inner, normalize_state, softmax


class NumPyTokenizer:
    """
    |ψ₀⟩ = Z^{-1/2} Σ_j α_j e^{iφ_j} · |ψ_j⟩
    ここで |ψ_j⟩ = Â_j |ψ_{j-1}⟩ + B̂_j |e_j⟩
    Â_j: 選択的ユニタリ遷移作用素（入力依存）
    B̂_j: 選択的入力作用素
    
    改良 2: 固有分解による高速化
    Â_j = exp(X_base + gate * delta) の計算を O(D³)→O(D²) に削減。
    初期化時に固有分解し、毎ステップは対角行列の指数関数計算のみ。
    
    Attributes:
        embeddings: 語彙埋め込み (V×D), 複素数
        T_eigvals: 固有値 (D,), 複素数
        T_eigvecs: 固有ベクトル (D×D), 複素数
        T_eigvecs_inv: 逆固有ベクトル (D×D), 複素数
        delta_eig: δの固有表現 (D,), 複素数
        delta_raw: 射影ベクトル (D,), 複素数
        B_mat: 入力作用素 (D×D), 複素数
        phase_weight: 位相重み
        attn_weight: 注意重み (D,)
    """
    
    def __init__(
        self,
        embeddings: ndarray,           # (V, D), 複素数
        T_eigvals: ndarray,            # (D,), 複素数
        T_eigvecs: ndarray,            # (D, D), 複素数
        T_eigvecs_inv: ndarray,        # (D, D), 複素数
        delta_eig: ndarray,            # (D,), 複素数
        delta_raw: ndarray,            # (D,), 複素数
        B_mat: ndarray,                # (D, D), 複素数
        phase_weight: float = 1.0,
        phase_bias: float = 0.0,
        attn_weight: ndarray = None,   # (D,)
        max_len: int = 128,
    ) -> None:
        """NumPy トークナイザー初期化
        
        Args:
            embeddings: 語彙埋め込み (V, D)
            T_eigvals: X_base の固有値 (D,)
            T_eigvecs: X_base の固有ベクトル (D, D)
            T_eigvecs_inv: 逆行列 (D, D)
            delta_eig: δの固有表現 (D,)
            delta_raw: 射影ベクトル (D,)
            B_mat: 入力作用素 (D, D)
            phase_weight: 位相重み w_φ
            phase_bias: 位相バイアス b_φ
            attn_weight: 注意重み ω (D,)
            max_len: 最大系列長
        """
        self.embeddings = embeddings.astype(np.complex128)
        self.T_eigvals = T_eigvals.astype(np.complex128)
        self.T_eigvecs = T_eigvecs.astype(np.complex128)
        self.T_eigvecs_inv = T_eigvecs_inv.astype(np.complex128)
        self.delta_eig = delta_eig.astype(np.complex128)
        self.delta_raw = delta_raw.astype(np.complex128)
        self.B_mat = B_mat.astype(np.complex128)
        self.phase_weight = float(phase_weight)
        self.phase_bias = float(phase_bias)
        self.attn_weight = attn_weight.astype(np.float64) if attn_weight is not None else None
        self.max_len = max_len
        self.vocab_size, self.dim = embeddings.shape
    
    def phase(self, positions: ndarray) -> ndarray:
        """φ_j = w_φ · j / M_max + b_φ
        
        Args:
            positions: 位置インデックス (L,)
            
        Returns:
            位相 (L,)
        """
        j = positions.astype(np.float64)
        return self.phase_weight * j / self.max_len + self.phase_bias
    
    def _compute_selective_unitary(self, psi_prev: ndarray) -> ndarray:
        """状態 ψ_prev に基づいて選択的ユニタリ作用素 Â_j を計算
        
        改良 2: 固有分解による高速化
        Â_j = exp(X_base + gate * delta)
        
        従来：matrix_exp(X_j)  # O(D³) 毎ステップ
        改良後：V @ diag(exp(λ + gate * δ_eig)) @ V^{-1}  # O(D²) 毎ステップ
        
        Args:
            psi_prev: 前一状態 (D,)
            
        Returns:
            ユニタリ行列 A_j (D, D)
        """
        # 状態依存スコア：score = |⟨ψ_prev|delta_raw⟩|^2
        inner_prod = complex_inner(psi_prev, self.delta_raw)  # スカラー
        score = np.abs(inner_prod) ** 2
        
        # ゲート：sigmoid(score - 0.5) → [0, 1]
        gate = 1.0 / (1.0 + np.exp(-(score - 0.5)))
        
        # 固有分解による高速計算
        # λ_j = exp(eigvals + gate * delta_eig)  # (D,)
        lambda_j = np.exp(self.T_eigvals + gate * self.delta_eig)
        
        # A_j = V @ diag(λ_j) @ V^{-1}
        # (V * λ_j) @ V^{-1} の形に最適化
        A_j = (self.T_eigvecs * lambda_j[np.newaxis, :]) @ self.T_eigvecs_inv
        
        return A_j
    
    def tokenize(self, token_ids: ndarray) -> ndarray:
        """トークン列から初期状態 |ψ₀⟩ を構成
        
        数式: ψ₀ = Z^{-1/2} Σⱼ αⱼ exp(iφⱼ) · Âⱼ|eⱼ⟩
        Âⱼ = T_eigvecs @ diag(exp(T_eigvals + gⱼ·delta_eig)) @ T_eigvecs_inv
        gⱼ = σ(|⟨ψⱼ₋₁|δ⟩|² - 0.5)
        
        Args:
            token_ids: トークン ID 列 (L,)
            
        Returns:
            初期状態 ψ₀ (D,), 正規化済み
        """
        L = len(token_ids)
        D = self.dim
        
        # 状態空間モデルによる逐次的な文脈蓄積
        psi = np.zeros(D, dtype=np.complex128)
        psi_states = []
        
        for j in range(L):
            e_j = self.embeddings[token_ids[j]]  # (D,)
            
            # 選択的遷移作用素の計算
            A_j = self._compute_selective_unitary(psi)
            
            # 状態遷移：|ψ_j⟩ = Â_j |ψ_{j-1}⟩ + B̂ |e_j⟩
            psi = A_j @ psi + self.B_mat @ e_j
            psi_states.append(psi.copy())
        
        psi_states = np.array(psi_states)  # (L, D)
        
        # 位相と注意重み
        positions = np.arange(L)
        phi = self.phase(positions)  # (L,)
        
        # 注意スコア計算
        if self.attn_weight is not None:
            scores = np.sum(self.attn_weight * psi_states.real, axis=-1)  # (L,)
        else:
            scores = np.sum(psi_states.real, axis=-1)  # (L,)
        
        alpha = softmax(scores)  # (L,)
        
        # 重ね合わせ：|ψ₀⟩ = Σ_j α_j e^{iφ_j} |ψ_j⟩
        psi_0 = np.sum(
            alpha[:, np.newaxis] * np.exp(1j * phi[:, np.newaxis]) * psi_states,
            axis=0
        )
        
        return normalize_state(psi_0)
    
    def inject(self, psi: ndarray, token_id: int, rate: float = 0.1) -> ndarray:
        """生成時：新トークンを状態に重ねる簡易注入
        
        Args:
            psi: 現在の状態 (D,)
            token_id: 注入するトークン ID
            rate: 注入率
            
        Returns:
            新しい状態 (D,), 正規化済み
        """
        emb = self.embeddings[token_id]
        mixed = psi + rate * emb
        return normalize_state(mixed)
