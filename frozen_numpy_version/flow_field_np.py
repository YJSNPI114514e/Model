"""エネルギー関数と勾配 NumPy 版。

推論・生成専用。逆伝播は含まない。
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray

from grim.geometry_np import complex_inner, tangent_project, fs_distance


class NumPyEnergyField:
    """
    エネルギー関数 E(ψ) = E_world + E_self + E_cont + E_explore
    とその勾配を計算する。
    
    Attributes:
        embeddings: 語彙埋め込み (V×D), 複素数
        lam: 自己無撞着性重み
        mu: 連続性重み
        sigma: カーネル幅
        beta: 探索重み
    """
    
    def __init__(
        self,
        embeddings: ndarray,  # (V, D), 複素数
        lam: float = 0.01,
        mu: float = 0.01,
        sigma: float = 0.693,
        beta: float = 0.01,
    ) -> None:
        """エネルギー場初期化
        
        Args:
            embeddings: 語彙埋め込み (V, D)
            lam: E_self の重み（正の実数）
            mu: E_cont の重み（正の実数）
            sigma: E_cont のカーネル幅（正の実数）
            beta: E_explore の重み（正の実数）
        """
        self.embeddings = embeddings.astype(np.complex128)
        self.lam = float(lam)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.beta = float(beta)
        self.vocab_size, self.dim = embeddings.shape
    
    def _born_probs(self, psi: ndarray) -> ndarray:
        """ボルン則による確率計算
        
        p_k = |⟨e_k|ψ⟩|² / Σ_j |⟨e_j|ψ⟩|²
        
        Args:
            psi: 状態ベクトル (D,)
            
        Returns:
            確率分布 (V,), Σp = 1
        """
        # 正規化済み埋め込みとの内積
        emb_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=-1, keepdims=True).clip(min=1e-8)
        overlaps = complex_inner(emb_norm, psi)  # (V,)
        scores = np.abs(overlaps) ** 2
        return scores / scores.sum().clip(min=1e-8)
    
    def energy_and_grad(
        self,
        psi: ndarray,       # (D,), 現在の状態
        psi_0: ndarray,     # (D,), 初期状態
        history: list,      # HistoryEntry のリスト
    ) -> tuple[ndarray, ndarray]:
        """エネルギーとその解析的勾配を計算
        
        E_world = -log|⟨ψ|ψ₀⟩|²
        E_self  = λ‖ψ‖²
        E_cont  = -(μ/|H|) Σ w_h exp(-d_FS²/(2σ²))
        E_explore = -β Σ p_k log p_k
        
        grad_E = ∂E/∂ψ*
        
        Args:
            psi: 現在の状態 (D,)
            psi_0: 初期状態 (D,)
            history: 履歴エントリリスト
            
        Returns:
            E: スカラーエネルギー
            grad_E: 勾配 (D,)
        """
        D = self.dim
        eps_w = 1e-8
        
        # --- 1. E_world: 外部入力との整合性 ---
        overlap = complex_inner(psi, psi_0)
        overlap_sq = np.abs(overlap) ** 2
        e_world = -np.log(overlap_sq.clip(min=eps_w))
        
        # 勾配: d/dψ* (-log(|⟨ψ|ψ₀⟩|²)) = -ψ₀ / ⟨ψ|ψ₀⟩
        grad_world = (-np.conj(overlap) / overlap_sq.clip(min=eps_w)) * psi_0
        
        # --- 2. E_self: 自己無撞着性 ---
        e_norm = np.sum(np.abs(psi) ** 2)
        e_self = self.lam * e_norm
        
        # 勾配: d/dψ* (λ|ψ|²) = λψ
        grad_self = self.lam * psi
        
        # --- 3. E_cont: 連続性 ---
        if not history:
            e_cont = 0.0
            grad_cont = np.zeros(D, dtype=np.complex128)
        else:
            psis = np.array([e.psi for e in history])  # (H_len, D)
            weights = np.array([e.weight for e in history])  # (H_len,)
            
            # overlaps: (H_len,)
            overlaps = complex_inner(psi, psis, axis=-1)
            abs_overlap = np.abs(overlaps).clip(0.0, 1.0)
            
            # FS 距離の近似: d ≈ sqrt(2 - 2*F)
            dist_approx = np.sqrt(np.clip(2.0 - 2.0 * abs_overlap, 1e-9, None))
            
            # ガウスカーネル
            kernel = np.exp(-(dist_approx ** 2) / (2 * self.sigma ** 2))
            
            e_cont = -(self.mu / len(history)) * np.sum(weights * kernel)
            
            # 勾配計算 (手動微分)
            factor = (self.mu / len(history)) * weights * kernel / (2 * self.sigma ** 2)
            phase = np.conj(overlaps) / abs_overlap.clip(min=1e-9)
            
            grad_cont = -np.sum((factor * phase)[:, np.newaxis] * psis, axis=0)
        
        # --- 4. E_explore: 探索 ---
        probs = self._born_probs(psi)
        eps_e = 1e-6
        log_probs = np.log(probs.clip(min=eps_e))
        e_explore = -self.beta * np.sum(probs * log_probs)
        
        # 勾配: 簡易近似
        mean_log = np.sum(probs * (log_probs + 1))
        grad_factor = log_probs + 1 - mean_log
        grad_explore = -self.beta * psi * grad_factor.mean()
        
        # --- 総合 ---
        E = e_world + e_self + e_cont + e_explore
        grad_E = grad_world + grad_self + grad_cont + grad_explore
        
        return E, grad_E
    
    def ode_rhs(
        self,
        t: float,
        psi: ndarray,
        psi_0: ndarray,
        history: list,
    ) -> ndarray:
        """ODE の右辺：v = -grad_E, 接空間へ射影
        
        dψ/dt = v = -∇E
        
        Args:
            t: 時間（未使用だが ODE ソルバーのために必要）
            psi: 現在の状態 (D,)
            psi_0: 初期状態 (D,)
            history: 履歴エントリリスト
            
        Returns:
            速度ベクトル (D,), 接空間成分のみ
        """
        _, grad_E = self.energy_and_grad(psi, psi_0, history)
        v = -grad_E
        return tangent_project(v, psi)
