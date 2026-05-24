"""K=3 メタ学習層（カーネルリッジ回帰版）。

sekkeisyo.txt に基づく GRIM の K=3 層メタ学習を、
カーネルリッジ回帰（Kernel Ridge Regression）で実装。
逆伝播・ODE 逆伝播は不要。解析解でλ, βを更新。
"""

from __future__ import annotations

import numpy as np


def complex_inner(a: np.ndarray, b: np.ndarray) -> complex:
    """複素ベクトルの内積 ⟨a|b⟩ = Σ conj(a_i) * b_i"""
    return np.vdot(a, b)


class K3MetaLearner:
    """K=3 メタ学習器（カーネルリッジ回帰ベース）。
    
    理論式:
        (λ_new, β_new) = argmin F(λ, β)
        F(λ, β) = E_ψ[L_val(ψ_T; λ, β)] + γ·KL((λ,β) || (λ_0,β_0))
    
    カーネルリッジ回帰:
        K_ij = |⟨ψ_T^(i)|ψ_T^(j)⟩|^2  （グラム行列）
        α = (K + γ·I)^(-1) · y      （解析解）
        y_i = -log P(y_true|ψ_T^(i))  （各サンプルの検証損失）
    
    Attributes:
        gamma: KL 正則化強度（γ）。大きいほど現在値に近づく。
        psi_buffer: 過去のψ_T を保存するバッファ。
        loss_buffer: 対応する検証損失のバッファ。
        lam_buffer: 過去のλ値のバッファ。
        beta_buffer: 過去のβ値のバッファ。
        max_buffer_size: バッファの最大サイズ（デフォルト 30）。
    """
    
    def __init__(
        self,
        gamma: float = 0.01,
        max_buffer_size: int = 30,
        update_interval: int = 3,
        smoothing_factor: float = 0.3,
    ):
        """初期化。
        
        Args:
            gamma: KL 正則化強度。γ=0.01 がデフォルト。
            max_buffer_size: 蓄積するサンプル数の上限。
            update_interval: 更新間隔（エポック数）。デフォルト 3。
            smoothing_factor: 移動平均の係数。新値への寄与率（0.3=30%）。
        """
        self.gamma = gamma
        self.max_buffer_size = max_buffer_size
        self.update_interval = update_interval
        self.smoothing_factor = smoothing_factor
        
        # バッファ
        self.psi_buffer: list[np.ndarray] = []
        self.loss_buffer: list[float] = []
        self.lam_buffer: list[float] = []
        self.beta_buffer: list[float] = []
    
    def accumulate(
        self,
        psi_T: np.ndarray,
        val_loss: float,
        lam: float,
        beta: float,
    ) -> None:
        """3 エポック分のψ_T と損失、メタパラメータを蓄積。
        
        Args:
            psi_T: 最終状態ψ_T（複素ベクトル）。
            val_loss: 検証損失（小さいほど良い）。
            lam: 現在のλ値。
            beta: 現在のβ値。
        """
        # バッファが満杯の場合は古いものを削除
        if len(self.psi_buffer) >= self.max_buffer_size:
            self.psi_buffer.pop(0)
            self.loss_buffer.pop(0)
            self.lam_buffer.pop(0)
            self.beta_buffer.pop(0)
        
        # データを蓄積
        self.psi_buffer.append(psi_T.copy())
        self.loss_buffer.append(val_loss)
        self.lam_buffer.append(lam)
        self.beta_buffer.append(beta)
    
    def _compute_gram_matrix(self) -> np.ndarray:
        """グラム行列 K_ij = |⟨ψ_T^(i)|ψ_T^(j)⟩|^2 を計算。
        
        Returns:
            N×N のグラム行列（N=蓄積サンプル数）。
        """
        N = len(self.psi_buffer)
        K = np.zeros((N, N), dtype=np.float64)
        
        for i in range(N):
            for j in range(i, N):
                # 複素内積の絶対値の二乗
                overlap = complex_inner(self.psi_buffer[i], self.psi_buffer[j])
                K[i, j] = np.abs(overlap) ** 2
                if i != j:
                    K[j, i] = K[i, j]  # 対称行列
        
        return K
    
    def update(
        self,
        lam: float,
        beta: float,
    ) -> tuple[float, float, np.ndarray | None]:
        """蓄積データからλ, βの解析的最適値を計算。
        
        カーネルリッジ回帰の解析解:
            α = (K + γ·I)^(-1) · y
        
        新しいλ, βは、損失が小さかった（良かった）エポックの
        λ, βに重み付けして近づく。
        
        Args:
            lam: 現在のλ値。
            beta: 現在のβ値。
        
        Returns:
            (lam_new, beta_new, weights) のタプル。
            weights は各サンプルの重み（N 次元配列）。
            データ不足の場合は元の値を返す。
        """
        N = len(self.psi_buffer)
        
        # データ不足の場合は更新しない
        if N < 10:
            return lam, beta, None
        
        # グラム行列の計算
        K = self._compute_gram_matrix()
        
        # 目的変数：検証損失（負の対数尤度、小さいほど良い）
        y = np.array(self.loss_buffer, dtype=np.float64)
        
        # カーネルリッジ回帰の解析解
        # α = (K + γ·I)^(-1) · y
        try:
            alpha = np.linalg.solve(
                K + self.gamma * np.eye(N),
                y
            )
        except np.linalg.LinAlgError:
            # 特異な場合は疑似逆行列を使用
            alpha = np.linalg.lstsq(
                K + self.gamma * np.eye(N),
                y,
                rcond=None
            )[0]
        
        # 重みの計算：損失が小さいほど重み大
        # softmax(-y) で、損失が小さいサンプルに高い重み
        weights = np.exp(-y) / np.sum(np.exp(-y))
        
        # 加重平均で新しいλ, βを計算
        # 損失が小さかったエポックのメタパラメータに近づける
        lam_weighted = np.sum(weights * np.array(self.lam_buffer))
        beta_weighted = np.sum(weights * np.array(self.beta_buffer))
        
        # カーネルリッジ回帰の結果も考慮（αの重みで）
        # αが小さい（予測損失が小さい）サンプルの影響を大きく
        alpha_weights = np.exp(-alpha) / np.sum(np.exp(-alpha))
        lam_krr = np.sum(alpha_weights * np.array(self.lam_buffer))
        beta_krr = np.sum(alpha_weights * np.array(self.beta_buffer))
        
        # 両者を組み合わせる（単純平均）
        lam_new = 0.5 * lam_weighted + 0.5 * lam_krr
        beta_new = 0.5 * beta_weighted + 0.5 * beta_krr
        
        # 安全のため、極端な値にならないようにクリップ
        lam_new = np.clip(lam_new, 1e-6, 1.0)
        beta_new = np.clip(beta_new, 1e-6, 1.0)
        
        # バッファクリア
        self.clear()
        
        return lam_new, beta_new, weights
    
    def clear(self) -> None:
        """バッファをクリア。"""
        self.psi_buffer = []
        self.loss_buffer = []
        self.lam_buffer = []
        self.beta_buffer = []
    
    def should_update(self, epoch: int) -> bool:
        """指定エポックで更新すべきか判定。
        
        Args:
            epoch: 現在のエポック番号（1 から開始）。
        
        Returns:
            更新すべき場合は True。
        """
        return (epoch % self.update_interval == 0) and (epoch > 0)
    
    def smooth_update(
        self,
        lam: float,
        beta: float,
        lam_new: float,
        beta_new: float,
    ) -> tuple[float, float]:
        """急激な変化を防ぐための移動平均。
        
        Args:
            lam: 現在のλ値。
            beta: 現在のβ値。
            lam_new: 新しく計算されたλ値。
            beta_new: 新しく計算されたβ値。
        
        Returns:
            (lam_smoothed, beta_smoothed) のタプル。
        """
        sf = self.smoothing_factor
        lam_smoothed = (1.0 - sf) * lam + sf * lam_new
        beta_smoothed = (1.0 - sf) * beta + sf * beta_new
        return lam_smoothed, beta_smoothed
