"""ODE 積分 NumPy 版。scipy.integrate.solve_ivp を使用。

推論・生成専用。逆伝播は含まない。
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray
from scipy.integrate import solve_ivp

from grim.geometry_np import normalize_state


def integrate_flow(
    ode_rhs,          # callable: (t, psi) -> dpsi/dt
    psi_0: ndarray,   # (D,), 初期状態
    psi_0_fixed: ndarray,  # (D,), 固定初期状態（エネルギー計算用）
    history: list,    # 履歴エントリリスト
    params: dict,     # パラメータ辞書
    embeddings: ndarray = None,  # (V, D), 埋め込み（オプション）
    T: float = 1.0,   # 積分時間
) -> ndarray:
    """フローに沿って状態を積分
    
    solve_ivp(ode_rhs, [0,T], psi_0, method='DOP853',
              rtol=1e-4, atol=1e-6, max_step=0.1)
    
    Args:
        ode_rhs: ODE の右辺関数 (t, psi, psi_0_fixed, history) -> dpsi/dt
        psi_0: 初期状態 (D,)
        psi_0_fixed: 固定初期状態（エネルギー計算用）(D,)
        history: 履歴エントリリスト
        params: パラメータ辞書（lam, mu, sigma, beta など）
        embeddings: 語彙埋め込み (V, D)（オプション）
        T: 積分時間
        
    Returns:
        ψ_T: 積分後の状態 (D,), 正規化済み
    """
    psi_0 = normalize_state(psi_0.astype(np.complex128))
    D = len(psi_0)
    
    # 複素数を実数・虚部に分解して solve_ivp に渡す
    def dynamics(t, y_real):
        """実数表現でのダイナミクス
        
        Args:
            t: 時間
            y_real: [Re(ψ), Im(ψ)] (2D,)
            
        Returns:
            dy/dt (2D,)
        """
        psi = y_real[:D] + 1j * y_real[D:]
        psi = normalize_state(psi)
        
        # ODE 右辺を計算
        if embeddings is not None and 'embeddings' in params:
            dpsi = ode_rhs(t, psi, psi_0_fixed, history, params)
        else:
            dpsi = ode_rhs(t, psi, psi_0_fixed, history)
        
        # 接空間射影（ode_rhs 内で実施済みの場合は不要だが、念のため）
        # dpsi = tangent_project(dpsi, psi)
        
        # 実数・虚部に分解
        return np.concatenate([dpsi.real, dpsi.imag])
    
    # 初期状態を実数表現に変換
    y0 = np.concatenate([psi_0.real, psi_0.imag])
    
    # ODE 積分
    result = solve_ivp(
        dynamics,
        [0.0, T],
        y0,
        method='DOP853',
        rtol=1e-4,
        atol=1e-6,
        max_step=0.1,
    )
    
    # 最終状態を復元
    y_T = result.y[:, -1]
    psi_T = y_T[:D] + 1j * y_T[D:]
    
    # 正規化
    psi_T = normalize_state(psi_T)
    
    return psi_T
