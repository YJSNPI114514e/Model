"""フビニ・スタディ計量・測地線・接空間射影（第 2.1–2.4 節）NumPy 版。

推論・生成専用。逆伝播は含まない。
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray


def complex_inner(a: ndarray, b: ndarray, axis: int = -1) -> ndarray:
    """⟨a|b⟩ = Σ conj(a) * b
    
    Args:
        a: 複素数配列 (..., D)
        b: 複素数配列 (..., D)
        axis: 内積を取る軸
        
    Returns:
        内積 (...,) スカラーまたは配列
    """
    return np.sum(np.conj(a) * b, axis=axis)


def normalize_state(psi: ndarray, eps: float = 1e-8) -> ndarray:
    """‖|ψ⟩‖ = 1 に正規化
    
    Args:
        psi: 複素数状態ベクトル (..., D)
        eps: ゼロ除算防止
        
    Returns:
        正規化済み状態 (..., D), ‖psi‖ = 1
    """
    n = np.linalg.norm(psi, axis=-1, keepdims=True).clip(min=eps)
    return psi / n


def tangent_project(v: ndarray, psi: ndarray) -> ndarray:
    """Π_tangent(v) = v - ⟨ψ|v⟩|ψ⟩（複素内積）
    
    接空間への直交射影。結果は常に ⟨ψ|v_tangent⟩ = 0 を満たす。
    
    Args:
        v: 射影するベクトル (..., D)
        psi: 基準状態 (..., D), ‖psi‖=1
        
    Returns:
        接空間成分 (..., D)
    """
    radial = complex_inner(psi, v, axis=-1)[..., np.newaxis] * psi
    return v - radial


def fs_distance(psi: ndarray, phi: ndarray, eps: float = 1e-8) -> ndarray:
    """Fubini-Study 距離: d_FS(ψ, φ) = arccos|⟨ψ|φ⟩|
    
    Args:
        psi: 状態ベクトル (..., D)
        phi: 状態ベクトル (..., D)
        eps: arccos の定義域制限用
        
    Returns:
        FS 距離 (...,), 範囲 [0, π/2]
    """
    overlap = np.abs(complex_inner(psi, phi, axis=-1)).clip(0.0, 1.0 - eps)
    return np.arccos(overlap)


def softmax(x: ndarray, axis: int = -1) -> ndarray:
    """ソフトマックス関数: exp(x) / Σexp(x)
    
    Args:
        x: 実数配列
        axis: 正規化する軸
        
    Returns:
        ソフトマックス出力, Σoutput = 1
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # 数値安定性
    return e_x / e_x.sum(axis=axis, keepdims=True)
