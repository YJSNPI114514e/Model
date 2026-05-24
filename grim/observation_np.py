"""観測・ボルン則 NumPy 版。

推論・生成専用。逆伝播は含まない。
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray

from grim.geometry_np import normalize_state


def born_probs(psi: ndarray, embeddings: ndarray) -> ndarray:
    """ボルン則による確率計算
    
    scores = |embeddings @ conj(psi)|²
    probs = scores / Σscores
    
    Args:
        psi: 状態ベクトル (D,), 正規化済み
        embeddings: 語彙埋め込み (V, D), 複素数
        
    Returns:
        確率分布 (V,), Σprobs = 1
    """
    psi = psi.astype(np.complex128)
    embeddings = embeddings.astype(np.complex128)
    
    # 正規化済み埋め込み
    emb_norm = normalize_state(embeddings)  # (V, D), 各行が単位ベクトル
    
    # 内積：⟨e_k|ψ⟩ = conj(e_k) @ psi
    # embeddings @ conj(psi) の形に注意
    overlaps = embeddings @ np.conj(psi)  # (V,)
    
    # ボルン則：p_k = |⟨e_k|ψ⟩|²
    scores = np.abs(overlaps) ** 2
    
    # ソフトマックス正規化
    probs = scores / scores.sum().clip(min=1e-8)
    
    return probs
