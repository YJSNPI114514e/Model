"""階層的履歴バッファ NumPy 版。

推論・生成専用。逆伝播は含まない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy import ndarray

from grim.geometry_np import complex_inner, normalize_state


@dataclass
class HistoryEntry:
    """履歴エントリ
    
    Attributes:
        psi: 状態ベクトル (D,)
        weight: 重み
        utility: 有用性スコア（未使用）
        age: 生存ステップ数
    """
    psi: ndarray
    weight: float
    utility: float = 0.0
    age: int = 0


class NumPyHistory:
    """
    階層的履歴バッファ (Short/Mid/Long term memory)
    
    OOM リスク低減のため、最大エントリ数を制限しつつ、
    Fubini-Study 距離に基づく自動クラスタリングで情報を圧縮保持する。
    
    構造:
    - 短期層（short_term）: 最大 10 エントリ。そのまま保持。
    - 中期層（mid_term）  : 最大 50 エントリ。短期から FS 距離で圧縮。
    - 長期層（long_term） : 最大 40 エントリ。中期からさらに圧縮。
    - 合計最大 100 エントリ
    """
    
    def __init__(self, max_entries: int = 100, gamma: float = 0.99) -> None:
        """履歴バッファ初期化
        
        Args:
            max_entries: 最大エントリ数（未使用、下位互換性のため）
            gamma: 重み減衰率
        """
        self.gamma = gamma
        self.max_entries = max_entries
        
        # 各層の最大サイズ
        self.max_short = 10
        self.max_mid = 50
        self.max_long = 40
        
        # 履歴ストレージ
        self.short_term: List[HistoryEntry] = []
        self.mid_term: List[HistoryEntry] = []
        self.long_term: List[HistoryEntry] = []
        
        self._step_counter = 0
    
    def clear(self) -> None:
        """全履歴をクリア"""
        self.short_term.clear()
        self.mid_term.clear()
        self.long_term.clear()
    
    def push(self, psi: ndarray, weight: float = 1.0) -> None:
        """新しい状態を短期メモリに追加し、溢れたら圧縮
        
        Args:
            psi: 現在の状態ベクトル (D,)
            weight: 初期重み
        """
        if psi.ndim == 2:
            psi = psi[0]
        
        psi_detached = psi.astype(np.complex128).copy()
        
        self.short_term.append(HistoryEntry(
            psi=psi_detached,
            weight=weight,
            utility=0.0,
            age=0
        ))
        
        # 短期が溢れたら中期へ圧縮
        if len(self.short_term) > self.max_short:
            self._compress_short_to_mid()
        
        # 中期が溢れたら長期へ移動
        if len(self.mid_term) > self.max_mid:
            self._compress_mid_to_long()
    
    def _compress_short_to_mid(self) -> None:
        """
        短期層の全エントリから、FS 距離が最も近い 2 つを見つけ、
        測地線中点に統合して中期層に追加する。
        """
        if len(self.short_term) < 2:
            return
        
        psis = np.array([e.psi for e in self.short_term])  # (S, D)
        
        # 重叠行列 (Overlap matrix) 計算：|<psi_i, psi_j>|
        overlaps = np.abs(psis @ psis.conj().T)  # (S, S)
        
        # 対角成分（自分自身）を除外
        np.fill_diagonal(overlaps, 0.0)
        
        # 最も近いペアを選択 (overlap が大きい = 距離が小さい)
        max_val = overlaps.max()
        idx_flat = np.argmax(overlaps.reshape(-1))
        
        if max_val == 0.0:
            # 類似度が 0 の場合、単純に先頭 2 つを処理対象とする
            i, j = 0, 1
        else:
            i = idx_flat // overlaps.shape[1]
            j = idx_flat % overlaps.shape[1]
        
        # 測地線中点に統合
        cos_theta = np.clip(max_val, 0.0, 1.0 - 1e-6)
        theta = np.arccos(cos_theta)
        
        # 係数計算：sin(0.5*theta) / sin(theta)
        if theta < 1e-5:
            coeff = 0.5
        else:
            coeff = np.sin(0.5 * theta) / (np.sin(theta) + 1e-8)
        
        psi_i = psis[i]
        psi_j = psis[j]
        
        psi_new = normalize_state(coeff * psi_i + coeff * psi_j)
        weight_new = self.short_term[i].weight + self.short_term[j].weight
        
        # 中期に追加
        self.mid_term.append(HistoryEntry(
            psi=psi_new,
            weight=weight_new,
            utility=0.0,
            age=0
        ))
        
        # 短期から削除 (インデックスが大きい方から削除しないとズレる)
        indices_to_remove = sorted([i, j], reverse=True)
        for idx in indices_to_remove:
            del self.short_term[idx]
    
    def _compress_mid_to_long(self) -> None:
        """
        中期層で最も重みの大きいエントリを長期に移動する。
        長期が溢れたら、最も重みの小さいものを削除する。
        """
        if not self.mid_term:
            return
        
        # 重みでソートし、最大のものを特定
        max_idx = max(range(len(self.mid_term)),
                      key=lambda k: self.mid_term[k].weight)
        
        entry_to_move = self.mid_term.pop(max_idx)
        self.long_term.append(entry_to_move)
        
        # 長期が溢れたら整理
        if len(self.long_term) > self.max_long:
            # 重みが最小のものを削除
            min_idx = min(range(len(self.long_term)),
                          key=lambda k: self.long_term[k].weight)
            del self.long_term[min_idx]
    
    def decay(self, boundary_prob: float = 0.0) -> None:
        """全層の重みを減衰させる
        
        gamma_dynamic = 0.99 - 0.1 * boundary_prob
        
        Args:
            boundary_prob: 文境界確率
        """
        self._step_counter += 1
        
        eta = 0.1
        gamma_dynamic = 0.99 - eta * boundary_prob
        gamma_dynamic = max(gamma_dynamic, 0.5)  # 下限設定
        
        for layer in [self.short_term, self.mid_term, self.long_term]:
            for entry in layer:
                entry.age += 1
                entry.weight *= gamma_dynamic
            
            # 重みが閾値以下のエントリを削除
            # ただし age < 10 のものは保護
            layer[:] = [e for e in layer if e.weight >= 1e-6 or e.age < 10]
    
    def summarize(self, psi_current: Optional[ndarray] = None) -> ndarray:
        """現在の状態に関連する履歴を全層から検索し、重み付き平均
        
        全エントリ中、|⟨ψ_cur|ψ_h⟩|² > 0.01 のものだけ重み付き平均
        
        Args:
            psi_current: 現在の状態 (D,)（関連度計算に使用）
            
        Returns:
            要約されたコンテキスト (D,)
        """
        if not (self.short_term or self.mid_term or self.long_term):
            return np.zeros(1, dtype=np.complex128)  # ダミー
        
        # 入力の正規化
        if psi_current is not None:
            if psi_current.ndim == 2:
                psi_ref = psi_current[0]
            else:
                psi_ref = psi_current.astype(np.complex128)
        else:
            # current_psi が提供されない場合は、短期メモリの最新を使用
            if self.short_term:
                psi_ref = self.short_term[-1].psi
            else:
                psi_ref = None
        
        all_entries = []
        all_weights = []
        
        # 全層を走査
        for layer in [self.short_term, self.mid_term, self.long_term]:
            for entry in layer:
                if psi_ref is not None:
                    # 現在の状態との関連度 (Relevance) 計算
                    overlap = np.abs(complex_inner(psi_ref, entry.psi))
                    relevance = overlap ** 2
                    
                    # 関連度閾値フィルタ
                    if relevance > 0.01:
                        all_entries.append(entry.psi)
                        all_weights.append(entry.weight * relevance)
                else:
                    # 関連度計算しない場合は全て使用
                    all_entries.append(entry.psi)
                    all_weights.append(entry.weight)
        
        if not all_entries:
            # ダミー返却
            if psi_ref is not None:
                return np.zeros_like(psi_ref)
            else:
                return np.zeros(1, dtype=np.complex128)
        
        psis = np.array(all_entries)  # (N, D)
        weights = np.array(all_weights, dtype=np.float64)
        
        # Softmax 正規化
        weights_norm = np.exp(weights - weights.max())
        weights_norm /= weights_norm.sum()
        
        # 重み付き和
        context_vec = np.sum(weights_norm[:, np.newaxis] * psis, axis=0)
        
        return context_vec
    
    def __len__(self) -> int:
        return len(self.short_term) + len(self.mid_term) + len(self.long_term)
    
    def get_stats(self) -> dict:
        """統計情報の取得"""
        return {
            'short': len(self.short_term),
            'mid': len(self.mid_term),
            'long': len(self.long_term),
            'total': len(self)
        }
