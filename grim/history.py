"""動的履歴バッファ H（第 2.5 節）。
階層的圧縮版：短期・中期・長期の 3 層構造で OOM リスク低減。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class HistoryEntry:
    psi: Tensor
    weight: float
    utility: float = 0.0  # 有用性スコア（下位互換性のため）
    age: int = 0  # 生存ステップ数（下位互換性のため）


class HistoryEmbedder(nn.Module):
    """κ_emb: M → R^{D_h}"""

    def __init__(self, dim: int, history_dim: int) -> None:
        super().__init__()
        self.proj_re = nn.Linear(dim * 2, history_dim)
        self.proj_im = nn.Linear(dim * 2, history_dim)

    def forward(self, psi: Tensor) -> Tensor:
        feat = torch.cat([psi.real, psi.imag], dim=-1)
        return self.proj_re(feat) + self.proj_im(feat)


def complex_inner(a: Tensor, b: Tensor) -> Tensor:
    """複素数ベクトルの内積 <a, b> = sum(a * conj(b))"""
    return torch.sum(a * torch.conj(b), dim=-1)


def normalize_state(psi: Tensor, dim: int = -1) -> Tensor:
    """状態ベクトルのノルムを 1 に正規化"""
    norm = torch.norm(psi, dim=dim, keepdim=True)
    return psi / (norm + 1e-8)


class HierarchicalHistoryBuffer:
    """
    階層的履歴バッファ (Short/Mid/Long term memory)
    
    OOM リスク低減のため、最大エントリ数を制限しつつ、
    Fubini-Study 距離に基づく自動クラスタリングで情報を圧縮保持する。
    
    構造:
    - 短期層（short_term）: 最大 10 エントリ。そのまま保持。
    - 中期層（mid_term）  : 最大 50 エントリ。短期から FS 距離で圧縮。
    - 長期層（long_term） : 最大 40 エントリ。中期からさらに圧縮。
    - 合計最大 100 エントリ（OOM 対策の N_max=100 と一致）。
    """
    
    def __init__(
        self,
        n_max: int,
        gamma: float,
        eps: float,
        embedder: HistoryEmbedder,
        device: torch.device,
    ) -> None:
        self.gamma = gamma
        self.eps = eps
        self.embedder = embedder
        self.device = device
        
        # 各層の最大サイズ
        self.max_short = 10
        self.max_mid = 50
        self.max_long = 40
        # 合計最大 100 エントリ
        
        # 履歴ストレージ
        self.short_term: List[HistoryEntry] = []
        self.mid_term: List[HistoryEntry] = []
        self.long_term: List[HistoryEntry] = []
        
        # 下位互換性のための属性
        self.n_max = n_max
        self._step_counter = 0
        
    def clear(self) -> None:
        """全履歴をクリア"""
        self.short_term.clear()
        self.mid_term.clear()
        self.long_term.clear()

    def push(self, psi: Tensor, weight: float = 1.0) -> None:
        """
        新しい状態を短期メモリに追加し、溢れたら圧縮を行う。
        
        Args:
            psi: 現在の状態ベクトル [B, D] または [D]
            weight: 初期重み
        """
        # バッチ次元がある場合は代表値（先頭）を取る
        if psi.dim() == 2:
            psi = psi[0]
        
        psi_detached = psi.detach().clone()
        
        self.short_term.append(HistoryEntry(
            psi=psi_detached, 
            weight=weight, 
            utility=0.0, 
            age=0
        ))
        
        # 短期が溢れたら中期へ圧縮（再帰的に中期もチェック）
        while len(self.short_term) > self.max_short:
            self._compress_short_to_mid()
            # 中期が溢れたら長期へ移動
            if len(self.mid_term) > self.max_mid:
                self._compress_mid_to_long()

    def _compress_short_to_mid(self) -> None:
        """
        短期層の全エントリから、FS 距離が最も近い 2 つを見つけ、
        測地線中点に統合して中期層に追加する。
        短期が max_short 以下になるまで繰り返し圧縮する。
        """
        # 短期が max_short 以下になるまで繰り返す
        while len(self.short_term) > self.max_short and len(self.short_term) >= 2:
            self._compress_one_pair()
    
    def _compress_one_pair(self) -> None:
        """短期層から最も近い 1 ペアを圧縮して中期に追加"""
        if len(self.short_term) < 2:
            return

        psis = torch.stack([e.psi for e in self.short_term])  # [S, D]
        
        # 重叠行列 (Overlap matrix) 計算：|<psi_i, psi_j>|
        overlaps = torch.abs(psis @ psis.conj().T)  # [S, S]
        
        # 対角成分（自分自身）を除外して最小値を設定
        mask = torch.ones_like(overlaps, dtype=torch.bool)
        mask.fill_diagonal_(False)
        overlaps.masked_fill_(~mask, 0.0)
        
        # 最も近いペアを選択 (overlap が大きい = 距離が小さい)
        max_val = overlaps.max()
        idx_flat = torch.argmax(overlaps.reshape(-1))
        
        if max_val == 0.0:
            # 類似度が 0 の場合、単純に先頭 2 つを処理対象とする
            i, j = 0, 1
        else:
            # unravel_index の互換性対応 (PyTorch 2.0 未満対策)
            try:
                i, j = torch.unravel_index(idx_flat, overlaps.shape)
            except AttributeError:
                # PyTorch < 2.0 fallback
                i = idx_flat // overlaps.shape[1]
                j = idx_flat % overlaps.shape[1]
        
        # 測地線中点に統合
        cos_theta = max_val.clamp(0.0, 1.0 - 1e-6)
        theta = torch.acos(cos_theta)
        
        # 係数計算：sin(0.5*theta) / sin(theta)
        if theta < 1e-5:
            coeff = 0.5
        else:
            coeff = torch.sin(0.5 * theta) / (torch.sin(theta) + 1e-8)
        
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

    def summarize(self, batch_size: int = 1, current_psi: Optional[Tensor] = None) -> Tensor:
        """
        現在の状態に関連する履歴を全層から検索し、重み付き平均を返す。
        
        Args:
            batch_size: 出力のバッチサイズ
            current_psi: 現在の状態 [B, D] または [D]（関連度計算に使用）
            
        Returns:
            summarized_context: [B, D_h]
        """
        if not (self.short_term or self.mid_term or self.long_term):
            return torch.zeros(batch_size, self.embedder.proj_re.out_features, device=self.device)

        # 入力の正規化
        if current_psi is not None:
            if current_psi.dim() == 2:
                psi_ref = current_psi[0]
            else:
                psi_ref = current_psi
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
                    overlap = torch.abs(complex_inner(psi_ref, entry.psi))
                    relevance = overlap ** 2
                    
                    # 関連度閾値フィルタ
                    if relevance > 0.01:
                        all_entries.append(entry.psi)
                        all_weights.append(entry.weight * relevance.item())
                else:
                    # 関連度計算しない場合は全て使用
                    all_entries.append(entry.psi)
                    all_weights.append(entry.weight)
        
        if not all_entries:
            return torch.zeros(batch_size, self.embedder.proj_re.out_features, device=self.device)
        
        psis = torch.stack(all_entries)  # [N, D]
        weights = torch.tensor(all_weights, device=self.device, dtype=torch.float32)  # float32 で計算
        
        # ボルン則による重み付け: w_j^norm = |⟨ψ_current|ψ_j⟩|^2 * w_j / Σ_k |⟨ψ_current|ψ_k⟩|^2 * w_k
        # all_weights は既に relevance^2 * weight を含んでいるため、これで規格化
        weights_norm = weights / (weights.sum() + 1e-8)
        
        # 埋め込み変換
        embs = self.embedder(psis)  # [N, D_h]
        
        # 重み付き和
        context_vec = torch.sum(weights_norm.unsqueeze(-1) * embs, dim=0)  # [D_h]
        
        # バッチ次元に展開
        return context_vec.unsqueeze(0).expand(batch_size, -1)

    def decay(self, boundary_prob: float = 0.0) -> None:
        """
        全層の重みを減衰させる。
        文境界確率が高い場合は追加で減衰させる。
        """
        self._step_counter += 1
        
        eta = 0.1
        gamma = 0.99 - eta * boundary_prob
        gamma = max(gamma, 0.5)  # 減衰率が極端にならないよう下限設定
        
        for layer in [self.short_term, self.mid_term, self.long_term]:
            for entry in layer:
                entry.age += 1
                entry.weight *= gamma
            
            # 重みが閾値以下のエントリを削除
            # ただし age < 10 のものは保護（下位互換性）
            layer[:] = [e for e in layer if e.weight >= self.eps or e.age < 10]

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


# 下位互換性：既存の HistoryBuffer という名前でアクセス可能に
HistoryBuffer = HierarchicalHistoryBuffer
