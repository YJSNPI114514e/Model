"""動的履歴バッファ H（第2.5節）。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class HistoryEntry:
    psi: Tensor
    weight: float
    utility: float  # 改良 3: 有用性スコア
    age: int  # 改良 3: 生存ステップ数


class HistoryEmbedder(nn.Module):
    """κ_emb: M → R^{D_h}"""

    def __init__(self, dim: int, history_dim: int) -> None:
        super().__init__()
        self.proj_re = nn.Linear(dim * 2, history_dim)
        self.proj_im = nn.Linear(dim * 2, history_dim)

    def forward(self, psi: Tensor) -> Tensor:
        feat = torch.cat([psi.real, psi.imag], dim=-1)
        return self.proj_re(feat) + self.proj_im(feat)


class HistoryBuffer:
    """
    H = { (|ψ^{(i)}⟩, w_i, u_i, age_i) }
    書込・減衰・容量制限（循環バッファ）
    
    改良 3: 有用性ベース管理
    - 各エントリに有用性スコア utility を付与
    - スコアが低いものから優先削除
    - utility > 0.01 → 減衰率 0.999（長く保持）
    - utility < 0.00 → 減衰率 0.95（早く忘れる）
    - 最低でも 10 ステップは生存させる（評価期間確保）
    """

    def __init__(
        self,
        n_max: int,
        gamma: float,
        eps: float,
        embedder: HistoryEmbedder,
        device: torch.device,
    ) -> None:
        self.n_max = n_max
        self.gamma = gamma
        self.eps = eps
        self.embedder = embedder
        self.device = device
        self._entries: deque[HistoryEntry] = deque(maxlen=n_max)
        
        # 改良 3: 有用性更新用
        self._step_counter = 0
        self._utility_update_interval = 10  # 10 ステップに 1 回更新
        self._min_survival_steps = 10  # 最低生存ステップ数

    def clear(self) -> None:
        self._entries.clear()

    def _compute_utility(self, entry: HistoryEntry, current_psi: Tensor | None = None) -> float:
        """
        改良 3: 有用性スコアの計算（簡易版）
        
        厳密には P(target|with_history) - P(target|without) だが、
        計算コストが高いため、以下の簡易指標を使用：
        - エントリの重みが大きいほど有用
        - 最近使われたほど有用
        - 年齢が若いほど未知の可能性
        """
        if current_psi is None:
            # 簡易スコア：weight と age の組み合わせ
            base_utility = entry.weight * (1.0 / (1.0 + entry.age * 0.1))
            return base_utility
        
        # ψとの類似度を追加（オプション）
        similarity = torch.abs(torch.vdot(entry.psi.squeeze(0), current_psi)).real.item()
        base_utility = entry.weight * similarity * (1.0 / (1.0 + entry.age * 0.1))
        return base_utility

    def decay(self) -> None:
        """
        改良 3: 有用性に基づいた減衰
        - utility > 0.01 → 減衰率 0.999（長く保持）
        - utility < 0.00 → 減衰率 0.95（早く忘れる）
        """
        self._step_counter += 1
        
        # 10 ステップに 1 回、有用性を更新
        if self._step_counter % self._utility_update_interval == 0:
            for entry in self._entries:
                entry.utility = self._compute_utility(entry)
        
        kept: deque[HistoryEntry] = deque(maxlen=self.n_max)
        for e in self._entries:
            # 年齢を更新
            e.age += 1
            
            # 有用性に基づく減衰率の調整
            if e.utility > 0.01:
                decay_rate = 0.999  # 長く保持
            elif e.utility < 0.0:
                decay_rate = 0.95  # 早く忘れる
            else:
                decay_rate = self.gamma  # デフォルト
            
            w = e.weight * decay_rate
            # 最低生存ステップ数は経過するまで削除しない
            if w >= self.eps or e.age < self._min_survival_steps:
                kept.append(HistoryEntry(e.psi, w, e.utility, e.age))
        
        self._entries = kept

    def push(self, psi: Tensor, weight: float = 1.0) -> None:
        """
        改良 3: 新規エントリは utility=0.0（未知）、age=0 で開始
        """
        if psi.dim() == 1:
            psi = psi.unsqueeze(0)
        self._entries.append(HistoryEntry(psi.detach().clone(), weight, utility=0.0, age=0))
        if len(self._entries) > self.n_max:
            # 改良 3: 容量オーバー時は utility スコアの低いものから削除
            # ただし age < min_survival_steps のものは保護
            entries_list = list(self._entries)
            
            # 保護対象を分離
            protected = [e for e in entries_list if e.age < self._min_survival_steps]
            removable = [e for e in entries_list if e.age >= self._min_survival_steps]
            
            if len(protected) >= self.n_max:
                # 全て保護対象の場合、最も古いものを削除
                self._entries.popleft()
            else:
                # 削除対象から utility が最小のものを探す
                if removable:
                    min_util_entry = min(removable, key=lambda e: e.utility)
                    removable.remove(min_util_entry)
                    self._entries = deque(protected + removable, maxlen=self.n_max)
                else:
                    self._entries.popleft()

    def summarize(self, batch_size: int = 1) -> Tensor:
        """H_emb = Σ w_i · κ_emb(|ψ^{(i)}⟩)"""
        D_h = self.embedder.proj_re.out_features
        if not self._entries:
            return torch.zeros(batch_size, D_h, device=self.device)
        psis = torch.stack([e.psi.squeeze(0) for e in self._entries], dim=0)
        weights = torch.tensor([e.weight for e in self._entries], device=self.device)
        weights = weights / weights.sum().clamp_min(1e-8)
        emb = self.embedder(psis)
        h = torch.sum(weights.unsqueeze(-1) * emb, dim=0)
        return h.unsqueeze(0).expand(batch_size, -1)
