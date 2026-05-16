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
    H = { (|ψ^{(i)}⟩, w_i) }
    書込・減衰・容量制限（循環バッファ）
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

    def clear(self) -> None:
        self._entries.clear()

    def decay(self) -> None:
        kept: deque[HistoryEntry] = deque(maxlen=self.n_max)
        for e in self._entries:
            w = e.weight * self.gamma
            if w >= self.eps:
                kept.append(HistoryEntry(e.psi, w))
        self._entries = kept

    def push(self, psi: Tensor, weight: float = 1.0) -> None:
        if psi.dim() == 1:
            psi = psi.unsqueeze(0)
        self._entries.append(HistoryEntry(psi.detach().clone(), weight))
        if len(self._entries) > self.n_max:
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
