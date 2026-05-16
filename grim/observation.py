"""観測基底・ボルン則・生成射影（sekkeisyo.txt COMPONENT 4 準拠）。

修正: W_proj を廃止。psi_T と token_embeddings の直接内積でボルン則を計算。
scores = |⟨e_k|ψ_T⟩|²
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from grim.geometry import complex_inner


class ObservationBasis(nn.Module):
    """
    O = {|o_1⟩, ..., |o_K⟩}  QR正規直交

    sekkeisyo COMPONENT 4 (Classification):
    probs = |⟨o_k|ψ_T⟩|²  — Born Rule
    """

    def __init__(self, num_classes: int, dim: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        raw = torch.randn(num_classes, dim, dtype=torch.cfloat) / math.sqrt(dim)
        basis = self._orthonormalize(raw)
        self.register_buffer("basis", basis)

    @staticmethod
    def _orthonormalize(O: Tensor) -> Tensor:
        q, _ = torch.linalg.qr(O.T)
        return q.T.contiguous()

    def reorthogonalize(self) -> None:
        with torch.no_grad():
            self.basis.copy_(self._orthonormalize(self.basis))

    def born_probs(self, psi: Tensor) -> Tensor:
        """p(k) = |⟨o_k|ψ_T⟩|²"""
        overlaps = complex_inner(self.basis.unsqueeze(0), psi.unsqueeze(1), dim=-1)
        probs = torch.abs(overlaps) ** 2
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def target_state(self, labels: Tensor) -> Tensor:
        return self.basis[labels]

    @staticmethod
    def entropy(probs: Tensor) -> Tensor:
        p = probs.clamp_min(1e-8)
        return -torch.sum(p * torch.log(p), dim=-1)

    @staticmethod
    def confidence(probs: Tensor) -> Tensor:
        return probs.max(dim=-1).values


class GenerationHead(nn.Module):
    """
    W_proj 廃止。psi_T と token_embeddings の直接ボルン則。

    scores = |⟨e_k|ψ_T⟩|²
    probs = scores / sum(scores)
    """

    def __init__(self, dim: int, tokenizer: nn.Module) -> None:
        super().__init__()
        self.dim = dim
        self.tokenizer = tokenizer
        # W_proj は廃止。psi_T を直接トークン埋め込みと比較する。

    def token_scores(self, psi: Tensor) -> Tensor:
        """
        scores = |⟨e_k|ψ_T⟩|²
        psi_T を直接全トークン埋め込みと内積。これがボルン則の射影。
        """
        emb = self.tokenizer.embeddings  # [V, D] complex
        # ⟨e_k|ψ_T⟩ = sum(conj(e_k) * ψ_T)
        # emb: [V, D], psi: [B, D] → overlaps: [B, V]
        overlaps = torch.mm(psi, emb.conj().T)  # [B, V]
        return torch.abs(overlaps) ** 2

    def born_probs(self, psi: Tensor) -> Tensor:
        """Born Rule 正規化済み確率分布。"""
        scores = self.token_scores(psi)
        return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def predict_token(self, psi: Tensor, forbid_ids: list[int] | None = None) -> Tensor:
        probs = self.born_probs(psi)
        if forbid_ids:
            for tid in forbid_ids:
                if 0 <= tid < probs.shape[-1]:
                    probs[:, tid] = 0.0
        return probs.argmax(dim=-1)

    def sample_token(
        self,
        psi: Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        forbid_ids: list[int] | None = None,
        recent_ids: list[int] | None = None,
        repetition_penalty: float = 1.0,
    ) -> Tensor:
        """Born Rule 確率からサンプリング。"""
        probs = self.born_probs(psi)

        # 禁止トークン
        if forbid_ids:
            for tid in forbid_ids:
                if 0 <= tid < probs.shape[-1]:
                    probs[:, tid] = 0.0

        # 繰り返しペナルティ
        if recent_ids and repetition_penalty > 1.0:
            for tid in set(recent_ids):
                if 0 <= tid < probs.shape[-1]:
                    probs[:, tid] = probs[:, tid] / repetition_penalty

        # Temperature scaling
        if temperature > 0 and temperature != 1.0:
            log_probs = torch.log(probs.clamp_min(1e-8))
            log_probs = log_probs / temperature
            probs = torch.exp(log_probs)

        # Top-k フィルタリング
        if top_k > 0 and top_k < probs.shape[-1]:
            v, _ = torch.topk(probs, top_k, dim=-1)
            probs = probs.masked_fill(probs < v[:, -1:], 0.0)

        # 再正規化
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
