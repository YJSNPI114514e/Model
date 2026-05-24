"""観測・ボルン則・生成（sekkeisyo.txt 準拠 — 統一トークン埋め込み版）。

W_proj 廃止済み。ObservationBasis は classify モード専用。
LM モードでは全てトークン埋め込み |e_k⟩ を正規化して使用。
scores = |⟨ê_k|ψ_T⟩|²  (ê = normalize(e))
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from grim.geometry import complex_inner, normalize_state


class ObservationBasis(nn.Module):
    """分類タスク専用。LM では使わない。"""

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
    W_proj 廃止。トークン埋め込みを正規化してボルン則で直接観測。

    ê_k = normalize(e_k)  — 全トークン埋め込みを単位ベクトルに
    scores = |⟨ê_k|ψ_T⟩|²
    probs = scores / sum(scores)

    FM target も normalize(e_y) なので、ODE の到着点と観測が完全に一致する。
    """

    def __init__(self, dim: int, tokenizer: nn.Module) -> None:
        super().__init__()
        self.dim = dim
        self.tokenizer = tokenizer

    def _normalized_embeddings(self) -> Tensor:
        """全トークン埋め込みを単位ノルムに正規化。FM target と同じ空間。"""
        return normalize_state(self.tokenizer.embeddings)  # [V, D] complex, norm=1

    def token_scores(self, psi: Tensor) -> Tensor:
        """
        scores = |⟨ê_k|ψ_T⟩|²
        ê_k = normalize(e_k) — FM target と同じ正規化
        """
        emb_norm = self._normalized_embeddings()  # [V, D] unit norm
        # ⟨ê_k|ψ_T⟩ = conj(ê_k) @ ψ_T
        overlaps = torch.mm(psi, emb_norm.conj().T)  # [B, V]
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

        if forbid_ids:
            for tid in forbid_ids:
                if 0 <= tid < probs.shape[-1]:
                    probs[:, tid] = 0.0

        if recent_ids and repetition_penalty > 1.0:
            for tid in set(recent_ids):
                if 0 <= tid < probs.shape[-1]:
                    probs[:, tid] = probs[:, tid] / repetition_penalty

        if temperature > 0 and temperature != 1.0:
            log_probs = torch.log(probs.clamp_min(1e-8))
            log_probs = log_probs / temperature
            probs = torch.exp(log_probs)

        if top_k > 0 and top_k < probs.shape[-1]:
            v, _ = torch.topk(probs, top_k, dim=-1)
            probs = probs.masked_fill(probs < v[:, -1:], 0.0)

        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
