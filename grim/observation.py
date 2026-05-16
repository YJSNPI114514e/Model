"""観測基底・ボルン則・生成射影（第2.6節）。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from grim.geometry import complex_inner


class ObservationBasis(nn.Module):
    """O = {|o_1⟩, ..., |o_K⟩}  QR正規直交"""

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
    """|e_out⟩ = W_proj |ψ_T⟩"""

    def __init__(self, dim: int, token_embeddings: nn.Parameter) -> None:
        super().__init__()
        scale = 1.0 / math.sqrt(dim)
        self.W_re = nn.Parameter(torch.randn(dim, dim) * scale)
        self.W_im = nn.Parameter(torch.randn(dim, dim) * scale)
        self.token_embeddings = token_embeddings

    @property
    def W_proj(self) -> Tensor:
        return torch.complex(self.W_re, self.W_im)

    def project(self, psi: Tensor) -> Tensor:
        return psi @ self.W_proj

    def token_scores(self, psi: Tensor) -> Tensor:
        e_out = self.project(psi)
        emb = self.token_embeddings
        overlaps = complex_inner(emb.unsqueeze(0), e_out.unsqueeze(1), dim=-1)
        return torch.abs(overlaps) ** 2

    def predict_token(self, psi: Tensor, forbid_ids: list[int] | None = None) -> Tensor:
        logp = self._log_probs(psi, forbid_ids, None, 1.0)
        return logp.argmax(dim=-1)

    def _log_probs(
        self,
        psi: Tensor,
        forbid_ids: list[int] | None,
        recent_ids: list[int] | None,
        repetition_penalty: float,
    ) -> Tensor:
        scores = self.token_scores(psi).clamp_min(1e-8)
        logp = torch.log(scores)
        logp = logp - torch.logsumexp(logp, dim=-1, keepdim=True)
        if forbid_ids:
            for tid in forbid_ids:
                if 0 <= tid < logp.shape[-1]:
                    logp[:, tid] = -1e9
        if recent_ids and repetition_penalty > 1.0:
            for tid in set(recent_ids):
                if 0 <= tid < logp.shape[-1]:
                    logp[:, tid] -= torch.log(torch.tensor(repetition_penalty, device=logp.device))
        return logp

    def sample_token(
        self,
        psi: Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        forbid_ids: list[int] | None = None,
        recent_ids: list[int] | None = None,
        repetition_penalty: float = 1.0,
    ) -> Tensor:
        logp = self._log_probs(psi, forbid_ids, recent_ids, repetition_penalty)
        if temperature > 0 and temperature != 1.0:
            logp = logp / temperature
        if top_k > 0 and top_k < logp.shape[-1]:
            v, _ = torch.topk(logp, top_k, dim=-1)
            logp = logp.masked_fill(logp < v[:, -1:], -1e9)
        probs = torch.softmax(logp, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
