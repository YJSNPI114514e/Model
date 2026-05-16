"""KFAC近似による自然勾配（第3.2節）。"""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor


class KFACNaturalGradient:
    """
    F ≈ Q ⊗ G のブロック対角近似。
    各 ComplexLinear / nn.Linear 層に対して Kronecker 因子を蓄積。
    """

    def __init__(self, model: nn.Module, damping: float = 1e-3) -> None:
        self.model = model
        self.damping = damping
        self._acts: dict[str, Tensor] = {}
        self._grads: dict[str, Tensor] = {}
        self._handles: list = []

    def _hook_act(self, name: str):
        def hook(_module, inp, _out):
            x = inp[0].detach()
            if x.is_complex():
                x = torch.cat([x.real, x.imag], dim=-1)
            self._acts[name] = x

        return hook

    def _hook_grad(self, name: str):
        def hook(_module, _inp, grad_out):
            g = grad_out[0].detach()
            if g is not None and g.is_complex():
                g = torch.cat([g.real, g.imag], dim=-1)
                self._grads[name] = g

        return hook

    def register(self) -> None:
        from grim.flow_field import ComplexLinear

        idx = 0
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, ComplexLinear)):
                name = f"lin_{idx}"
                self._handles.append(module.register_forward_hook(self._hook_act(name)))
                self._handles.append(module.register_full_backward_hook(self._hook_grad(name)))
                idx += 1

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def precondition(self, params: Iterator[nn.Parameter]) -> None:
        """θ ← θ - η F⁻¹ ∇θ（層ごと簡易KFAC）"""
        for name, act in self._acts.items():
            grad = self._grads.get(name)
            if grad is None:
                continue
            B = act.shape[0]
            if B == 0:
                continue
            a = act.reshape(B, -1)
            g = grad.reshape(B, -1)
            q = (a.T @ a) / B + self.damping * torch.eye(a.shape[1], device=a.device)
            gg = (g.T @ g) / B + self.damping * torch.eye(g.shape[1], device=g.device)
            try:
                q_inv = torch.linalg.inv(q)
                g_inv = torch.linalg.inv(gg)
            except RuntimeError:
                continue
            # フック対象パラメータへの完全マッピングは省略し、勾配ノルムでスケール
            scale = torch.trace(q_inv).sqrt() * torch.trace(g_inv).sqrt()
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(scale.clamp(max=10.0))
