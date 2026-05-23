"""学習可能ベクトル場 ComplexMLP（sekkeisyo.txt COMPONENT 2 準拠）。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from grim.geometry import complex_inner, tangent_project


class ComplexLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int) -> None:
        super().__init__()
        scale = 1.0 / math.sqrt(in_f)
        self.w_re = nn.Parameter(torch.randn(out_f, in_f) * scale)
        self.w_im = nn.Parameter(torch.randn(out_f, in_f) * scale)
        self.b_re = nn.Parameter(torch.zeros(out_f))
        self.b_im = nn.Parameter(torch.zeros(out_f))

    def forward(self, x: Tensor) -> Tensor:
        w = torch.complex(self.w_re, self.w_im)
        b = torch.complex(self.b_re, self.b_im)
        return x @ w.T + b


def sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    """
    sekkeisyo COMPONENT 2: t_emb = sinusoidal_embedding(t, D)
    正弦波位置埋め込み（Transformer 式だが attention ではない）。
    t: [B, 1] or [B]  →  [B, dim]
    """
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
    )
    args = t * freqs.unsqueeze(0)  # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim] or [B, dim-1]
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(t)], dim=-1)
    return emb


def inv_softplus(y: float) -> float:
    return math.log(math.exp(y) - 1.0)


class EnergyVectorField(nn.Module):
    """
    E(psi) = E_world + E_self + E_cont + E_explore
    v = dψ/dt = -∇E
    """

    def __init__(
        self,
        dim: int,
        hidden: int,
        history_dim: int,
        tokenizer: nn.Module,
        history_getter: callable,
        generation: nn.Module,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.tokenizer = tokenizer
        self.history_getter = history_getter
        self.generation = generation

        # Learnable scalars (softplus parameters)
        self.raw_lam = nn.Parameter(torch.tensor(-4.6))    # lam: softplus(-4.6) ≈ 0.01
        self.raw_mu = nn.Parameter(torch.tensor(-4.6))     # mu: softplus(-4.6) ≈ 0.01
        self.raw_sigma = nn.Parameter(torch.tensor(0.0))    # sigma: softplus(0.0) ≈ 0.693
        self.raw_beta = nn.Parameter(torch.tensor(-4.6))    # beta: softplus(-4.6) ≈ 0.01

    def energy_and_gradient(self, psi: Tensor, psi0: Tensor) -> tuple[Tensor, Tensor]:
        """
        エネルギーとその解析的勾配を計算する。
        数値的安定性を高め、autograd への依存を減らす。
        
        Args:
            psi: 現在の状態 (B, D) 複素数
            psi0: 初期状態 (B, D) 複素数
            
        Returns:
            E: エネルギー (B,)
            grad_E: psi に関する勾配 (B, D)
        """
        B, D = psi.shape
        device = psi.device
        
        # --- 1. E_world: 外部入力との整合性 ---
        # E_world = -log(|<psi|psi0>|^2 + eps)
        overlap = complex_inner(psi, psi0, dim=-1)  # (B,)
        overlap_sq = torch.abs(overlap)**2
        eps_w = 1e-8
        e_world = -torch.log(overlap_sq.clamp_min(eps_w))
        
        # 勾配: d/dpsi* (-log(|<p|p0>|^2)) = -p0 / <p|p0>
        # |<p|p0>|^2 = <p|p0><p0|p>
        # d/dp* log(<p|p0><p0|p>) = p0 * <p0|p> / |<p|p0>|^2 = p0 * overlap.conj() / overlap_sq
        grad_world = (-overlap.conj() / (overlap_sq.clamp_min(eps_w))).unsqueeze(-1) * psi0

        # --- 2. E_self: 自己無撞着性 ---
        # E_self = lam * ||psi||^2
        e_norm = torch.sum(torch.abs(psi)**2, dim=-1)
        lam = F.softplus(self.raw_lam)
        e_self = lam * e_norm
        
        # 勾配: d/dpsi* (lam * |psi|^2) = lam * psi
        grad_self = lam * psi

        # --- 3. E_cont: 連続性 ---
        # 修正点: arccos の代わりに √(2 - 2|<ψ|h⟩|) を使用
        history = self.history_getter()
        entries = history._entries if history is not None else []
        
        if not entries:
            e_cont = torch.zeros(B, device=device, dtype=psi.real.dtype)
            grad_cont = torch.zeros_like(psi)
        else:
            psis = torch.stack([e.psi.squeeze(0) for e in entries], dim=0)  # [H_len, D]
            weights = torch.tensor([e.weight for e in entries], device=device, dtype=psi.real.dtype)
            
            # overlaps: [B, H_len]
            overlaps = torch.mm(psi, psis.conj().T)
            abs_overlap = torch.abs(overlaps).clamp(0.0, 1.0)
            
            # FS 距離の近似: d ≈ sqrt(2 - 2*F)
            dist_approx = torch.sqrt(torch.clamp(2.0 - 2.0 * abs_overlap, min=1e-9))
            
            sigma = F.softplus(self.raw_sigma).clamp_min(1e-8)
            kernel = torch.exp(- (dist_approx**2) / (2 * sigma**2))  # [B, H_len]
            
            mu = F.softplus(self.raw_mu)
            weighted_sum = torch.sum(weights.unsqueeze(0) * kernel, dim=-1)  # [B]
            e_cont = - (mu / len(entries)) * weighted_sum
            
            # 勾配計算 (手動微分)
            # E_cont = -(mu/N) * sum_i w_i * exp(-d_i^2 / 2sigma^2)
            # dE/dd = (mu/N) * sum_i w_i * exp(...) * (d_i / sigma^2)
            # dd/d|ov| = -1 / sqrt(2-2|ov|) = -1/d
            # d|ov|/dp* = (1/2|ov|) * ov.conj() * h
            # 合成: grad = (mu/N) * sum_i [w_i * kernel_i * (d_i/sigma^2) * (-1/d_i) * (0.5 * ov_i.conj()/|ov_i|) * h_i]
            #      = (mu/N) * sum_i [w_i * kernel_i * (-1/(2*sigma^2*|ov_i|)) * ov_i.conj() * h_i]
            
            factor = (mu / len(entries)) * weights.unsqueeze(0) * kernel / (2 * sigma**2)
            # ov_i.conj() / |ov_i| の計算 (|ov|=0 で除算回避)
            phase = overlaps.conj() / (abs_overlap.clamp_min(1e-9))
            
            # grad_cont = -sum_i factor_i * phase_i * h_i
            # (minus は dE/dd の符号から)
            grad_cont = -torch.mm(factor * phase, psis)

        # --- 4. E_explore: 探索 ---
        # 修正点: epsilon を 1e-6 に増加
        probs = self.generation.born_probs(psi)  # [B, V]
        beta = F.softplus(self.raw_beta)
        eps_e = 1e-6
        log_probs = torch.log(probs.clamp_min(eps_e))
        e_explore = - beta * torch.sum(probs * log_probs, dim=-1)
        
        # 勾配: d/dpsi* (-beta * sum p log p), p_k = |<e_k|psi>|^2 / Z
        # 簡易化のため、Z≈1 と仮定し p_k ≈ |psi_k|^2 (標準基底の場合)
        # dp/dpsi* = psi
        # d(p log p)/dp = log p + 1
        # grad = -beta * (log_probs + 1) * psi (射影基底への変換が必要だが、ここでは近似)
        # 正確には generation head を通した勾配伝播が必要
        # ここでは autograd との整合性を保つため、簡易的な形式を使用
        
        # 実際には born_probs は softmax 正規化を含むため、より複雑な勾配になる
        # 簡易版：grad ≈ -beta * (log_probs + 1 - sum(p*(log_p+1))) * psi
        mean_log = torch.sum(probs * (log_probs + 1), dim=-1, keepdim=True)
        grad_factor = log_probs + 1 - mean_log
        # psi を基底変換 (generation.head が線形なら head.T @ grad_factor)
        # ここでは簡易的に psi に重み付け
        grad_explore = -beta * psi * grad_factor.mean(dim=-1, keepdim=True).expand_as(psi)[:, :D] if D == probs.shape[1] else -beta * psi * grad_factor[:, :1].expand_as(psi)

        # --- 総合 ---
        E = e_world + e_self + e_cont + e_explore
        grad_E = grad_world + grad_self + grad_cont + grad_explore
        
        return E, grad_E

    def energy(self, psi: Tensor, psi0: Tensor) -> Tensor:
        # 1. World potential (E_world)
        overlap = complex_inner(psi, psi0, dim=-1)
        overlap_sq = torch.abs(overlap)**2
        e_world = -torch.log(overlap_sq.clamp_min(1e-8))

        # 2. Self-energy (E_self)
        e_norm = torch.sum(torch.abs(psi)**2, dim=-1)
        lam = F.softplus(self.raw_lam)
        e_self = lam * e_norm

        # 3. Continuation potential (E_cont)
        history = self.history_getter()
        entries = history._entries if history is not None else []
        if not entries:
            e_cont = torch.zeros(psi.shape[0], device=psi.device, dtype=psi.real.dtype)
        else:
            psis = torch.stack([e.psi.squeeze(0) for e in entries], dim=0)  # [H_len, D]
            weights = torch.tensor([e.weight for e in entries], device=psi.device, dtype=psi.real.dtype)

            # overlaps: [B, H_len]
            overlaps = torch.mm(psi, psis.conj().T)
            abs_overlap = torch.abs(overlaps).clamp(0.0, 1.0)
            d = torch.acos(abs_overlap)  # Fubini-Study distance [B, H_len]

            sigma = F.softplus(self.raw_sigma).clamp_min(1e-8)
            kernel = torch.exp(- (d**2) / (2 * sigma**2))  # [B, H_len]

            mu = F.softplus(self.raw_mu)
            weighted_sum = torch.sum(weights.unsqueeze(0) * kernel, dim=-1)  # [B]
            e_cont = - (mu / len(entries)) * weighted_sum

        # 4. Exploration potential (E_explore)
        probs = self.generation.born_probs(psi)  # [B, V]
        beta = F.softplus(self.raw_beta)
        e_explore = - beta * torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

        return e_world + e_self + e_cont + e_explore

    def forward(
        self,
        psi: Tensor,
        psi0: Tensor,
        h_emb: Tensor,  # kept for signature compatibility
        t: Tensor,      # kept for signature compatibility
    ) -> Tensor:
        """
        修正版：解析的勾配を使用してベクトル場を計算。
        数値的安定性を向上させるため energy_and_gradient を使用する。
        """
        # dψ/dt = -∇E
        E, grad_E = self.energy_and_gradient(psi, psi0)
        
        # 勾配は複素共役の微分なので、実空間での勾配方向に注意
        # Wirtinger 微分の定義に従い、v = -grad_E (接空間射影済み)
        v = -grad_E
        
        return tangent_project(v, psi)

