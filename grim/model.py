"""GRIM 統合モデル（sekkeisyo.txt 準拠）。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from grim.config import GRIMConfig
from grim.flow_field import EnergyVectorField
from grim.geometry import (
    fs_norm_sq,
    geodesic_interp,
    geodesic_velocity_target,
    normalize_state,
)
from grim.history import HistoryBuffer, HistoryEmbedder
from grim.meta import MetaParams
from grim.observation import GenerationHead, ObservationBasis
from grim.ode_solver import integrate_flow
from grim.tokenizer import ComplexTokenizer


class GRIM(nn.Module):
    def __init__(self, config: GRIMConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = ComplexTokenizer(
            vocab_size=config.V,
            dim=config.D,
            max_len=config.M_max,
            w_alpha=config.w_alpha,
        )
        self.history_embedder = HistoryEmbedder(config.D, config.D_h)
        self.generation = GenerationHead(config.D, self.tokenizer)
        self.flow_field = EnergyVectorField(
            dim=config.D,
            hidden=config.flow_hidden,
            history_dim=config.D_h,
            tokenizer=self.tokenizer,
            history_getter=lambda: self.history,
            generation=self.generation,
        )
        # 分類モードのみ ObservationBasis を使用
        # LM モードではトークン埋め込みが観測基底を兼ねる
        if config.task_mode == "classify":
            self.observation = ObservationBasis(config.K, config.D)
        self.meta = MetaParams(beta=config.beta_kl)
        self._history: HistoryBuffer | None = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def init_history(self) -> HistoryBuffer:
        self._history = HistoryBuffer(
            n_max=self.config.N_max,
            gamma=self.config.gamma,
            eps=self.config.history_eps,
            embedder=self.history_embedder,
            device=self.device,
        )
        return self._history

    @property
    def history(self) -> HistoryBuffer:
        if self._history is None:
            return self.init_history()
        return self._history

    def summarize_history(self, batch_size: int) -> Tensor:
        return self.history.summarize(batch_size)

    def tokenize(self, token_ids: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.tokenizer(token_ids, mask)

    def token_target_state(self, token_ids: Tensor) -> Tensor:
        """次トークン埋め込み |e_t⟩ を多様体上の目標状態に正規化。"""
        emb = self.tokenizer.embeddings[token_ids]
        return normalize_state(emb)

    def flow_matching_loss(
        self,
        psi0: Tensor,
        target_state: Tensor,
        h_emb: Tensor,
        t: Tensor | None = None,
    ) -> Tensor:
        target = target_state
        if t is None:
            t = torch.rand(psi0.shape[0], device=psi0.device)
        psi_t = geodesic_interp(psi0, target, t)
        v_target = geodesic_velocity_target(psi0, target, t, psi_t)
        v_pred = self.flow_field(psi_t, psi0, h_emb, t)
        return torch.mean(fs_norm_sq(v_pred - v_target))

    def flow_matching_loss_to_tokens(
        self,
        psi0: Tensor,
        target_token_ids: Tensor,
        h_emb: Tensor,
        t: Tensor | None = None,
    ) -> Tensor:
        target = self.token_target_state(target_token_ids)
        return self.flow_matching_loss(psi0, target, h_emb, t)

    def language_modeling_loss(self, psi_T: Tensor, target_token_ids: Tensor) -> Tensor:
        """
        改良 1: マージン最大化損失（ヒンジ損失 + ボルン則）
        
        correct_score = |⟨e_y|ψ_T⟩|²
        best_wrong_score = max_{k≠y} |⟨e_k|ψ_T⟩|²
        margin = correct_score - best_wrong_score
        loss = max(0, 1.0 - margin)  # ヒンジ損失
        
        マージンが 1 未満なら罰則、1 以上ならゼロ。
        εや log は不要。計算が軽くなる。
        """
        scores = self.generation.token_scores(psi_T)  # [B, V]
        B = scores.shape[0]
        
        # 正解スコア
        y = target_token_ids
        correct_score = scores[torch.arange(B, device=scores.device), y]  # [B]
        
        # 不正解の最大スコア：正解をマスク
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[torch.arange(B, device=scores.device), y] = False
        wrong_scores = scores.masked_fill(mask, float('-inf'))
        best_wrong_score = wrong_scores.max(dim=-1).values  # [B]
        
        # マージンとヒンジ損失
        margin = correct_score - best_wrong_score
        loss = torch.clamp(1.0 - margin, min=0.0)  # max(0, 1 - margin)
        
        return loss.mean()

    def observation_loss(self, psi_T: Tensor, labels: Tensor) -> Tensor:
        probs = self.observation.born_probs(psi_T)
        return F.nll_loss(torch.log(probs.clamp_min(1e-8)), labels)

    def integrate(self, psi0: Tensor, h_emb: Tensor, use_amp: bool = False) -> Tensor:
        """
        sekkeisyo COMPONENT 3 / VIOLATION 6:
        DOPRI5 のみ使用。Euler フォールバック禁止。
        
        use_amp: True で混合精度計算（GPU 時のみ有効）
        """
        return integrate_flow(
            self.flow_field,
            psi0,
            h_emb,
            method=self.config.ode_method,
            rtol=self.config.ode_rtol,
            atol=self.config.ode_atol,
            use_amp=use_amp,
        )

    def forward_train(
        self,
        token_ids: Tensor,
        labels: Tensor,
        mask: Tensor | None = None,
        use_amp: bool = False,
    ) -> dict[str, Tensor]:
        B = token_ids.shape[0]
        psi0 = self.tokenize(token_ids, mask)
        h_emb = self.summarize_history(B)
        t = torch.rand(B, device=self.device)

        target = self.observation.target_state(labels)
        L_fm = torch.tensor(0.0, device=self.device)
        psi_T = self.integrate(psi0, h_emb, use_amp=use_amp)
        L_obs = self.observation_loss(psi_T, labels)

        # L_fm を捨てて L_obs のみで学習する
        w = self.meta
        L = F.softplus(w.obs_weight) * L_obs

        return {
            "loss": L,
            "loss_fm": L_fm,
            "loss_obs": L_obs,
            "psi0": psi0,
            "psi_T": psi_T,
        }

    def forward_train_lm(
        self,
        context_ids: Tensor,
        target_ids: Tensor,
        mask: Tensor | None = None,
        use_amp: bool = False,
    ) -> dict[str, Tensor]:
        """自然言語: 文脈 → 次トークン（Flow Matching + 語彙上の観測損失）。"""
        B = context_ids.shape[0]
        psi0 = self.tokenize(context_ids, mask)
        h_emb = self.summarize_history(B)
        t = torch.rand(B, device=self.device)

        L_fm = torch.tensor(0.0, device=self.device)
        psi_T = self.integrate(psi0, h_emb, use_amp=use_amp)
        L_lm = self.language_modeling_loss(psi_T, target_ids)

        # L_fm を捨てて L_obs のみで学習する
        w = self.meta
        L = F.softplus(w.obs_weight) * L_lm

        return {
            "loss": L,
            "loss_fm": L_fm,
            "loss_obs": L_lm,
            "psi0": psi0,
            "psi_T": psi_T,
        }

    def forward_train_batch(
        self,
        context_ids: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if self.config.task_mode == "lm":
            return self.forward_train_lm(context_ids, targets, mask)
        return self.forward_train(context_ids, targets, mask)

    @torch.no_grad()
    def predict_next_token(self, context_ids: Tensor, mask: Tensor | None = None) -> Tensor:
        psi0 = self.tokenize(context_ids, mask)
        h_emb = self.summarize_history(context_ids.shape[0])
        psi_T = self.integrate(psi0, h_emb, use_amp=False)
        return self.generation.predict_token(psi_T)

    @torch.no_grad()
    def classify(self, token_ids: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """GRIM_classify: クラス, 信頼度, エントロピー"""
        B = token_ids.shape[0]
        psi0 = self.tokenize(token_ids, mask)
        h_emb = self.summarize_history(B)
        psi_T = self.integrate(psi0, h_emb, use_amp=False)
        probs = self.observation.born_probs(psi_T)
        pred = probs.argmax(dim=-1)
        conf = self.observation.confidence(probs)
        ent = self.observation.entropy(probs)
        self.history.decay()
        self.history.push(psi_T[0])
        return pred, conf, ent

    def _forbid_special_ids(self) -> list[int]:
        return [self.config.pad_id, self.config.eos_id, 2]

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 64,
        local_history_max: int | None = None,
        min_new_tokens: int = 1,
        forbid_special: bool = True,
        temperature: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        use_sliding_context: bool | None = None,
        greedy: bool = False,
    ) -> list[int]:
        """GRIM_generate（第4.3節）+ サンプリングで繰り返し抑制"""
        cfg = self.config
        if local_history_max is None:
            local_history_max = min(24, cfg.N_max)
        temperature = cfg.temperature if temperature is None else temperature
        top_k = cfg.top_k if top_k is None else top_k
        repetition_penalty = cfg.repetition_penalty if repetition_penalty is None else repetition_penalty
        use_sliding_context = cfg.use_sliding_context if use_sliding_context is None else use_sliding_context

        device = self.device
        context: list[int] = prompt_ids.view(-1).tolist()
        local_hist: list[Tensor] = []
        forbid = self._forbid_special_ids() if forbid_special else None
        generated: list[int] = []

        def h_emb(batch: int = 1) -> Tensor:
            if not local_hist:
                return torch.zeros(batch, cfg.D_h, device=device)
            psis = torch.stack(local_hist[-local_history_max:], dim=0)
            emb = self.history_embedder(psis)
            return emb.mean(dim=0, keepdim=True).expand(batch, -1)

        for step in range(max_new_tokens):
            if use_sliding_context:
                window = context[-cfg.M_max :]
                psi0 = self.tokenize(torch.tensor([window], device=device, dtype=torch.long))
            else:
                if step == 0:
                    s = self.tokenize(prompt_ids.view(1, -1))
                psi0 = s

            s_T = self.integrate(psi0, h_emb(), use_amp=False)
            recent = context[-8:] + generated[-8:]

            if greedy:
                next_id = int(self.generation.predict_token(s_T, forbid_ids=forbid).item())
            else:
                next_id = int(
                    self.generation.sample_token(
                        s_T,
                        temperature=temperature,
                        top_k=top_k,
                        forbid_ids=forbid,
                        recent_ids=recent,
                        repetition_penalty=repetition_penalty,
                    ).item()
                )

            generated.append(next_id)
            context.append(next_id)

            if step + 1 >= min_new_tokens and next_id == cfg.eos_id:
                break

            if not use_sliding_context:
                s = self.tokenizer.inject(s_T, torch.tensor(next_id, device=device))
            local_hist.append(s_T.squeeze(0))

        return generated

    def reorthogonalize_obs(self) -> None:
        if self.config.task_mode == "classify":
            self.observation.reorthogonalize()
