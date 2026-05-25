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
        改良 1: ボルン則に基づくクロスエントロピー損失
        
        Born 則により確率を計算：
        p(k) = |⟨e_k|ψ_T⟩|² / Σ_j |⟨e_j|ψ_T⟩|²
        
        クロスエントロピー損失：
        loss = -log(p(y))
        """
        # トークン埋め込みとの内積の二乗（ボルン則）
        token_embeddings = self.tokenizer.embeddings  # [V, D]
        scores = torch.abs(token_embeddings @ psi_T.conj().T) ** 2  # [V, B]
        
        # 正規化して確率に変換
        probs = scores / (scores.sum(dim=0, keepdim=True) + 1e-8)  # [V, B]
        
        # ターゲットトークンの確率を取得
        B = psi_T.shape[0]
        target_probs = probs[target_token_ids, torch.arange(B, device=psi_T.device)]
        
        # クロスエントロピー損失
        loss = -torch.log(target_probs + 1e-8).mean()
        
        return loss

    def observation_loss(self, psi_T: Tensor, labels: Tensor) -> Tensor:
        probs = self.observation.born_probs(psi_T)
        return F.nll_loss(torch.log(probs.clamp_min(1e-8)), labels)

    def integrate(self, psi0: Tensor, h_emb: Tensor, num_steps: int | None = None) -> Tensor:
        """
        Residual Flow: K 段オイラー法（接空間射影済み）
        
        d|ψ⟩/dt = v を固定ステップで離散化：
        ψ_{k+1} = normalize(ψ_k + dt * v(ψ_k))
        
        Args:
            psi0: 初期状態 (B, D)
            h_emb: 履歴埋め込み (B, D_h)
            num_steps: 積分ステップ数（K）。None の場合は config.num_flow_steps を使用。
            
        Returns:
            psi_T: 終端状態 (B, D)
        """
        if num_steps is None:
            num_steps = self.config.num_flow_steps
        
        psi = psi0
        dt = 1.0 / num_steps
        for _ in range(num_steps):
            v = self.flow_field(psi, psi0, h_emb)
            psi = normalize_state(psi + dt * v)
        return psi

    def forward_train(
        self,
        token_ids: Tensor,
        labels: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        B = token_ids.shape[0]
        psi0 = self.tokenize(token_ids, mask)
        h_emb = self.summarize_history(B)
        t = torch.rand(B, device=self.device)

        target = self.observation.target_state(labels)
        L_fm = torch.tensor(0.0, device=self.device)
        psi_T = self.integrate(psi0, h_emb)
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
    ) -> dict[str, Tensor]:
        """自然言語: 文脈 → 次トークン（Flow Matching + 語彙上の観測損失）。"""
        B = context_ids.shape[0]
        psi0 = self.tokenize(context_ids, mask)
        h_emb = self.summarize_history(B)
        t = torch.rand(B, device=self.device)

        L_fm = torch.tensor(0.0, device=self.device)
        psi_T = self.integrate(psi0, h_emb)
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
        psi_T = self.integrate(psi0, h_emb)
        return self.generation.predict_token(psi_T)

    @torch.no_grad()
    def classify(self, token_ids: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """GRIM_classify: クラス, 信頼度, エントロピー"""
        B = token_ids.shape[0]
        psi0 = self.tokenize(token_ids, mask)
        h_emb = self.summarize_history(B)
        psi_T = self.integrate(psi0, h_emb)
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
        """GRIM_generate（第 4.3 節）+ サンプリングで繰り返し抑制
        
        改良：逐次 Born 観測を停止し、期待値埋め込みで状態をソフトに更新。
        これにより「同じ単語のループ」と「漢字偏り」を大幅に改善。
        """
        cfg = self.config
        if local_history_max is None:
            local_history_max = min(24, cfg.N_max)
        temperature = cfg.temperature if temperature is None else temperature
        top_k = cfg.top_k if top_k is None else top_k
        repetition_penalty = cfg.repetition_penalty if repetition_penalty is None else repetition_penalty
        use_sliding_context = cfg.use_sliding_context if use_sliding_context is None else use_sliding_context

        mix_coeff = cfg.expected_mix_coeff
        device = self.device
        context: list[int] = prompt_ids.view(-1).tolist()
        local_hist: list[Tensor] = []
        forbid = self._forbid_special_ids() if forbid_special else None
        generated: list[int] = []
        
        # 生成開始時に履歴をクリア（指示 4: 既存動作維持）
        self.history.clear()

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

            s_T = self.integrate(psi0, h_emb())
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

            # 改良：期待値埋め込みによる状態更新（逐次 Born 観測の代替）
            # トークン ID のサンプリングは「出力用」としてのみ使用し、状態更新には使わない
            if not use_sliding_context:
                probs = self.generation.born_probs(s_T)  # [1, V] 確率分布
                # 複素埋め込みと float 確率の乗算：実部と虚部に分けて計算
                emb = self.tokenizer.embeddings  # [V, D] complex
                expected_emb_real = probs @ emb.real  # [1, D]
                expected_emb_imag = probs @ emb.imag  # [1, D]
                expected_emb = torch.complex(expected_emb_real, expected_emb_imag)  # [1, D] complex
                expected_emb = normalize_state(expected_emb)  # 正規化
                s = normalize_state(s_T + mix_coeff * expected_emb)  # 微小混合（上書きしない）
            
            local_hist.append(s_T.squeeze(0))  # 履歴には生の状態を保存

        return generated

    def reorthogonalize_obs(self) -> None:
        if self.config.task_mode == "classify":
            self.observation.reorthogonalize()
