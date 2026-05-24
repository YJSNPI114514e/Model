"""GRIM NumPy 版：推論・生成専用モデル。

訓練ループ（逆伝播）は含まない。
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray
from typing import List, Optional

from grim.geometry_np import normalize_state, softmax
from grim.tokenizer_np import NumPyTokenizer
from grim.flow_field_np import NumPyEnergyField
from grim.ode_solver_np import integrate_flow
from grim.observation_np import born_probs
from grim.history_np import NumPyHistory, HistoryEntry


class NumPyGRIM:
    """
    GRIM NumPy 版：推論・生成専用
    
    訓練メソッドは含まない。バックプロパゲーションが必要な場合は
    PyTorch 版 (grim.model.GRIM) を使用すること。
    
    Attributes:
        tokenizer: トークナイザー
        energy_field: エネルギー場
        history: 履歴バッファ
        embeddings: 語彙埋め込み
    """
    
    def __init__(
        self,
        tokenizer: NumPyTokenizer,
        lam: float = 0.01,
        mu: float = 0.01,
        sigma: float = 0.693,
        beta: float = 0.01,
        history_max_entries: int = 100,
        history_gamma: float = 0.99,
    ) -> None:
        """NumPy GRIM モデル初期化
        
        Args:
            tokenizer: NumPy トークナイザー
            lam: 自己無撞着性重み
            mu: 連続性重み
            sigma: カーネル幅
            beta: 探索重み
            history_max_entries: 履歴最大エントリ数
            history_gamma: 履歴減衰率
        """
        self.tokenizer = tokenizer
        self.embeddings = tokenizer.embeddings
        self.energy_field = NumPyEnergyField(
            embeddings=self.embeddings,
            lam=lam,
            mu=mu,
            sigma=sigma,
            beta=beta,
        )
        self.history = NumPyHistory(
            max_entries=history_max_entries,
            gamma=history_gamma,
        )
        self.dim = tokenizer.dim
        self.vocab_size = tokenizer.vocab_size
    
    def tokenize(self, token_ids: ndarray) -> ndarray:
        """トークン列から初期状態 |ψ₀⟩ を構成
        
        Args:
            token_ids: トークン ID 列 (L,)
            
        Returns:
            初期状態 ψ₀ (D,), 正規化済み
        """
        return self.tokenizer.tokenize(token_ids.astype(np.int64))
    
    def integrate(
        self,
        psi_0: ndarray,
        T: float = 1.0,
    ) -> ndarray:
        """フローに沿って状態を積分
        
        Args:
            psi_0: 初期状態 (D,)
            T: 積分時間
            
        Returns:
            積分後の状態 ψ_T (D,), 正規化済み
        """
        psi_0 = normalize_state(psi_0.astype(np.complex128))
        
        # ODE 右辺関数のラッパー
        def ode_rhs(t, psi, psi_0_fixed, history):
            return self.energy_field.ode_rhs(t, psi, psi_0_fixed, history)
        
        psi_T = integrate_flow(
            ode_rhs=ode_rhs,
            psi_0=psi_0,
            psi_0_fixed=psi_0,
            history=self.history.short_term + self.history.mid_term + self.history.long_term,
            params={},
            T=T,
        )
        
        return psi_T
    
    def observe(self, psi_T: ndarray) -> ndarray:
        """ボルン則による確率計算
        
        Args:
            psi_T: 積分後の状態 (D,), 正規化済み
            
        Returns:
            確率分布 (V,), Σprobs = 1
        """
        return born_probs(psi_T.astype(np.complex128), self.embeddings)
    
    def generate(
        self,
        prompt_ids: ndarray,
        max_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
    ) -> List[int]:
        """自己回帰生成
        
        Args:
            prompt_ids: プロンプトトークン (L,)
            max_tokens: 最大生成トークン数
            temperature: 温度パラメータ
            top_k: トップ K サンプリング
            repetition_penalty: 繰り返しペナルティ
            
        Returns:
            生成されたトークン ID リスト
        """
        prompt_ids = prompt_ids.astype(np.int64)
        generated: List[int] = []
        recent_ids: List[int] = prompt_ids.tolist()
        
        for step in range(max_tokens):
            # トークナイズ
            psi_0 = self.tokenize(prompt_ids if step == 0 else np.array([generated[-1]]))
            
            # 積分
            psi_T = self.integrate(psi_0)
            
            # 観測
            probs = self.observe(psi_T)
            
            # 温度調整
            if temperature > 0 and temperature != 1.0:
                log_probs = np.log(probs.clip(min=1e-8))
                log_probs = log_probs / temperature
                probs = softmax(log_probs)
            
            # トップ K サンプリング
            if top_k > 0 and top_k < len(probs):
                sorted_indices = np.argsort(probs)[::-1]
                top_k_indices = sorted_indices[:top_k]
                mask = np.zeros_like(probs)
                mask[top_k_indices] = 1.0
                probs = probs * mask
                probs = probs / probs.sum().clip(min=1e-8)
            
            # 繰り返しペナルティ
            if repetition_penalty > 1.0:
                for tid in set(recent_ids[-8:]):
                    if 0 <= tid < len(probs):
                        probs[tid] = probs[tid] / repetition_penalty
                probs = probs / probs.sum().clip(min=1e-8)
            
            # サンプリング
            next_id = np.random.choice(len(probs), p=probs)
            generated.append(int(next_id))
            recent_ids.append(int(next_id))
            
            # EOS チェック（簡易的に 0 を EOS と仮定）
            if next_id == 0:
                break
        
        return generated
    
    def reset_history(self) -> None:
        """履歴をリセット"""
        self.history.clear()
    
    def update_history(self, psi: ndarray, weight: float = 1.0) -> None:
        """履歴に状態を追加
        
        Args:
            psi: 状態ベクトル (D,)
            weight: 重み
        """
        self.history.push(psi.astype(np.complex128), weight)
    
    def decay_history(self, boundary_prob: float = 0.0) -> None:
        """履歴重みを減衰
        
        Args:
            boundary_prob: 文境界確率
        """
        self.history.decay(boundary_prob)
