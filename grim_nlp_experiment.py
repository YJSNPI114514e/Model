#!/usr/bin/env python3
"""GRIM-NLP 実装検証実験：数式定義書（実験用）の最小実装

勾配フローを正しく機能させるため、Tokenizer を簡略化
Householder 変換も勾配対応版に変更
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

D = 32
L = 16
EPOCHS = 10
LR = 0.001
WEIGHT_DECAY = 0.01
OMEGA = 2 * np.pi
EPS = 1e-8
ODE_STEPS = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class Tokenizer(nn.Module):
    def __init__(self, vocab_size: int = 100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, D)
        self.q = nn.Parameter(torch.randn(D))
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        emb = self.embedding(token_ids)
        q_expanded = self.q.unsqueeze(0).unsqueeze(0)
        scores = (q_expanded * emb).sum(dim=-1) / np.sqrt(D)
        alpha = F.softmax(scores, dim=-1)
        positions = torch.arange(1, seq_len + 1, device=emb.device).float()
        phi = OMEGA * (positions / L)
        phase_real = torch.cos(phi).unsqueeze(0)
        phase_imag = torch.sin(phi).unsqueeze(0)
        psi_0_real = ((alpha * phase_real).unsqueeze(-1) * emb).sum(dim=1)
        psi_0_imag = ((alpha * phase_imag).unsqueeze(-1) * emb).sum(dim=1)
        psi_0 = torch.complex(psi_0_real, psi_0_imag)
        norm = torch.norm(psi_0, dim=-1, keepdim=True)
        psi_0 = psi_0 / (norm + EPS)
        return psi_0


def vector_field(psi, psi_0):
    inner = torch.sum(psi_0.conj() * psi, dim=-1, keepdim=True)
    denom = torch.abs(inner)**2 + EPS
    v = (inner / denom) * psi_0
    proj_coeff = torch.sum(psi.conj() * v, dim=-1, keepdim=True)
    v_perp = v - proj_coeff * psi
    return v_perp


def euler_integrate(psi_0, steps=ODE_STEPS):
    psi = psi_0.clone()
    dt = 1.0 / steps
    for _ in range(steps):
        v = vector_field(psi, psi_0)
        psi = psi + dt * v
    norm = torch.norm(psi, dim=-1, keepdim=True)
    psi = psi / (norm + EPS)
    return psi, steps


class HouseholderTransform(nn.Module):
    """Vectorized Householder transform that preserves gradients"""
    def __init__(self, K=None):
        super().__init__()
        if K is None:
            K = min(D, 8)
        self.K = K
        self.u = nn.Parameter(torch.randn(K, D, dtype=torch.complex64))
        
    def forward(self, psi):
        # psi: [batch, D]
        result = psi
        for k in range(self.K):
            u_k = self.u[k]  # [D]
            u_conj = u_k.conj()
            denom = torch.sum(u_conj * u_k) + EPS
            # Vectorized: compute for all batches at once
            numer = torch.sum(result * u_conj.unsqueeze(0), dim=-1, keepdim=True)  # [batch, 1]
            result = result - 2 * (numer / denom) * u_k.unsqueeze(0)  # [batch, D]
        return result


class Observer(nn.Module):
    def __init__(self, use_b=False):
        super().__init__()
        self.use_b = use_b
        if use_b:
            self.B = HouseholderTransform()
        
    def forward(self, psi_T):
        if self.use_b:
            psi_tilde = self.B(psi_T)
        else:
            psi_tilde = psi_T
        s = torch.abs(psi_tilde)**2
        P = s / (torch.sum(s, dim=-1, keepdim=True) + EPS)
        log_P = torch.log(P + EPS)
        return P, log_P


class GRIMNLP(nn.Module):
    def __init__(self, vocab_size=100, use_b=False):
        super().__init__()
        self.tokenizer = Tokenizer(vocab_size)
        self.observer = Observer(use_b)
        
    def ode_step(self, psi_0):
        psi_T, num_steps = euler_integrate(psi_0)
        return psi_T, num_steps
    
    def forward(self, token_ids):
        psi_0 = self.tokenizer(token_ids)
        psi_T, num_steps = self.ode_step(psi_0)
        P, log_P = self.observer(psi_T)
        y_true = token_ids[:, -1] % D
        loss = -log_P[torch.arange(log_P.shape[0]), y_true].mean()
        return psi_T, P, loss, num_steps
    
    def generate(self, prompt_ids, max_tokens=32, temperature=0.8, alpha_prev=0.5):
        generated = []
        current_ids = prompt_ids.unsqueeze(0)
        with torch.no_grad():
            for tau in range(max_tokens):
                psi_0 = self.tokenizer(current_ids)
                psi_T, _ = self.ode_step(psi_0)
                P, _ = self.observer(psi_T)
                logits = torch.log(P[0] + EPS) / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
                phi_tau = OMEGA * ((tau + 1) / L)
                phase = torch.exp(torch.tensor(1j * phi_tau, device=device))
                new_basis = torch.zeros(D, dtype=torch.complex64, device=device)
                new_basis[next_token % D] = phase
                psi_new = alpha_prev * psi_T[0] + 0.5 * new_basis
                psi_new = psi_new / (torch.norm(psi_new) + EPS)
                new_ids = torch.cat([current_ids[0], torch.tensor([next_token], device=device)]).unsqueeze(0)
                if new_ids.shape[1] > L:
                    new_ids = new_ids[:, -L:]
                current_ids = new_ids
        return generated


def create_corpus():
    corpus = []
    for _ in range(10):
        corpus.append(random.sample(list(range(0, 46)), 10))
    for _ in range(10):
        corpus.append(random.sample(list(range(46, 92)), 10))
    for _ in range(10):
        corpus.append([random.choice(list(range(92, 98))) for _ in range(10)])
    for _ in range(10):
        corpus.append([random.randint(0, 99) for _ in range(10)])
    return corpus


def evaluate_metrics(model, corpus):
    metrics = {
        'state_transition': [],
        'norm_error': [],
        'cosine_similarities': [],
        'token_frequencies': {'hiragana': 0, 'katakana': 0, 'kanji': 0, 'alphanumeric': 0},
        'total_tokens': 0
    }
    model.eval()
    all_psi_T = []
    with torch.no_grad():
        for sent in corpus:
            token_ids = torch.tensor([sent], device=device)
            psi_0 = model.tokenizer(token_ids)
            psi_T, _ = model.ode_step(psi_0)
            delta = torch.norm(psi_T - psi_0, dim=-1).item()
            metrics['state_transition'].append(delta)
            norm_error = abs(torch.norm(psi_T).item() - 1.0)
            metrics['norm_error'].append(norm_error)
            P, _ = model.observer(psi_T)
            for k in range(D):
                prob = P[0, k].item()
                if k < 46:
                    metrics['token_frequencies']['hiragana'] += prob
                elif k < 92:
                    metrics['token_frequencies']['katakana'] += prob
                elif k < 98:
                    metrics['token_frequencies']['kanji'] += prob
                else:
                    metrics['token_frequencies']['alphanumeric'] += prob
                metrics['total_tokens'] += prob
            all_psi_T.append(psi_T[0])
    if len(all_psi_T) > 1:
        cos_sims = []
        for i in range(len(all_psi_T)):
            for j in range(i+1, len(all_psi_T)):
                cos_sim = torch.abs(torch.sum(all_psi_T[i].conj() * all_psi_T[j])).item()
                cos_sims.append(cos_sim)
        metrics['cosine_similarities'] = cos_sims
    if metrics['total_tokens'] > 0:
        for key in metrics['token_frequencies']:
            metrics['token_frequencies'][key] /= metrics['total_tokens']
    return metrics


def train_model(name, use_b, corpus, epochs=EPOCHS):
    print(f"\n{'='*60}\nTraining {name} (B-separation: {use_b})\n{'='*60}")
    model = GRIMNLP(vocab_size=100, use_b=use_b).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for sent in corpus:
            token_ids = torch.tensor([sent], device=device)
            optimizer.zero_grad()
            psi_T, P, loss, num_steps = model(token_ids)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(corpus)
        metrics = evaluate_metrics(model, corpus)
        metrics['epoch'] = epoch
        metrics['loss'] = avg_loss
        history.append(metrics)
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Δ={np.mean(metrics['state_transition']):.4f}, Norm err={np.mean(metrics['norm_error']):.2e}")
    return model, history


def check_consecutive_tokens(generated, threshold=5):
    if len(generated) < threshold:
        return False
    count = 1
    for i in range(1, len(generated)):
        if generated[i] == generated[i-1]:
            count += 1
            if count >= threshold:
                return True
        else:
            count = 1
    return False


def main():
    print("GRIM-NLP 実装検証実験\n" + "="*60)
    corpus = create_corpus()
    print(f"Created corpus with {len(corpus)} sentences")
    
    model_a, history_a = train_model("Experiment A (No B)", False, corpus)
    model_b, history_b = train_model("Experiment B (With B)", True, corpus)
    
    print("\n" + "="*60 + "\nRESULTS COMPARISON\n" + "="*60)
    final_a, final_b = history_a[-1], history_b[-1]
    
    print("\n1. 状態遷移量 Δ (基準：> 0.1)")
    delta_a, delta_b = np.mean(final_a['state_transition']), np.mean(final_b['state_transition'])
    print(f"   A: Δ = {delta_a:.4f} {'✓' if delta_a > 0.1 else '✗'}")
    print(f"   B: Δ = {delta_b:.4f} {'✓' if delta_b > 0.1 else '✗'}")
    
    print("\n2. 軌道崩壊 (ノルム誤差 < 1e-5)")
    norm_a, norm_b = np.mean(final_a['norm_error']), np.mean(final_b['norm_error'])
    print(f"   A: err = {norm_a:.2e} {'✓' if norm_a < 1e-5 else '✗'}")
    print(f"   B: err = {norm_b:.2e} {'✓' if norm_b < 1e-5 else '✗'}")
    
    print("\n3. B 分離の効果（トークン分布）")
    print(f"   A: {final_a['token_frequencies']}")
    print(f"   B: {final_b['token_frequencies']}")
    
    print("\n4. 軌道多様性（コサイン類似度の分散）")
    var_a = np.var(final_a['cosine_similarities']) if final_a['cosine_similarities'] else 0
    var_b = np.var(final_b['cosine_similarities']) if final_b['cosine_similarities'] else 0
    print(f"   A: var = {var_a:.4f}")
    print(f"   B: var = {var_b:.4f}")
    
    print("\n5. 損失の減少")
    print(f"   A: {history_a[0]['loss']:.4f} → {history_a[-1]['loss']:.4f}")
    print(f"   B: {history_b[0]['loss']:.4f} → {history_b[-1]['loss']:.4f}")
    
    print("\n" + "="*60 + "\nGENERATION TEST\n" + "="*60)
    prompt = torch.tensor([0, 1, 2, 3], device=device)
    gen_a = model_a.generate(prompt, max_tokens=32, temperature=0.8)
    gen_b = model_b.generate(prompt, max_tokens=32, temperature=0.8)
    print(f"\nA: {gen_a}\nLoop: {'YES ✗' if check_consecutive_tokens(gen_a) else 'NO ✓'}")
    print(f"\nB: {gen_b}\nLoop: {'YES ✗' if check_consecutive_tokens(gen_b) else 'NO ✓'}")
    
    print("\n" + "="*60 + "\nCONCLUSION\n" + "="*60)
    success_count = 0
    if delta_b > 0.1:
        success_count += 1
        print("✓ 状態遷移量：十分大きい")
    else:
        print("✗ 状態遷移量：小さい")
    if norm_b < 1e-5 and not check_consecutive_tokens(gen_b):
        success_count += 1
        print("✓ 軌道崩壊：なし")
    else:
        print("✗ 軌道崩壊：あり")
    if final_b['token_frequencies']['kanji'] < final_a['token_frequencies']['kanji']:
        success_count += 1
        print("✓ B 分離効果：漢字偏り軽減")
    else:
        print("✗ B 分離効果：漢字偏り軽減せず")
    if var_b > var_a:
        success_count += 1
        print("✓ 軌道多様性：B の方が高い")
    else:
        print("✗ 軌道多様性：B の方が低い")
    if history_b[-1]['loss'] < history_b[0]['loss']:
        success_count += 1
        print("✓ 損失改善：B で確認")
    else:
        print("✗ 損失改善：B で確認できず")
    
    print(f"\nTotal: {success_count}/5 checks passed")
    if success_count >= 4:
        print("\n🎉 SUCCESS: B 分離の有効性が実証されました！")
    else:
        print("\n⚠️  理論の再検討が必要です")
    return success_count, 5

if __name__ == "__main__":
    main()
