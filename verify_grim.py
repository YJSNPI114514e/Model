"""GRIM 数値検証スクリプト：Wirtinger 微分の正当性とユニタリ性のモニタリング"""

import torch
import torch.nn.functional as F
from torch import Tensor

# GRIM モジュールから必要なものをインポート
from grim.flow_field import EnergyVectorField
from grim.geometry import complex_inner, normalize_state, tangent_project
from grim.tokenizer import ComplexTokenizer
from grim.observation import GenerationHead


class DummyHistoryEntry:
    def __init__(self, psi):
        self.psi = psi
        self.weight = 1.0


class DummyHistory:
    def __init__(self, dim, device, num_entries=3):
        self._entries = [
            DummyHistoryEntry(normalize_state(torch.randn(1, dim, dtype=torch.cfloat, device=device)))
            for _ in range(num_entries)
        ]


def verify_wirtinger_gradient(
    dim: int = 16,
    vocab_size: int = 32,
    eps: float = 1e-5,
    seed: int = 42
) -> dict:
    """
    検証 1: Wirtinger 微分の正当性確認
    
    エネルギー勾配が複素空間で正しく計算されているかを、
    autograd 勾配と数値微分勾配を比較して検証する。
    """
    torch.manual_seed(seed)
    device = torch.device('cpu')
    
    # テスト用コンポーネント作成
    tokenizer = ComplexTokenizer(vocab_size=vocab_size, dim=dim, max_len=8, w_alpha=0.1)
    dummy_history = DummyHistory(dim, device, num_entries=3)
    generation = GenerationHead(dim, tokenizer)
    
    flow_field = EnergyVectorField(
        dim=dim,
        hidden=dim * 2,
        history_dim=dim,
        tokenizer=tokenizer,
        history_getter=lambda: dummy_history,
        generation=generation,
    )
    
    # ランダムな初期状態ψ（正規化済み）
    psi_raw = torch.randn(1, dim, dtype=torch.cfloat, device=device)
    psi = normalize_state(psi_raw)
    psi0 = normalize_state(torch.randn(1, dim, dtype=torch.cfloat, device=device))
    
    results = {
        'overall': {'error': 0.0, 'status': 'OK'},
        'by_term': {}
    }
    
    # --- 全体エネルギーの勾配検証 ---
    def compute_energy(psi_c: Tensor) -> Tensor:
        return flow_field.energy(psi_c, psi0)
    
    # autograd 勾配
    psi_test = psi.clone().detach().requires_grad_(True)
    E_auto = compute_energy(psi_test).sum()
    grad_auto = torch.autograd.grad(E_auto, psi_test)[0]
    
    # 数値微分勾配
    grad_num = torch.zeros_like(psi_test)
    for d in range(dim):
        # 実部方向
        psi_plus_re = psi_test.clone()
        psi_plus_re[0, d] += eps
        psi_minus_re = psi_test.clone()
        psi_minus_re[0, d] -= eps
        dE_re = (compute_energy(psi_plus_re).sum() - compute_energy(psi_minus_re).sum()) / (2 * eps)
        
        # 虚部方向
        psi_plus_im = psi_test.clone()
        psi_plus_im[0, d] += eps * 1j
        psi_minus_im = psi_test.clone()
        psi_minus_im[0, d] -= eps * 1j
        dE_im = (compute_energy(psi_plus_im).sum() - compute_energy(psi_minus_im).sum()) / (2 * eps)
        
        grad_num[0, d] = dE_re + 1j * dE_im
    
    # 相対誤差計算
    diff = torch.abs(grad_auto - grad_num)
    norm_sum = torch.abs(grad_auto) + torch.abs(grad_num)
    relative_error = diff / (norm_sum + 1e-12)
    overall_error = relative_error.mean().item()
    
    results['overall']['error'] = overall_error
    if overall_error < 1e-4:
        results['overall']['status'] = '問題なし'
    elif overall_error < 1e-2:
        results['overall']['status'] = '注意'
    else:
        results['overall']['status'] = '要修正'
    
    # --- 各項ごとの勾配検証 ---
    term_names = ['E_world', 'E_self', 'E_cont', 'E_explore']
    
    def compute_term_energy(psi_c: Tensor, psi0: Tensor, term: str) -> Tensor:
        overlap = complex_inner(psi_c, psi0, dim=-1)
        overlap_sq = torch.abs(overlap)**2
        
        if term == 'E_world':
            e_world = -torch.log(overlap_sq.clamp_min(1e-8))
            return e_world.sum()
        
        elif term == 'E_self':
            lam = F.softplus(flow_field.raw_lam)
            e_norm = torch.sum(torch.abs(psi_c)**2, dim=-1)
            return (lam * e_norm).sum()
        
        elif term == 'E_cont':
            entries = dummy_history._entries
            psis = torch.stack([e.psi.squeeze(0) for e in entries], dim=0)
            weights = torch.tensor([e.weight for e in entries], device=psi_c.device)
            overlaps = torch.mm(psi_c, psis.conj().T)
            abs_overlap = torch.abs(overlaps).clamp(0.0, 1.0)
            d = torch.acos(abs_overlap)
            sigma = F.softplus(flow_field.raw_sigma).clamp_min(1e-8)
            kernel = torch.exp(- (d**2) / (2 * sigma**2))
            mu = F.softplus(flow_field.raw_mu)
            e_cont = - (mu / len(entries)) * torch.sum(weights.unsqueeze(0) * kernel, dim=-1)
            return e_cont.sum()
        
        elif term == 'E_explore':
            probs = generation.born_probs(psi_c)
            beta = F.softplus(flow_field.raw_beta)
            e_explore = - beta * torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            return e_explore.sum()
        
        return torch.tensor(0.0, device=psi_c.device)
    
    for term in term_names:
        psi_term = psi.clone().detach().requires_grad_(True)
        E_term = compute_term_energy(psi_term, psi0, term)
        
        try:
            grad_auto_term = torch.autograd.grad(E_term, psi_term, retain_graph=True)[0]
        except Exception as e:
            results['by_term'][term] = {'error': float('inf'), 'status': f'エラー：{str(e)}'}
            continue
        
        grad_num_term = torch.zeros_like(psi_term)
        for d in range(dim):
            psi_plus_re = psi_term.clone()
            psi_plus_re[0, d] += eps
            psi_minus_re = psi_term.clone()
            psi_minus_re[0, d] -= eps
            dE_re = (compute_term_energy(psi_plus_re, psi0, term) - compute_term_energy(psi_minus_re, psi0, term)) / (2 * eps)
            
            psi_plus_im = psi_term.clone()
            psi_plus_im[0, d] += eps * 1j
            psi_minus_im = psi_term.clone()
            psi_minus_im[0, d] -= eps * 1j
            dE_im = (compute_term_energy(psi_plus_im, psi0, term) - compute_term_energy(psi_minus_im, psi0, term)) / (2 * eps)
            
            grad_num_term[0, d] = dE_re + 1j * dE_im
        
        diff_term = torch.abs(grad_auto_term - grad_num_term)
        norm_sum_term = torch.abs(grad_auto_term) + torch.abs(grad_num_term)
        rel_err_term = diff_term / (norm_sum_term + 1e-12)
        term_error = rel_err_term.mean().item()
        
        if term_error < 1e-4:
            status = '問題なし'
        elif term_error < 1e-2:
            status = '注意'
        else:
            status = '要修正'
        
        results['by_term'][term] = {'error': term_error, 'status': status}
    
    return results


def verify_unitarity(
    dim: int = 64,
    vocab_size: int = 100,
    num_steps: int = 50,
    seed: int = 42
) -> dict:
    """
    検証 2: ユニタリ性のモニタリング
    
    ODE 積分中のノルム保存がどの程度保たれているかを定量化する。
    """
    torch.manual_seed(seed)
    device = torch.device('cpu')
    
    # テスト用コンポーネント作成
    tokenizer = ComplexTokenizer(vocab_size=vocab_size, dim=dim, max_len=16, w_alpha=0.1)
    dummy_history = DummyHistory(dim, device, num_entries=3)
    generation = GenerationHead(dim, tokenizer)
    
    flow_field = EnergyVectorField(
        dim=dim,
        hidden=dim * 2,
        history_dim=dim,
        tokenizer=tokenizer,
        history_getter=lambda: dummy_history,
        generation=generation,
    )
    
    # 初期状態
    psi0 = normalize_state(torch.randn(1, dim, dtype=torch.cfloat, device=device))
    
    # ODE 積分中のノルムを追跡するための簡易オイラー法
    norms = []
    psi = psi0.clone()
    dt = 0.02  # 50 ステップで t=1 まで
    
    for step in range(num_steps):
        norm = torch.linalg.vector_norm(psi, dim=-1).item()
        norms.append(norm)
        
        # 勾配計算
        psi_r = psi.real.detach().requires_grad_(True)
        psi_i = psi.imag.detach().requires_grad_(True)
        psi_c = torch.complex(psi_r, psi_i)
        
        E = flow_field.energy(psi_c, psi0).sum()
        grad_r, grad_i = torch.autograd.grad(E, [psi_r, psi_i], create_graph=False)
        v = torch.complex(-grad_r, -grad_i)
        v_proj = tangent_project(v, psi)
        
        # オイラー更新
        psi = psi + dt * v_proj
    
    # 最終ノルム
    final_norm = torch.linalg.vector_norm(psi, dim=-1).item()
    norms.append(final_norm)
    
    # 指標計算
    deviations = [abs(n - 1.0) for n in norms]
    max_deviation = max(deviations)
    mean_deviation = sum(deviations) / len(deviations)
    final_deviation = deviations[-1]
    
    results = {
        'max_deviation': max_deviation,
        'mean_deviation': mean_deviation,
        'final_deviation': final_deviation,
        'num_steps': num_steps,
    }
    
    if max_deviation < 1e-4:
        results['status'] = '問題なし'
    elif max_deviation < 1e-2:
        results['status'] = '許容範囲'
    else:
        results['status'] = '要修正'
    
    return results


def main():
    print("=" * 70)
    print("GRIM 数値検証レポート")
    print("=" * 70)
    
    # 検証 1: Wirtinger 微分
    print("\n=== 検証 1: Wirtinger 微分の正当性確認 ===\n")
    wirtinger_results = verify_wirtinger_gradient(dim=16, vocab_size=32)
    
    print(f"全体の相対誤差：{wirtinger_results['overall']['error']:.6e}")
    print(f"判定：{wirtinger_results['overall']['status']}")
    print("\n項ごとの詳細:")
    for term, data in wirtinger_results['by_term'].items():
        print(f"  {term}: 誤差={data['error']:.6e}, 判定={data['status']}")
    
    # 検証 2: ユニタリ性
    print("\n=== 検証 2: ユニタリ性のモニタリング ===\n")
    unitarity_results = verify_unitarity(dim=64, vocab_size=100, num_steps=50)
    
    print(f"最大逸脱：{unitarity_results['max_deviation']:.6e}")
    print(f"平均逸脱：{unitarity_results['mean_deviation']:.6e}")
    print(f"最終逸脱：{unitarity_results['final_deviation']:.6e}")
    print(f"判定：{unitarity_results['status']}")
    
    # 総合評価
    print("\n" + "=" * 70)
    print("総合評価")
    print("=" * 70)
    
    issues = []
    if wirtinger_results['overall']['status'] == '要修正':
        issues.append("Wirtinger 微分に重大な問題あり")
    elif wirtinger_results['overall']['status'] == '注意':
        issues.append("Wirtinger 微分に軽微な問題あり")
    
    if unitarity_results['status'] == '要修正':
        issues.append("ユニタリ性に重大な問題あり")
    elif unitarity_results['status'] == '許容範囲':
        issues.append("ユニタリ性は許容範囲内")
    
    if not issues:
        print("✓ すべての検証に合格しました。")
    else:
        print("⚠ 以下の問題が検出されました:")
        for issue in issues:
            print(f"  - {issue}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
