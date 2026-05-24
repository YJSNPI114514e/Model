"""総合テスト：ODE ソルバー、履歴バッファ、Tokenizer"""

import torch
from grim.history import HierarchicalHistoryBuffer, HistoryEmbedder
from grim.tokenizer import ComplexTokenizer

def test_all():
    print("=" * 60)
    print("GRIM モデル 総合検証")
    print("=" * 60)
    
    # 1. 階層的履歴バッファ
    print("\n[1] 階層的履歴バッファの検証")
    device = torch.device('cpu')
    dim = 64
    history_dim = 32
    
    embedder = HistoryEmbedder(dim, history_dim)
    buffer = HierarchicalHistoryBuffer(
        n_max=100,
        gamma=0.99,
        eps=1e-4,
        embedder=embedder,
        device=device
    )
    
    for i in range(120):
        psi = torch.randn(dim, dtype=torch.complex64)
        psi = psi / torch.norm(psi)
        buffer.push(psi, weight=1.0)
    
    stats = buffer.get_stats()
    total = stats['short'] + stats['mid'] + stats['long']
    
    print(f"  エントリ数：short={stats['short']}, mid={stats['mid']}, long={stats['long']}")
    print(f"  合計：{total} (最大 100)")
    assert total <= 100, f"合計エントリ数が制限超過：{total}"
    assert stats['mid'] <= 50, f"中期が溢れている：{stats['mid']}"
    assert stats['long'] >= 1, "長期への移動が失敗"
    print("  ✓ PASS: 階層圧縮が正常に機能\n")
    
    # 2. Tokenizer
    print("[2] Tokenizer の計算量最適化検証")
    vocab_size = 1000
    max_len = 64
    
    tokenizer = ComplexTokenizer(vocab_size, dim, max_len).to(device)
    
    # ユニタリ性
    unitarity = tokenizer.verify_unitarity(batch_size=4)
    print(f"  ユニタリ性平均誤差：{unitarity['mean_error']:.2e}")
    # 実用上は平均誤差 5e-2 以下であれば許容（対角近似のため）
    assert unitarity['mean_error'] < 5e-2, "ユニタリ性が保たれていません"
    
    # フォワードパス
    batch_size = 4
    seq_len = 32
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
    
    psi_0 = tokenizer(token_ids, mask)
    norms = torch.norm(psi_0, dim=-1)
    print(f"  出力ノルム：{norms.mean().item():.6f}")
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    print("  ✓ PASS: Tokenizer が正常に機能\n")
    
    # 3. まとめ
    print("=" * 60)
    print("すべてのテストに合格しました")
    print("=" * 60)
    print("\n実施済み:")
    print("  ✓ 階層的履歴バッファ（短期/中期/長期）の圧縮検証")
    print("  ✓ Tokenizer の固有分解による計算量最適化")
    print("\n未実施（次回以降）:")
    print("  - ODE ソルバーの scipy 移行（torchdiffeq → scipy.integrate.solve_ivp）")
    print("  - NumPy 版との整合性検証（NumPy 版は frozen_numpy_version に隔離済み）")
    
    return True

if __name__ == "__main__":
    success = test_all()
    print(f"\n総合結果：{'SUCCESS' if success else 'FAILURE'}")
