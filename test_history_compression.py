"""階層的履歴バッファの圧縮検証"""

import torch
from grim.history import HierarchicalHistoryBuffer, HistoryEmbedder

def test_hierarchical_compression():
    """階層圧縮が正しく機能するか検証"""
    
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
    
    print("=== 階層的履歴バッファ圧縮テスト ===\n")
    
    # 短期→中期の圧縮をテスト (11 エントリ追加)
    print("短期メモリに 11 エントリを追加...")
    for i in range(11):
        psi = torch.randn(dim, dtype=torch.complex64)
        psi = psi / torch.norm(psi)
        buffer.push(psi, weight=1.0)
    
    stats = buffer.get_stats()
    print(f"  短期：{stats['short']}, 中期：{stats['mid']}, 長期：{stats['long']}")
    assert stats['short'] <= 10, f"短期メモリが溢れている：{stats['short']}"
    assert stats['mid'] >= 1, f"中期への圧縮が失敗：{stats['mid']}"
    print("  ✓ 短期→中期の圧縮が正常\n")
    
    # 中期→長期の圧縮をテスト (合計 120 エントリで中期が溢れる)
    print("さらに 110 エントリを追加（中期→長期の圧縮）...")
    for i in range(110):
        psi = torch.randn(dim, dtype=torch.complex64)
        psi = psi / torch.norm(psi)
        buffer.push(psi, weight=1.0)
    
    stats = buffer.get_stats()
    print(f"  短期：{stats['short']}, 中期：{stats['mid']}, 長期：{stats['long']}")
    assert stats['mid'] <= 50, f"中期メモリが溢れている：{stats['mid']}"
    assert stats['long'] >= 1, f"長期への移動が失敗：{stats['long']}"
    print("  ✓ 中期→長期の圧縮が正常\n")
    
    # 合計エントリ数の制限を確認
    total_entries = stats['short'] + stats['mid'] + stats['long']
    print(f"総エントリ数：{total_entries}")
    assert total_entries <= 100, f"総エントリ数が制限を超えている：{total_entries}"
    print("  ✓ 総エントリ数が 100 以下に制限されている\n")
    
    # summarize の機能テスト
    print("summarize 機能をテスト...")
    current_psi = torch.randn(1, dim, dtype=torch.complex64)
    current_psi = current_psi / torch.norm(current_psi)
    summary = buffer.summarize(batch_size=1, current_psi=current_psi)
    print(f"  要約ベクトル形状：{summary.shape}")
    assert summary.shape == (1, history_dim), f"形状が不正：{summary.shape}"
    print("  ✓ summarize が正常に機能\n")
    
    # 減衰機能テスト
    print("減衰機能をテスト...")
    initial_total = len(buffer)
    buffer.decay(boundary_prob=0.5)
    after_decay_total = len(buffer)
    print(f"  減衰前：{initial_total}, 減衰後：{after_decay_total}")
    print("  ✓ 減衰機能が実行された\n")
    
    print("=== すべてのテストに合格 ===")
    return True

if __name__ == "__main__":
    success = test_hierarchical_compression()
    print(f"\nテスト結果：{'SUCCESS' if success else 'FAILURE'}")
