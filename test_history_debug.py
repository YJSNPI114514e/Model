"""階層的履歴バッファのデバッグ"""

import torch
from grim.history import HierarchicalHistoryBuffer, HistoryEmbedder

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

print(f"max_short={buffer.max_short}, max_mid={buffer.max_mid}, max_long={buffer.max_long}")

# 11 エントリ追加
for i in range(11):
    psi = torch.randn(dim, dtype=torch.complex64)
    psi = psi / torch.norm(psi)
    buffer.push(psi, weight=1.0)
    stats = buffer.get_stats()
    if i >= 8:
        print(f"Step {i+1}: short={stats['short']}, mid={stats['mid']}, long={stats['long']}")

print("\nさらに 50 エントリ...")
for i in range(50):
    psi = torch.randn(dim, dtype=torch.complex64)
    psi = psi / torch.norm(psi)
    buffer.push(psi, weight=1.0)
    if i % 10 == 9:
        stats = buffer.get_stats()
        print(f"Step {i+11}: short={stats['short']}, mid={stats['mid']}, long={stats['long']}")

final_stats = buffer.get_stats()
print(f"\n最終：short={final_stats['short']}, mid={final_stats['mid']}, long={final_stats['long']}")
print(f"中期エントリ数：{len(buffer.mid_term)} (max_mid={buffer.max_mid})")
