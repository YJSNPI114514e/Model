"""
tokenizer と重みの詳細分析スクリプト
漢字偏りの原因を究明する
"""
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os

os.makedirs('outputs', exist_ok=True)

from grim.model import GRIM
from grim.data.text import CharVocab
from grim.config import GRIMConfig

def analyze_tokenizer_and_weights():
    print("=" * 60)
    print("Tokenizer と重みの詳細分析")
    print("=" * 60)
    
    # データと語彙の読み込み
    corpus_text = open('data/sample_corpus.txt', 'r', encoding='utf-8').read()
    vocab = CharVocab()
    vocab.build([corpus_text])  # build は list[str] を受け取る
    
    print(f"\n語彙サイズ：{vocab.size}")
    
    # 文字種ごとのトークン数をカウント
    char_counts = defaultdict(int)
    char_types = {}
    
    for i, char in vocab.id2char.items():
        if '\u4e00' <= char <= '\u9fff':
            ctype = 'kanji'
        elif '\u3040' <= char <= '\u309f':
            ctype = 'hiragana'
        elif '\u30a0' <= char <= '\u30ff':
            ctype = 'katakana'
        else:
            ctype = 'other'
        char_counts[ctype] += 1
        char_types[char] = ctype
    
    print("\n【語彙の文字種内訳】")
    type_names = {'kanji': '漢字', 'hiragana': 'ひらがな', 'katakana': 'カタカナ', 'other': 'その他'}
    for ctype, count in char_counts.items():
        print(f"  {type_names[ctype]:8s}: {count:4d} トークン ({count/vocab.size*100:.1f}%)")
    
    # 学習データ中の文字頻度を計算
    print("\n【学習データ中の文字出現頻度 Top 20】")
    from collections import Counter
    char_freq = Counter(corpus_text)
    top_20 = char_freq.most_common(20)
    
    kanji_count_in_top = sum(1 for c, _ in top_20 if '\u4e00' <= c <= '\u9fff')
    hira_count_in_top = sum(1 for c, _ in top_20 if '\u3040' <= c <= '\u309f')
    print(f"  Top 20 中 漢字：{kanji_count_in_top}, ひらがな：{hira_count_in_top}")
    for char, count in top_20:
        ctype = char_types.get(char, 'other')
        marker = '★' if ctype == 'kanji' else '☆'
        print(f"  {marker} '{char}' ({type_names.get(ctype, '?'):4s}): {count:5d} 回")
    
    # モデルの設定と初期化
    config = GRIMConfig(
        V=vocab.size,
        D=64,
        M_max=128,
        N_max=16,
        seq_len=64,
        task_mode="lm",
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRIM(config).to(device)
    
    # 保存された重みがあれば読み込む
    model_path = 'outputs/model_latest.pth'
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\n✓ 保存された重み ({model_path}) を読み込みました")
        except Exception as e:
            print(f"\n✗ 重み読み込みエラー：{e}")
            print("  初期重みで分析します")
    else:
        print(f"\n⚠ 保存された重みが見つかりません ({model_path})")
        print("  初期重みで分析します")
    
    # Tokenizer の埋め込みを分析
    print("\n" + "=" * 60)
    print("Tokenizer 埋め込みベクトルの分析")
    print("=" * 60)
    
    emb_re = model.tokenizer.emb_re.detach().cpu()
    emb_im = model.tokenizer.emb_im.detach().cpu()
    emb_complex = torch.complex(emb_re, emb_im)
    
    # 各トークンのノルムを計算
    token_norms = []
    for i in range(vocab.size):
        vec = emb_complex[i]
        norm = vec.norm().item()
        char = vocab.id2char[i]
        ctype = char_types.get(char, 'other')
        token_norms.append({
            'idx': i,
            'char': char,
            'type': ctype,
            'norm': norm
        })
    
    # 文字種ごとの統計
    norms_by_type = defaultdict(list)
    for t in token_norms:
        norms_by_type[t['type']].append(t['norm'])
    
    print("\n【埋め込みベクトルのノルム平均（文字種別）】")
    for ctype in ['kanji', 'hiragana', 'katakana', 'other']:
        norms = norms_by_type[ctype]
        if len(norms) > 0:
            avg = np.mean(norms)
            std = np.std(norms)
            min_n = np.min(norms)
            max_n = np.max(norms)
            print(f"  {type_names[ctype]:8s}: 平均={avg:.4f}, 標準偏差={std:.4f}, 範囲=[{min_n:.4f}, {max_n:.4f}]")
    
    # ノルムが大きい Top 10
    print("\n【ノルムが大きい Top 10 トークン】")
    sorted_norms = sorted(token_norms, key=lambda x: x['norm'], reverse=True)[:10]
    for t in sorted_norms:
        print(f"  Rank {t['idx']:3d}: '{t['char']}' ({type_names[t['type']]:4s}) - ノルム={t['norm']:.4f}")
    
    # ノルムが小さい Top 10
    print("\n【ノルムが小さい Top 10 トークン】")
    sorted_norms_low = sorted(token_norms, key=lambda x: x['norm'])[:10]
    for t in sorted_norms_low:
        print(f"  Rank {t['idx']:3d}: '{t['char']}' ({type_names[t['type']]:4s}) - ノルム={t['norm']:.4f}")
    
    # 可視化：ノルム分布
    plt.figure(figsize=(12, 6))
    
    data_to_plot = [norms_by_type[t] for t in ['kanji', 'hiragana', 'katakana', 'other']]
    labels = [type_names[t] for t in ['kanji', 'hiragana', 'katakana', 'other']]
    colors = ['salmon', 'lightblue', 'lightgreen', 'gray']
    
    bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('Embedding Norm Distribution by Character Type', fontsize=14)
    plt.ylabel('Vector Norm', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('outputs/norm_distribution.png', dpi=150)
    print("\n✓ ノルム分布グラフを outputs/norm_distribution.png に保存しました")
    
    # 追加分析：勾配の影響を調べる（MLP 層の重み）
    print("\n" + "=" * 60)
    print("Flow Field (MLP) の重み分析")
    print("=" * 60)
    
    mlp_params = []
    for name, param in model.flow_field.named_parameters():
        if param.requires_grad:
            w = param.detach().cpu()
            mlp_params.append({
                'name': name,
                'mean': w.mean().item(),
                'std': w.std().item(),
                'min': w.min().item(),
                'max': w.max().item(),
                'norm': w.norm().item()
            })
    
    print("\n【MLP 層の重み統計】")
    for p in mlp_params:
        print(f"  {p['name']:30s}: 平均={p['mean']:+.4f}, 標準偏差={p['std']:.4f}, 範囲=[{p['min']:+.4f}, {p['max']:+.4f}]")
    
    # Generation Head の分析
    print("\n" + "=" * 60)
    print("Generation Head の分析")
    print("=" * 60)
    
    gen_head = model.generation
    if hasattr(gen_head, 'basis_states'):
        basis = gen_head.basis_states
        print(f"\n基底状態の数：{basis.shape}")
        
        # 各基底のノルム
        basis_norms = basis.norm(dim=-1).cpu()
        print(f"基底ノルム - 平均：{basis_norms.mean().item():.4f}, 標準偏差：{basis_norms.std().item():.4f}")
        
        # 各トークンへの射影をシミュレーション
        print("\n【各文字種への平均射影強度（初期状態）】")
        # ランダムな初期状態から各トークンへの内積を計算
        psi_init = torch.randn(64, dtype=torch.complex64)
        psi_init = psi_init / psi_init.norm()
        
        projections_by_type = defaultdict(list)
        for i in range(vocab.size):
            emb = model.tokenizer.embeddings[i]
            emb_norm = emb / emb.norm()
            proj = torch.abs(torch.vdot(psi_init, emb_norm)).item()
            ctype = char_types.get(vocab.id2char[i], 'other')
            projections_by_type[ctype].append(proj)
        
        for ctype in ['kanji', 'hiragana', 'katakana', 'other']:
            projs = projections_by_type[ctype]
            if len(projs) > 0:
                avg_proj = np.mean(projs)
                total_prob = sum(p**2 for p in projs)  # ボルン則確率の総和
                print(f"  {type_names[ctype]:8s}: 平均射影={avg_proj:.4f}, 総確率寄与={total_prob:.4f}")
    
    print("\n" + "=" * 60)
    print("分析完了")
    print("=" * 60)
    print("\n【考察のポイント】")
    print("1. 学習データ中に特定の漢字が極端に多く出現していないか？")
    print("2. 学習後、漢字トークンの埋め込みノルムが他より大きくなっていないか？")
    print("3. MLP 層のバイアスが特定方向に偏っていないか？")
    print("4. 生成時の ODE 積分で、漢字トークン方向への収束が速すぎていないか？")

if __name__ == '__main__':
    analyze_tokenizer_and_weights()
