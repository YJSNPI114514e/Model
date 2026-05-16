# GRIM-NLP 実装仕様書

本ドキュメントは、Geometric RKHS Integrative Model (GRIM) を用いた自然言語処理プロジェクト `grim-nlp` の実装仕様をまとめたものです。本プロジェクトは Python パッケージとして構造化されています。

## 1. プロジェクト概要
GRIM は、幾何学的な再生核ヒルベルト空間 (RKHS) 上で状態遷移を定義する、新しい形式の自己回帰生成モデルです。
状態は多様体上の点として表現され、トークンの生成や学習は Flow Matching と常微分方程式 (ODE) に基づいて行われます。

## 2. パッケージ構造
プロジェクトは以下の構造で pip パッケージ化されています。

```text
grim-nlp/
├── pyproject.toml      # パッケージ定義・依存関係
├── grim/               # コアライブラリ
│   ├── model.py        # メインモデル (GRIM クラス)
│   ├── config.py       # ハイパーパラメータ定義 (GRIMConfig)
│   ├── training.py     # 学習ループ・損失関数
│   ├── data/           # データ処理 (ローカル・Hugging Face 対応)
│   └── ...             # 各種幾何学演算・サブモジュール
└── scripts/            # 実行用スクリプト (エントリポイント)
    ├── train.py        # 学習用 CLI
    ├── webui.py        # Gradio ベースの Web UI
    └── generate.py     # テキスト生成 CLI
```

## 3. 主要コンポーネント

### 3.1 GRIMConfig (`grim.config`)
モデルの全てのハイパーパラメータを管理するデータクラスです。
- `D`: 状態空間の次元 (既定: 512)
- `V`: 語彙サイズ
- `ode_solver`: ODE 解法 (`euler`, `dopri5`)
- `apply_fast_preset()`: 低リソース環境向けに設定を一括変更するメソッド

### 3.2 GRIM (`grim.model`)
PyTorch の `nn.Module` を継承したメインモデルです。
- `forward_train_lm`: 言語モデルとしての学習 (Flow Matching 損失 + 観測損失)
- `generate`: 自己回帰的なテキスト生成 (温度調節、Top-k、繰り返し抑制対応)
- `integrate`: 初期状態から目標状態への遷移計算

### 3.3 データハンドリング (`grim.data`)
- `TextCorpus`: ローカルのテキストファイルを読み込み、語彙の構築とトークナイズを行います。
- `hf_dataset`: Hugging Face の `datasets` ライブラリを統合し、リモートデータのストリーミング読み込みに対応しています。

## 4. CLI / エントリポイント
パッケージをインストールすると、以下のコマンドが直接利用可能になります。

| コマンド | 説明 | 主なオプション |
| :--- | :--- | :--- |
| `grim-train` | モデルの学習 | `--dataset`, `--epochs`, `--fast`, `--amp` |
| `grim-webui` | Web UI (Gradio) の起動 | なし |
| `grim-generate` | CLI からのテキスト生成 | `--prompt`, `--checkpoint`, `--temperature` |

## 5. インストールと実行

### 開発モードでのインストール
```bash
pip install -e .
```

### Google Colab での利用
```python
!pip install git+<リポジトリURL>
!grim-train --dataset "tiny_shakespeare" --fast
```

## 6. 特徴的なアルゴリズム
- **Flow Matching**: 状態空間上の軌跡を ODE として学習し、決定論的または確率的な遷移を実現します。
- **Born's Rule**: 量子力学的な観測に基づき、状態ベクトルと埋め込みベクトルの内積から生成確率を算出します。
- **Complex Tokenizer**: 位相情報を保持した埋め込み表現を使用します。
