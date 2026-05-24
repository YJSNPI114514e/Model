# GRIM-NLP(仮称)　

## 概要

GRIM (Geometric RKHS Integrative Model) は、複素射影ヒルベルト空間 $\mathbb{C}P^{D-1}$ 上の状態遷移として言語モデルを定式化する新しい自己回帰生成モデルです。本ドキュメントは、理論的定式化と実装の対応を数式で明示し、将来のモデル更新時に理論と実装の整合性を検証可能にすることを目的とします。

---

## NumPy 版実装状況（2026 年更新）

### ✅ NumPy 移行済みファイル（推論・生成専用）

| ファイル | 内容 | 用途 |
|---------|------|------|
| `grim/geometry_np.py` | 複素内積、FS 距離、正規化、接空間射影 | 幾何学演算 |
| `grim/tokenizer_np.py` | `NumPyTokenizer` クラス | トークン埋め込み・状態構成 |
| `grim/flow_field_np.py` | `NumPyEnergyField` クラス | エネルギー関数と勾配 |
| `grim/ode_solver_np.py` | `integrate_flow` 関数 | scipy.integrate.solve_ivp 使用 |
| `grim/observation_np.py` | `born_probs` 関数 | ボルン則による確率計算 |
| `grim/history_np.py` | `NumPyHistory` クラス | 階層的履歴バッファ |
| `grim/model_np.py` | `NumPyGRIM` クラス | 推論・生成専用モデル |

**特徴:**
- 依存：numpy, scipy.integrate.solve_ivp のみ
- dtype：np.complex128（状態）, np.float64（実数パラメータ）
- 逆伝播（BP）を含まないため、訓練には使用不可
- 推論・生成のみを対象

**動作確認:**
```bash
python test_numpy_grim.py
```
全 7 モジュールのテストに合格しました（2025 年現在）。

### ❌ PyTorch 版のままのファイル（訓練用）

| ファイル | 内容 | 後日移行予定 |
|---------|------|-------------|
| `grim/training.py` | 訓練ループ（BP 使用） | 未定 |
| `grim/natural_grad.py` | KFAC 自然勾配 | 未定 |
| `grim/meta.py` | K=3 メタ学習 | 未定 |
| `scripts/train.py` | 訓練スクリプト | 未定 |
| `data/*.py` | DataLoader | PyTorch のまま併用可 |

---

## Web UI の使用方法

### NumPy モードでの推論

Web UI の「文章生成」タブで **「NumPy モード (推論のみ)」** チェックボックスをオンにすると、PyTorch ではなく NumPy 版モデルを使用してテキストを生成できます。

**特徴:**
- 訓練済みチェックポイントからパラメータを読み取り、NumPy 版で推論
- バックプロパゲーション不使用のため高速
- 実験的な機能として提供

### 学習ログの確認

Web UI の「学習」タブでモデルを訓練すると、**「ログ出力」** テキストボックスに以下の情報がリアルタイムで表示されます：

- 使用デバイス（CPU/GPU）
- データセット情報
- エポック進行状況
- 損失値（loss, loss_obs）
- トークン精度（acc）
- パープレキシティ（ppl）
- チェックポイント保存情報

**起動方法:**
```bash
python scripts/webui.py --port 7860
```

ブラウザで `http://localhost:7860` にアクセスしてください。

---

## 1. 数学的基礎

### 1.1 状態空間

**理論**: 言語状態 $|\psi\rangle$ は複素ヒルベルト空間 $\mathcal{H} \cong \mathbb{C}^D$ 上の単位ベクトル
$$\|\psi\| = 1, \quad |\psi\rangle \in \mathbb{C}P^{D-1}$$

**実装**: `grim/geometry.py::normalize_state()`
```python
def normalize_state(psi: Tensor, eps: float = 1e-8) -> Tensor:
    n = torch.linalg.vector_norm(psi, dim=-1, keepdim=True).clamp_min(eps)
    return psi / n
```

### 1.2 フビニ・スタディ計量

**理論**: 2 状態間の距離
$$d_{FS}(|\psi\rangle, |\phi\rangle) = \arccos|\langle\psi|\phi\rangle|$$

**実装**: `grim/geometry.py::fs_angle()`
```python
def fs_angle(psi0: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    overlap = torch.abs(complex_inner(psi0, target, dim=-1)).clamp(0.0, 1.0 - eps)
    return torch.arccos(overlap)
```

### 1.3 内積

**理論**: 複素内積 $\langle\psi|\phi\rangle = \sum_i \overline{\psi_i}\phi_i$

**実装**: `grim/geometry.py::complex_inner()`
```python
def complex_inner(a: Tensor, b: Tensor, dim: int = -1) -> Tensor:
    return torch.sum(a.conj() * b, dim=dim)
```

---

## 2. トークナイザー（第 2.2 節）

### 2.1 初期状態構成

**理論**: 文脈トークン列 $\{t_1, \ldots, t_L\}$ から初期状態 $|\psi_0\rangle$ を構成
$$|\psi_0\rangle = \mathcal{Z}^{-1/2} \sum_{j=1}^{L} \alpha_j e^{i\varphi_j} |\psi_j\rangle$$

ここで $|\psi_j\rangle$ は状態空間モデルによる逐次更新:
$$|\psi_j\rangle = \hat{A}_j |\psi_{j-1}\rangle + \hat{B} |e_{t_j}\rangle$$

- $\hat{A}_j$: 入力依存ユニタリ遷移作用素
- $\hat{B}$: 入力作用素
- $|e_{t_j}\rangle$: トークン埋め込み
- $\varphi_j = w_\phi \cdot j / M_{\max} + b_\phi$: 位置符号
- $\alpha_j$: 注意重み（softmax 正規化）

**実装**: `grim/tokenizer.py::ComplexTokenizer::forward()`
```python
def forward(self, token_ids: Tensor, mask: Tensor | None = None) -> Tensor:
    # ...
    for j in range(L):
        A_j = self._compute_selective_unitary(psi)  # Â_j
        psi = einsum('bxy,by->bx', A_j, psi) + einsum('xy,by->bx', B_mat, e_j)
    
    # 重ね合わせ
    psi_0 = sum(alpha_j * exp(i*phi_j) * psi_j)
    return normalize_state(psi_0)
```

### 2.2 選択的ユニタリ作用素

**理論**: 
$$\hat{A}_j = \exp(X_{\text{base}} + g_j \Delta)$$
$$g_j = \sigma(|\langle\psi_{j-1}|\delta\rangle|^2 - 0.5)$$

ここで $X_{\text{base}}$ は歪エルミート行列 ($X^\dagger = -X$)、$\Delta$ は状態依存補正

**実装**: `grim/tokenizer.py::_compute_selective_unitary()`
```python
def _compute_selective_unitary(self, psi_prev: Tensor) -> Tensor:
    X_base = self._build_skew_hermitian(self.A_base_re, self.A_base_im)
    score = |inner_prod(psi_prev, delta_raw)|^2
    gate = sigmoid(score - 0.5)
    X_j = X_base + gate * delta
    return matrix_exp(X_j)  # ユニタリ行列
```

---

## 3. 動的履歴バッファ（第 2.5 節）

### 3.1 履歴表現

**理論**: 履歴バッファ $H = \{(|\psi^{(i)}\rangle, w_i)\}_{i=1}^N$
- 各ステップで $|\psi_T\rangle$ を保存
- 重み減衰: $w_i \leftarrow \gamma w_i$
- 容量制限: $N \leq N_{\max}$

**実装**: `grim/history.py::HistoryBuffer`
```python
def decay(self) -> None:
    for e in self._entries:
        w = e.weight * self.gamma
        if w >= self.eps:
            kept.append(HistoryEntry(e.psi, w))

def push(self, psi: Tensor, weight: float = 1.0) -> None:
    self._entries.append(HistoryEntry(psi.detach().clone(), weight))
```

### 3.2 履歴埋め込み

**理論**: 履歴要約ベクトル
$$h_{\text{emb}} = \sum_{i} w_i \cdot \kappa_{\text{emb}}(|\psi^{(i)}\rangle)$$
$$\kappa_{\text{emb}}: \mathbb{C}^D \to \mathbb{R}^{D_h}$$

**実装**: `grim/history.py::HistoryBuffer::summarize()`
```python
def summarize(self, batch_size: int = 1) -> Tensor:
    psis = stack([e.psi for e in entries])
    weights = softmax([e.weight for e in entries])
    emb = self.embedder(psis)  # κ_emb
    return sum(weights * emb)
```

---

## 4. ベクトル場とエネルギー関数（第 2.3 節）

### 4.1 エネルギー関数

**理論**: 全エネルギー
$$E(\psi) = E_{\text{world}} + E_{\text{self}} + E_{\text{cont}} + E_{\text{explore}}$$

#### (a) 外部入力との整合性
$$E_{\text{world}} = -\log|\langle\psi|\psi_0\rangle|^2$$

**実装**: `grim/flow_field.py::energy_and_gradient()`
```python
overlap = complex_inner(psi, psi0)
e_world = -log(|overlap|^2 + eps)
grad_world = -conj(overlap) / |overlap|^2 * psi0
```

#### (b) 自己無撞着性
$$E_{\text{self}} = \lambda \|\psi\|^2$$

**実装**:
```python
lam = softplus(self.raw_lam)
e_self = lam * sum(|psi|^2)
grad_self = lam * psi
```

#### (c) 連続性（履歴との整合）
$$E_{\text{cont}} = -\frac{\mu}{N} \sum_{i=1}^N w_i \exp\left(-\frac{d_{FS}^2(|\psi\rangle, |\psi^{(i)}\rangle)}{2\sigma^2}\right)$$

近似: $d_{FS} \approx \sqrt{2 - 2|\langle\psi|\psi^{(i)}\rangle|}$

**実装**:
```python
overlaps = mm(psi, psis.conj().T)  # [B, H_len]
dist_approx = sqrt(2 - 2*|overlaps|)
kernel = exp(-dist^2 / (2*sigma^2))
e_cont = -(mu/N) * sum(weights * kernel)
```

#### (d) 探索項
$$E_{\text{explore}} = -\beta H(\psi) = -\beta \sum_k p_k \log p_k$$
$$p_k = \frac{|\langle e_k|\psi\rangle|^2}{\sum_j |\langle e_j|\psi\rangle|^2}$$

**実装**:
```python
probs = generation.born_probs(psi)  # ソフトマックス正規化済み
beta = softplus(self.raw_beta)
e_explore = -beta * sum(probs * log(probs))
```

### 4.2 ベクトル場

**理論**: 状態遷移方程式
$$\frac{d|\psi\rangle}{dt} = -\nabla_{\psi^*} E(\psi)$$

接空間射影:
$$v_{\text{tangent}} = v - \langle\psi|v\rangle|\psi\rangle$$

**実装**: `grim/flow_field.py::forward()`
```python
def forward(self, psi, psi0, h_emb, t):
    E, grad_E = self.energy_and_gradient(psi, psi0)
    v = -grad_E
    return tangent_project(v, psi)  # 接空間へ射影
```

---

## 5. Flow Matching（第 3.1 節）

### 5.1 測地線補間

**理論**: 初期状態 $|\psi_0\rangle$ と目標状態 $|\psi_T\rangle$ の測地線
$$|\psi_t\rangle = \frac{\sin((1-t)\theta)}{\sin\theta}|\psi_0\rangle + \frac{\sin(t\theta)}{\sin\theta}|\psi_T\rangle$$
$$\theta = d_{FS}(|\psi_0\rangle, |\psi_T\rangle)$$

**実装**: `grim/geometry.py::geodesic_interp()`
```python
def geodesic_interp(psi0, target, t):
    theta = fs_angle(psi0, target)
    c0 = sin((1-t)*theta) / sin(theta)
    c1 = sin(t*theta) / sin(theta)
    return normalize_state(c0*psi0 + c1*target)
```

### 5.2 目標速度場

**理論**:
$$v_{\text{target}} = \frac{d|\psi_t\rangle}{dt} = \frac{-\theta\cos((1-t)\theta)}{\sin\theta}|\psi_0\rangle + \frac{\theta\cos(t\theta)}{\sin\theta}|\psi_T\rangle$$

**実装**: `grim/geometry.py::geodesic_velocity_target()`
```python
def geodesic_velocity_target(psi0, target, t, psi_t):
    coef0 = -theta*cos((1-t)*theta)/sin(theta)
    coef1 = theta*cos(t*theta)/sin(theta)
    v = coef0*psi0 + coef1*target
    return tangent_project(v, psi_t)
```

### 5.3 Flow Matching 損失

**理論**:
$$\mathcal{L}_{FM} = \mathbb{E}_{t,\psi_0,\psi_T}\left[\|v_\theta(\psi_t, t) - v_{\text{target}}\|^2_{FS}\right]$$

**実装**: `grim/model.py::flow_matching_loss()`
```python
def flow_matching_loss(self, psi0, target_state, h_emb, t):
    psi_t = geodesic_interp(psi0, target_state, t)
    v_target = geodesic_velocity_target(psi0, target_state, t, psi_t)
    v_pred = self.flow_field(psi_t, psi0, h_emb, t)
    return mean(|v_pred - v_target|^2)
```

---

## 6. 観測と生成（第 2.4 節）

### 6.1 ボルン則

**理論**: 確率
$$p(k) = |\langle e_k|\psi_T\rangle|^2$$

**実装**: `grim/observation.py::GenerationHead::born_probs()`
```python
def born_probs(self, psi: Tensor) -> Tensor:
    emb_norm = normalize_state(self.tokenizer.embeddings)  # [V, D]
    overlaps = mm(psi, emb_norm.conj().T)  # [B, V]
    scores = |overlaps|^2
    return scores / sum(scores)  # ソフトマックス正規化
```

### 6.2 言語モデリング損失

**理論**: ボルン則に基づくクロスエントロピー

確率計算（ボルン則）:
$$p(k) = \frac{|\langle e_k|\psi_T\rangle|^2}{\sum_j |\langle e_j|\psi_T\rangle|^2}$$

クロスエントロピー損失:
$$\mathcal{L}_{LM} = -\log p(y_{\text{true}})$$

**実装**: `grim/model.py::language_modeling_loss()`
```python
def language_modeling_loss(self, psi_T: Tensor, target_token_ids: Tensor) -> Tensor:
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
```

---

## 7. ODE 積分（第 2.3 節）

### 7.1 数値積分

**理論**: 常微分方程式
$$\frac{d|\psi\rangle}{dt} = v(|\psi\rangle, t), \quad t \in [0, 1]$$

**実装要件**: DOPRI5 法必須（Euler 法禁止）

**実装**: `grim/ode_solver.py::integrate_flow()`
```python
def integrate_flow(flow_field, psi0, h_emb, method="dopri5", rtol=1e-4, atol=1e-6):
    solution = odeint(dynamics, y0, t_eval=[0,1], method='dopri5', rtol=rtol, atol=atol)
    psi_T = solution[-1]
    ASSERT: allclose(norm(psi_T), 1.0, atol=1e-5)
    return psi_T
```

---

## 8. メタ学習（第 3.3 節）

### 8.1 メタパラメータ

**理論**: 損失重み
$$\mathcal{L} = f_{\text{softplus}}(w_{\text{obs}}) \cdot \mathcal{L}_{\text{obs}}$$

**実装**: `grim/meta.py::MetaParams`
```python
class MetaParams(nn.Module):
    obs_weight = Parameter(inv_softplus(1.0))  # softplus 適用後≈1.0
    
    def effective_weights(self):
        return {"obs_weight": softplus(self.obs_weight)}
```

### 8.2 更新間隔

**理論**: K2 パラメータ（モデル本体）は毎ステップ更新、K3 パラメータ（メタ重み）は $k_3$ ステップ毎更新

**実装**: `grim/training.py::train_epoch()`
```python
do_meta = ((global_step + 1) % config.k3_interval == 0)
# K2 update: every step
# K3 update: every k3_interval steps
```

---

## 9. 自然勾配（第 3.2 節）

### 9.1 KFAC 近似

**理論**: Fisher 情報行列の Kronecker 積近似
$$F \approx Q \otimes G$$

**実装**: `grim/natural_grad.py::KFACNaturalGradient`
```python
def precondition(self, params):
    # 各層に対して F⁻¹∇θ を計算
    q = (a.T @ a) / B + damping*I  # 活性化共分散
    g = (g.T @ g) / B + damping*I  # 勾配共分散
    scale = trace(q_inv)^0.5 * trace(g_inv)^0.5
    param.grad *= scale
```

---

## 10. 訓練ループ

### 10.1 全体損失

**理論**: 
$$\mathcal{L}_{\text{total}} = f_{\text{softplus}}(w_{\text{obs}}) \cdot \mathcal{L}_{\text{obs}}$$

Flow Matching 損失は無効化（観測損失のみ使用）

**実装**: `grim/model.py::forward_train_lm()`
```python
def forward_train_lm(self, context_ids, target_ids, mask, use_amp):
    psi0 = self.tokenize(context_ids, mask)
    h_emb = self.summarize_history(B)
    psi_T = self.integrate(psi0, h_emb, use_amp)
    L_lm = self.language_modeling_loss(psi_T, target_ids)
    L = softplus(w.obs_weight) * L_lm  # FM 損失は不使用
    return {"loss": L, "loss_obs": L_lm}
```

### 10.2 評価指標

**理論**: 
- トークン精度: $\text{acc} = \frac{1}{N}\sum_i \mathbb{I}[\hat{y}_i = y_i]$
- パープレキシティ: $\text{ppl} = \exp\left(-\frac{1}{N}\sum_i \log p(y_i)\right)$

**実装**: `grim/training.py::evaluate_lm()`
```python
def evaluate_lm(model, loader, device):
    correct += (pred == y).sum()
    total_nll += nll_loss(log_probs, y, reduction="sum")
    acc = correct / total
    ppl = exp(total_nll / total)
```

---

## 11. 生成アルゴリズム

### 11.1 自己回帰生成

**理論**: 
$$|\psi_T^{(t)}\rangle = \text{ODEIntegrate}(|\psi_0^{(t)}\rangle, h_{\text{emb}}^{(t)})$$
$$\hat{y}_t \sim p(y|\psi_T^{(t)})$$
$$|\psi_0^{(t+1)}\rangle = \text{Inject}(|\psi_T^{(t)}\rangle, \hat{y}_t)$$

**実装**: `grim/model.py::generate()`
```python
def generate(self, prompt_ids, max_new_tokens=64):
    for step in range(max_new_tokens):
        psi0 = self.tokenize(context)
        s_T = self.integrate(psi0, h_emb())
        next_id = sample_token(s_T, temperature, top_k, repetition_penalty)
        generated.append(next_id)
        context.append(next_id)
        if not use_sliding_context:
            s = self.tokenizer.inject(s_T, next_id)
```

---

## 12. ハイパーパラメータ

**実装**: `grim/config.py::GRIMConfig`

| パラメータ | 記号 | 既定値 | 説明 |
|-----------|------|--------|------|
| 状態次元 | $D$ | 512 | $\mathbb{C}^D$ の次元 |
| 語彙サイズ | $V$ | 256 | トークン数 |
| 最大系列長 | $M_{\max}$ | 128 | トークナイザー入力 |
| 履歴容量 | $N_{\max}$ | 500 | 履歴バッファ最大サイズ |
| 減衰率 | $\gamma$ | 0.97 | 履歴重み減衰 |
| ODE 許容誤差 | $\text{rtol}, \text{atol}$ | 1e-4, 1e-6 | DOPRI5 精度 |
| 学習率 | $\eta$ | 0.002 | K2 オプティマイザ |
| メタ学習率 | $\eta_{\text{meta}}$ | 0.001 | K3 オプティマイザ |
| KFAC 制動 | $\lambda_{\text{damp}}$ | 1e-3 | 自然勾配正則化 |

---

## 13. 理論と実装の対応表

| 理論概念 | 数式 | 実装ファイル | クラス/関数 |
|---------|------|-------------|------------|
| 状態正規化 | $\|\psi\|=1$ | `geometry.py` | `normalize_state()` |
| 複素内積 | $\langle\psi\|\phi\rangle$ | `geometry.py` | `complex_inner()` |
| FS 距離 | $d_{FS} = \arccos\|\langle\psi\|\phi\rangle\|$ | `geometry.py` | `fs_angle()` |
| 測地線 | $|\psi_t\rangle$ | `geometry.py` | `geodesic_interp()` |
| トークナイザー | $|\psi_0\rangle = \sum \alpha_j e^{i\varphi_j}|\psi_j\rangle$ | `tokenizer.py` | `ComplexTokenizer` |
| エネルギー | $E = E_w + E_s + E_c + E_e$ | `flow_field.py` | `EnergyVectorField` |
| ベクトル場 | $d\psi/dt = -\nabla E$ | `flow_field.py` | `forward()` |
| ODE 積分 | $\psi_T = \int_0^1 v dt$ | `ode_solver.py` | `integrate_flow()` |
| ボルン則 | $p(k) = \|\langle e_k\|\psi\rangle\|^2$ | `observation.py` | `GenerationHead` |
| 履歴 | $H = \{(\psi^{(i)}, w_i)\}$ | `history.py` | `HistoryBuffer` |
| メタ重み | $w_{\text{obs}}, w_{\text{fm}}$ | `meta.py` | `MetaParams` |
| 自然勾配 | $F^{-1}\nabla\theta$ | `natural_grad.py` | `KFACNaturalGradient` |
| 訓練ループ | $\mathcal{L} = f(w)\cdot\mathcal{L}_{obs}$ | `training.py` | `train_epoch()` |
| メインモデル | 統合 | `model.py` | `GRIM` |

---

## 14. 変更履歴と検証ポイント

### モデル更新時のチェックリスト

1. **幾何学演算の整合性**
   - [ ] `normalize_state()` が $\|\psi\|=1$ を保証
   - [ ] `tangent_project()` が $\langle\psi|v_{\text{tangent}}\rangle = 0$ を満たす
   - [ ] `geodesic_interp()` が $t=0$ で $\psi_0$、$t=1$ で $\psi_T$ を返す

2. **エネルギー勾配の正確性**
   - [ ] `energy_and_gradient()` の解析勾配が autograd と一致
   - [ ] 各エネルギー項 $E_{\text{world}}, E_{\text{self}}, E_{\text{cont}}, E_{\text{explore}}$ が定義通り

3. **ODE 積分の精度**
   - [ ] DOPRI5 法を使用（Euler 法ではない）
   - [ ] $\text{rtol}=10^{-4}, \text{atol}=10^{-6}$
   - [ ] 積分後 $\|\psi_T\| \approx 1$（誤差 $10^{-5}$ 以下）

4. **観測の正当性**
   - [ ] Born 確率が $\sum_k p(k) = 1$ を満たす
   - [ ] トークン埋め込みが正規化済み

5. **メタ学習の動作**
   - [ ] K3 パラメータが $k_3$ ステップ毎更新
   - [ ] 重みが softplus で正値制約

---

## 15. インストールと使用方法

```bash
# 開発モードインストール
pip install -e .

# 学習実行
grim-train --dataset your_dataset --epochs 10 --amp

# Web UI 起動
grim-webui

# テキスト生成
grim-generate --prompt "Hello" --checkpoint checkpoints/best.pt
```

---

## 参考文献

1. Flow Matching: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
2. 自然勾配: Amari, "Natural Gradient Works Efficiently in Learning", Neural Computation 1998
3. KFAC: Martens & Grosse, "Optimizing Neural Networks with Kronecker-factored Approximate Curvature", ICML 2015
4. 複素ニューラルネット: Trabelsi et al., "Deep Complex Networks", ICLR 2018
