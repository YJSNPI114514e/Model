"""ハイパーパラメータ（sekkeisyo.txt 準拠）。"""

from dataclasses import dataclass, field


@dataclass
class GRIMConfig:
    # タスク: "lm" = 自然言語（次トークン）, "classify" = 観測基底分類
    task_mode: str = "lm"

    # 状態空間
    D: int = 1024
    V: int = 256
    K: int = 10
    M_max: int = 128
    seq_len: int = 64

    # 履歴バッファ
    N_max: int = 500
    gamma: float = 0.97
    history_eps: float = 1e-4
    D_h: int = 128

    flow_hidden: int = 256
    flow_layers: int = 3
    num_flow_steps: int = 8  # Residual Flow の段数（K）。大きいほど精密、小さいほど高速

    # 非推奨：ODE ソルバー関連パラメータ（互換性のため残存）
    ode_solver: str = "dopri5"          # deprecated
    ode_method: str = "dopri5"          # deprecated
    ode_rtol: float = 1e-4              # deprecated
    ode_atol: float = 1e-6              # deprecated

    # 学習
    lr: float = 0.002
    meta_lr: float = 0.001
    alpha_fm: float = 0.5
    beta_kl: float = 0.01
    k3_interval: int = 100              # sekkeisyo: K3 updates every 100 steps
    
    # K=3 カーネルリッジ回帰メタ学習（NumPy 版）
    use_k3_kernel_ridge: bool = False   # True でカーネルリッジ回帰による K=3 更新
    k3_krr_gamma: float = 0.01          # KL 正則化強度γ
    k3_krr_update_interval: int = 3     # 更新間隔（エポック数）
    k3_krr_smoothing: float = 0.3       # 移動平均係数
    k3_krr_max_buffer: int = 30         # バッファ最大サイズ
    
    batch_size: int = 32
    epochs: int = 10
    grad_clip: float = 1.0
    use_natural_grad: bool = True       # sekkeisyo: K2 uses Natural Gradient (KFAC)
    kfac_damping: float = 1e-3

    # 埋め込み重み
    w_alpha: float = 1.0

    # 生成
    eos_id: int = 1
    pad_id: int = 0
    temperature: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.25
    use_sliding_context: bool = True
    expected_mix_coeff: float = 0.1  # 期待値埋め込みの混合係数（逐次 Born 観測の代替）

    device: str = "cpu"

    def apply_fast_preset(self) -> "GRIMConfig":
        """速度優先: 小型モデル + 短い履歴（ODE は常に DOPRI5）"""
        self.D = 1024
        self.D_h = 128
        self.flow_hidden = 256
        self.flow_layers = 2
        # sekkeisyo VIOLATION 6: Euler solver is FORBIDDEN
        self.ode_solver = "dopri5"
        self.N_max = 24
        self.seq_len = 48
        self.M_max = 48
        self.use_natural_grad = True    # K2 always uses Natural Gradient
        self.k3_interval = 100          # K3 updates every 100 steps
        self.lr = 0.002
        self.batch_size = 16
        return self

    def __post_init__(self) -> None:
        if self.task_mode not in ("lm", "classify"):
            raise ValueError('task_mode must be "lm" or "classify"')
        if self.task_mode == "classify" and self.K > self.D:
            raise ValueError(f"K ({self.K}) must be <= D ({self.D}) for orthonormal obs basis")
