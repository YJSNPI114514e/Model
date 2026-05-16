"""ハイパーパラメータ（tekisuto.txt 第5節）。"""

from dataclasses import dataclass, field


@dataclass
class GRIMConfig:
    # タスク: "lm" = 自然言語（次トークン）, "classify" = 観測基底分類
    task_mode: str = "lm"

    # 状態空間
    D: int = 512
    V: int = 256
    K: int = 10
    M_max: int = 128
    seq_len: int = 64

    # 履歴バッファ
    N_max: int = 500
    gamma: float = 0.97
    history_eps: float = 1e-4
    D_h: int = 128

    # Flow Matching / ODE  ("euler" が速い, "dopri5" が高精度)
    flow_hidden: int = 256
    flow_layers: int = 3
    ode_solver: str = "euler"
    euler_steps: int = 8
    ode_method: str = "dopri5"
    ode_rtol: float = 1e-5
    ode_atol: float = 1e-7

    # 学習
    lr: float = 0.002
    meta_lr: float = 0.001
    alpha_fm: float = 0.5
    beta_kl: float = 0.01
    k3_interval: int = 200
    batch_size: int = 32
    epochs: int = 10
    grad_clip: float = 1.0
    use_natural_grad: bool = False
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

    device: str = "cpu"

    def apply_fast_preset(self) -> "GRIMConfig":
        """速度優先: 小型モデル + Euler 積分 + 履歴短縮"""
        self.D = 128
        self.D_h = 64
        self.flow_hidden = 128
        self.flow_layers = 2
        self.ode_solver = "euler"
        self.euler_steps = 6
        self.N_max = 24
        self.seq_len = 48
        self.M_max = 48
        self.use_natural_grad = False
        self.k3_interval = 10_000
        self.lr = 0.002
        self.batch_size = 16
        return self

    def __post_init__(self) -> None:
        if self.task_mode not in ("lm", "classify"):
            raise ValueError('task_mode must be "lm" or "classify"')
        if self.task_mode == "classify" and self.K > self.D:
            raise ValueError(f"K ({self.K}) must be <= D ({self.D}) for orthonormal obs basis")
