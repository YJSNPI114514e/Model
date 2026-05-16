#!/usr/bin/env python3
"""GPU が使えるか確認。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grim.bootstrap import ensure_torch, reexec_in_venv_if_needed

reexec_in_venv_if_needed()
ensure_torch()

import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count", torch.cuda.device_count())
    print("device_name", torch.cuda.get_device_name(0))
    print("cuda_version", torch.version.cuda)
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("matmul_ok", y.shape)
else:
    print("GPU not available. For NVIDIA GPU run: .\\setup_env_gpu.ps1")
    print("Current install may be CPU-only (setup_env.ps1).")
