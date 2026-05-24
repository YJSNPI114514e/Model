"""ユーティリティ。"""

from __future__ import annotations

import torch

from grim.config import GRIMConfig


def resolve_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def config_from_checkpoint(ckpt: dict) -> GRIMConfig:
    cfg = ckpt.get("config")
    if isinstance(cfg, GRIMConfig):
        return cfg
    if isinstance(cfg, dict):
        return GRIMConfig(**cfg)
    return GRIMConfig()
