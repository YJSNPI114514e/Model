import torch
from grim.config import GRIMConfig
from grim.model import GRIM

config = GRIMConfig(V=100, D=64)
model = GRIM(config)

for name, p in model.named_parameters():
    if p.is_complex():
        print(f"Complex parameter found: {name}, dtype={p.dtype}")
