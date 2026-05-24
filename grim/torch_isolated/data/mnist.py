"""MNIST → トークン列（Phase 1 検証）。"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class MNISTTokenDataset(Dataset):
    """各行を1トークン（0..255）に離散化した系列 [28]"""

    def __init__(self, root: str, train: bool, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size
        self.ds = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, label = self.ds[idx]
        row_means = img.squeeze(0).mean(dim=1)
        tokens = (row_means * (self.vocab_size - 2)).long().clamp(0, self.vocab_size - 3) + 2
        return tokens, label


def get_mnist_loaders(
    root: str = "data",
    batch_size: int = 32,
    vocab_size: int = 256,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_ds = MNISTTokenDataset(root, train=True, vocab_size=vocab_size)
    val_ds = MNISTTokenDataset(root, train=False, vocab_size=vocab_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
