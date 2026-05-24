"""自然言語コーパス・文字トークナイザ（次トークン言語モデル）。"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class CharVocab:
    PAD, EOS, UNK = 0, 1, 2

    def __init__(self) -> None:
        self.char2id: dict[str, int] = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
        self.id2char: dict[int, str] = {0: "<pad>", 1: "<eos>", 2: "<unk>"}

    def build(self, texts: list[str]) -> None:
        for text in texts:
            for ch in text:
                if ch not in self.char2id:
                    i = len(self.char2id)
                    self.char2id[ch] = i
                    self.id2char[i] = ch

    @property
    def size(self) -> int:
        return len(self.char2id)

    def encode(self, text: str, max_len: int | None = None, *, add_eos: bool = True) -> list[int]:
        ids = [self.char2id.get(ch, self.UNK) for ch in text]
        if add_eos:
            ids.append(self.EOS)
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def encode_prompt(self, text: str, max_len: int | None = None) -> list[int]:
        """生成用: 末尾に EOS を付けない。"""
        return self.encode(text, max_len=max_len, add_eos=False)

    def to_state(self) -> dict:
        return {
            "char2id": dict(self.char2id),
            "id2char": {int(k): v for k, v in self.id2char.items()},
        }

    @classmethod
    def from_state(cls, state: dict) -> "CharVocab":
        vocab = cls()
        vocab.char2id = dict(state["char2id"])
        vocab.id2char = {int(k): v for k, v in state["id2char"].items()}
        return vocab

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        chars = []
        for i in ids:
            if skip_special and i <= self.UNK:
                if i == self.EOS:
                    break
                continue
            ch = self.id2char.get(i, "")
            if not ch and i == self.UNK:
                ch = "?"
            chars.append(ch)
        return "".join(chars)


class TextCorpus:
    DEFAULT_SAMPLE = (
        "意識は場である。GRIMは測地線流で思考する。\n"
        "Flow Matchingはフビニ・スタディ多様体上を進む。\n"
        "ユニタリ性により情報は失われず、文脈から次の文字へと流れる。\n"
        "自然言語処理では、文脈トークン列から次トークンをボルン則で予測する。\n"
        "履歴バッファは長い対話の文脈を要約し、忘却を抑える。\n"
    )

    def __init__(self, path: str | Path | None = None, *, text: str | None = None, source: str = "", vocab: CharVocab | None = None) -> None:
        if text is not None:
            self.text = text
            self.source = source or "inline"
        elif path and Path(path).exists():
            self.text = Path(path).read_text(encoding="utf-8")
            self.source = str(path)
        else:
            self.text = self.DEFAULT_SAMPLE
            self.source = "default"
        
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = CharVocab()
            self.vocab.build([self.text])

    @classmethod
    def from_text(cls, text: str, source: str = "inline", vocab: CharVocab | None = None) -> "TextCorpus":
        return cls(text=text, source=source, vocab=vocab)

    def prepare_for_training(self) -> None:
        """巨大テキストを token id 列に変換し、元文字列を解放する。"""
        if getattr(self, "_ids", None) is not None:
            return
        import gc

        self._ids = self.vocab.encode(self.text)
        self.text = ""
        gc.collect()

    @property
    def token_ids(self) -> list[int]:
        if getattr(self, "_ids", None) is None:
            self.prepare_for_training()
        return self._ids

    def lm_dataset(self, seq_len: int) -> "LanguageModelDataset":
        self.prepare_for_training()
        return LanguageModelDataset(self.token_ids, self.vocab, seq_len)


class LanguageModelDataset(Dataset):
    """文脈 [t_{i}..t_{i+L-1}] → 次トークン t_{i+L}"""

    def __init__(self, text_or_ids: str | list[int], vocab: CharVocab, seq_len: int) -> None:
        if isinstance(text_or_ids, list):
            self.ids = text_or_ids
        else:
            self.ids = vocab.encode(text_or_ids)
        self.vocab = vocab
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.ids) - self.seq_len - 1)

    def __getitem__(self, idx: int):
        ctx = self.ids[idx : idx + self.seq_len]
        nxt = self.ids[idx + self.seq_len]
        return (
            torch.tensor(ctx, dtype=torch.long),
            torch.tensor(nxt, dtype=torch.long),
        )


def get_lm_loaders(
    corpus: TextCorpus,
    seq_len: int = 64,
    batch_size: int = 16,
    val_ratio: float = 0.3,
    pin_memory: bool = False,
    num_workers: int = 0,
    prefetch_factor: int | None = None,
) -> tuple[DataLoader, DataLoader, CharVocab]:
    token_ids = corpus.token_ids
    n = len(token_ids)
    if n < seq_len + 2:
        raise ValueError("コーパスが短すぎます。data/sample_corpus.txt を長くしてください。")
    
    n_val_tokens = int(n * val_ratio)
    n_train_tokens = n - n_val_tokens
    
    train_ids = token_ids[:n_train_tokens]
    val_ids = token_ids[n_train_tokens:]
    
    train_ds = LanguageModelDataset(train_ids, corpus.vocab, seq_len)
    val_ds = LanguageModelDataset(val_ids, corpus.vocab, seq_len)
    
    kw = {"num_workers": num_workers, "pin_memory": pin_memory}
    if prefetch_factor is not None and num_workers > 0:
        kw["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    return train_loader, val_loader, corpus.vocab


# 後方互換
get_text_loaders = get_lm_loaders
