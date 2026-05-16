"""Hugging Face datasets からコーパスを構築。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterator

_TEXT_COLUMN_CANDIDATES = (
    "text",
    "content",
    "article",
    "document",
    "paragraph",
    "sentence",
    "body",
    "raw",
    "ja",
)


def _extract_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_extract_cell(v) for v in value)
    return str(value)


def infer_text_column(feature_names: list[str], preferred: str | None = None) -> str:
    if preferred:
        if preferred not in feature_names:
            raise ValueError(f"列 {preferred!r} がありません。利用可能: {feature_names}")
        return preferred
    for cand in _TEXT_COLUMN_CANDIDATES:
        if cand in feature_names:
            return cand
    if not feature_names:
        raise ValueError("データセットに列がありません")
    return feature_names[0]


def _resolve_split(raw: Any, split: str) -> tuple[Any, str]:
    if not hasattr(raw, "keys"):
        return raw, split
    names = list(raw.keys())
    if split in raw:
        return raw[split], split
    if len(names) == 1:
        return raw[names[0]], names[0]
    raise ValueError(f"split={split!r} が見つかりません。利用可能: {names}")


def _iter_rows(ds: Any, max_samples: int | None) -> Iterator[dict]:
    n = 0
    for row in ds:
        yield row
        n += 1
        if max_samples is not None and n >= max_samples:
            break


def load_hf_text(
    dataset: str,
    *,
    split: str = "train",
    text_column: str | None = None,
    max_samples: int | None = None,
    max_chars: int | None = None,
    streaming: bool = False,
    dataset_config: str | None = None,
    trust_remote_code: bool = False,
    cache_dir: str | Path | None = None,
    vocab: Any | None = None,
    data_files: str | list[str] | dict[str, str | list[str]] | None = None,
    **kwargs: Any,
) -> tuple[str, str]:
    """
    load_dataset("fn-aka-mur/wiki40b_ja") 相当をテキスト1本に連結。

    Returns:
        (text, column_name)
    """
    dataset = dataset.strip().strip("'\"")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("pip install datasets が必要です") from exc

    if cache_dir is not None:
        key = f"{dataset}|{dataset_config}|{split}|{max_samples}|{max_chars}|{streaming}"
        digest = hashlib.md5(key.encode()).hexdigest()[:12]
        cache_path = Path(cache_dir) / f"hf_{digest}.txt"
        if cache_path.is_file():
            print(f"  cache hit: {cache_path}", flush=True)
            return cache_path.read_text(encoding="utf-8"), "text"

    base_kw: dict[str, Any] = {"trust_remote_code": trust_remote_code, **kwargs}
    if data_files is not None:
        base_kw["data_files"] = data_files

    if streaming:
        stream_kw = {**base_kw, "split": split, "streaming": True}
        if dataset_config:
            ds = load_dataset(dataset, dataset_config, **stream_kw)
        else:
            ds = load_dataset(dataset, **stream_kw)
    else:
        if dataset_config:
            raw = load_dataset(dataset, dataset_config, **base_kw)
        else:
            raw = load_dataset(dataset, **base_kw)
        ds, split = _resolve_split(raw, split)
        if max_samples is not None and len(ds) > max_samples:
            ds = ds.select(range(max_samples))

    parts: list[str] = []
    row_iter = _iter_rows(ds, max_samples if streaming else None)
    try:
        first = next(row_iter)
    except StopIteration:
        first = None

    if first is None:
        raise ValueError(f"{dataset} が空です（split={split}）")

    col = infer_text_column(list(first.keys()), text_column)

    total_chars = 0
    n_rows = 0
    for row in (first, *row_iter):
        s = _extract_cell(row[col]).strip()
        if not s:
            continue
        if max_chars is not None and total_chars + len(s) > max_chars:
            remain = max_chars - total_chars
            if remain > 0:
                parts.append(s[:remain])
            break
        parts.append(s)
        total_chars += len(s)
        n_rows += 1
        if n_rows % 100 == 0:
            print(f"  ... {n_rows} rows, {total_chars:,} chars", flush=True)

    if not parts:
        raise ValueError(f"{dataset} からテキストを抽出できませんでした（列: {col}）")

    text = "\n".join(parts)
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"hf_{hashlib.md5(key.encode()).hexdigest()[:12]}.txt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
        print(f"  cached: {cache_path}", flush=True)

    return text, col

def load_hf_corpus(
    dataset: str,
    *,
    split: str = "train",
    text_column: str | None = None,
    max_samples: int | None = None,
    max_chars: int | None = None,
    streaming: bool = False,
    dataset_config: str | None = None,
    cache_dir: str | Path | None = None,
    vocab: Any | None = None,
    data_files: str | list[str] | dict[str, str | list[str]] | None = None,
    **kwargs: Any,
):
    """TextCorpus を HF データセットから構築。"""
    from grim.data.text import TextCorpus

    # URL 形式の場合、データセット ID を抽出
    if dataset.startswith("https://huggingface.co/datasets/"):
        dataset = dataset.replace("https://huggingface.co/datasets/", "").split("?")[0].split("#")[0]
        if dataset.endswith("/"):
            dataset = dataset[:-1]

    text, col = load_hf_text(
        dataset,
        split=split,
        text_column=text_column,
        max_samples=max_samples,
        max_chars=max_chars,
        streaming=streaming,
        dataset_config=dataset_config,
        cache_dir=cache_dir,
        vocab=vocab,
        data_files=data_files,
        **kwargs,
    )
    source = f"hf:{dataset}" + (f"/{dataset_config}" if dataset_config else "") + f"/{split}:{col}"
    return TextCorpus.from_text(text, source=source, vocab=vocab), col
