from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def load_split_spec(path: str) -> Dict:
    spec_path = Path(path)
    with spec_path.open("r", encoding="utf-8") as fh:
        spec = json.load(fh)
    if not isinstance(spec, dict):
        raise ValueError("split spec 必须是 JSON 对象")
    docs = spec.get("docs")
    folds = spec.get("folds")
    if not isinstance(docs, list) or not isinstance(folds, list):
        raise ValueError("split spec 缺少 docs 或 folds 列表")
    return spec


def load_named_split(path: str, split_name: str) -> Tuple[List[Dict], Dict, Dict]:
    spec = load_split_spec(path)
    docs = spec["docs"]
    for fold in spec["folds"]:
        if fold.get("name") == split_name:
            _validate_fold(fold, len(docs))
            return docs, fold, spec.get("protocol", {})
    raise KeyError(f"split spec 中不存在名为 {split_name!r} 的 fold")


def slice_split_docs(docs: Sequence[Dict], ids: Sequence[int]) -> Tuple[List[str], List[int]]:
    paths: List[str] = []
    labels: List[int] = []
    for raw_idx in ids:
        idx = int(raw_idx)
        item = docs[idx]
        path = item.get("path")
        label = item.get("label")
        if not isinstance(path, str):
            raise ValueError(f"doc[{idx}] 缺少 path")
        if label not in (0, 1):
            raise ValueError(f"doc[{idx}] 的 label 必须是 0/1")
        paths.append(path)
        labels.append(int(label))
    return paths, labels


def _validate_fold(fold: Dict, total_docs: int) -> None:
    for key in ("train_ids", "val_ids", "test_ids"):
        values = fold.get(key)
        if not isinstance(values, list):
            raise ValueError(f"fold 缺少列表字段 {key}")
        for raw_idx in values:
            idx = int(raw_idx)
            if idx < 0 or idx >= total_docs:
                raise ValueError(f"{key} 中索引 {idx} 超出范围（共有 {total_docs} 个 doc）")
