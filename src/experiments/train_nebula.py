#!/usr/bin/env python3
"""
Fine-tune Nebula TransformerEncoderChunks on本地 CAPE JSON 数据。

流程：
 1. 将本地 CAPE JSON 报告编码为 Nebula BPE token 序列；
 2. 按 holdout / dir / time_ood / split_spec 协议准备训练 / 验证 / 测试集；
 3. 使用官方 ModelTrainer 进行监督训练，可选 GPU；
 4. 在验证 / 测试集上评估并输出指标与模型权重。

依赖：
  - third_party/nebula 仓库；
  - torch、numpy、sklearn、tqdm、sentencepiece 等。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
NEBULA_REPO = ROOT_DIR / "third_party" / "nebula"
if not NEBULA_REPO.exists():
    raise RuntimeError(
        "未找到 third_party/nebula。请先执行：\n"
        "  git clone https://github.com/dtrizna/nebula third_party/nebula"
    )

import sys

if str(NEBULA_REPO) not in sys.path:
    sys.path.insert(0, str(NEBULA_REPO))

import torch
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss

from nebula import Nebula, ModelTrainer  # type: ignore
from src.baselines.cape_adapter import cape_to_speakeasy_format
from src.utils.json_splits import (
    load_json_split_spec_splits,
    load_json_time_ood_splits,
)

LOGGER = logging.getLogger("train_nebula")

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    orjson = None  # type: ignore


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    split_mode: str = "holdout"  # holdout / dir / time_ood / split_spec
    mal_dir: Optional[str] = None
    benign_dir: Optional[str] = None
    train_dir: Optional[str] = None
    val_dir: Optional[str] = None
    test_dir: Optional[str] = None
    split_spec_path: Optional[str] = None
    split_name: Optional[str] = None
    malicious_manifest_path: Optional[str] = None
    benign_manifest_path: Optional[str] = None
    train_end_date: Optional[str] = None
    val_end_date: Optional[str] = None
    benign_split_strategy: str = "random"
    output_dir: str = "outputs/nebula_train"
    model_out: Optional[str] = None
    summary_out: Optional[str] = None
    tokenizer: str = "bpe"  # bpe / whitespace
    seq_len: int = 512
    limit_per_class: Optional[int] = None
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    seed: int = 42
    seeds: Optional[List[int]] = None
    epochs: int = 5
    batch_size: int = 96
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    time_budget_min: Optional[float] = None  # minutes
    device: Optional[str] = None  # cpu / cuda / mps / auto
    cache_npz: Optional[str] = None
    progress: bool = False
    target_fpr: float = 0.01

    config_path: Optional[str] = field(default=None, repr=False, compare=False)

    def validate(self) -> None:
        if self.split_mode not in {"holdout", "dir", "time_ood", "split_spec"}:
            raise ValueError("split_mode 必须是 holdout / dir / time_ood / split_spec")
        if self.split_mode == "holdout":
            missing = [name for name in ("mal_dir", "benign_dir") if not getattr(self, name)]
            if missing:
                raise ValueError(f"holdout 模式需要提供 {missing}")
            for name in ("mal_dir", "benign_dir"):
                path = Path(str(getattr(self, name)))
                if not path.exists() or not path.is_dir():
                    raise FileNotFoundError(f"{name} 目录不存在：{path}")
        if self.split_mode == "dir":
            missing = [name for name in ("train_dir", "val_dir", "test_dir") if not getattr(self, name)]
            if missing:
                raise ValueError(f"dir 模式需要提供 {missing}")
            for name in ("train_dir", "val_dir", "test_dir"):
                path = Path(str(getattr(self, name)))
                if not path.exists() or not path.is_dir():
                    raise FileNotFoundError(f"{name} 目录不存在：{path}")
        if self.split_mode == "split_spec":
            missing = [
                name
                for name in ("split_spec_path", "split_name", "mal_dir", "benign_dir")
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(f"split_spec 模式需要提供 {missing}")
            for name in ("split_spec_path", "mal_dir", "benign_dir"):
                path = Path(str(getattr(self, name)))
                if not path.exists():
                    raise FileNotFoundError(f"{name} 路径不存在：{path}")
        if self.split_mode == "time_ood":
            missing = [
                name
                for name in ("malicious_manifest_path", "benign_manifest_path", "mal_dir", "benign_dir")
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(f"time_ood 模式需要提供 {missing}")
            if self.benign_split_strategy not in {"random", "time"}:
                raise ValueError("benign_split_strategy 必须是 random 或 time")
            for name in ("malicious_manifest_path", "benign_manifest_path", "mal_dir", "benign_dir"):
                path = Path(str(getattr(self, name)))
                if not path.exists():
                    raise FileNotFoundError(f"{name} 路径不存在：{path}")
        if self.limit_per_class is not None and self.limit_per_class < 1:
            raise ValueError("limit_per_class 至少为 1")
        if not (0 < self.val_ratio < 1) or not (0 < self.test_ratio < 1):
            raise ValueError("val_ratio/test_ratio 必须在 (0,1) 内")
        if self.val_ratio + self.test_ratio >= 0.95:
            raise ValueError("val_ratio + test_ratio 过大")
        if self.seq_len <= 0:
            raise ValueError("seq_len 必须为正")
        if self.epochs <= 0:
            raise ValueError("epochs 必须为正")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须为正")
        if not (0 < self.target_fpr < 1):
            raise ValueError("target_fpr 必须在 (0,1) 之间")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if self.model_out:
            Path(self.model_out).parent.mkdir(parents=True, exist_ok=True)
        if self.summary_out:
            Path(self.summary_out).parent.mkdir(parents=True, exist_ok=True)


def load_config_file(path: Optional[str]) -> Dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("配置文件需为 JSON 对象")
    return data


def build_parser(defaults: Dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="JSON 配置文件路径")
    parser.add_argument("--split_mode", choices=["holdout", "dir", "time_ood", "split_spec"], default=defaults.get("split_mode", "holdout"))
    parser.add_argument("--mal_dir", type=str, default=defaults.get("mal_dir"))
    parser.add_argument("--benign_dir", type=str, default=defaults.get("benign_dir"))
    parser.add_argument("--train_dir", type=str, default=defaults.get("train_dir"))
    parser.add_argument("--val_dir", type=str, default=defaults.get("val_dir"))
    parser.add_argument("--test_dir", type=str, default=defaults.get("test_dir"))
    parser.add_argument("--split_spec_path", type=str, default=defaults.get("split_spec_path"))
    parser.add_argument("--split_name", type=str, default=defaults.get("split_name"))
    parser.add_argument("--malicious_manifest_path", type=str, default=defaults.get("malicious_manifest_path"))
    parser.add_argument("--benign_manifest_path", type=str, default=defaults.get("benign_manifest_path"))
    parser.add_argument("--train_end_date", type=str, default=defaults.get("train_end_date"))
    parser.add_argument("--val_end_date", type=str, default=defaults.get("val_end_date"))
    parser.add_argument("--benign_split_strategy", choices=["random", "time"], default=defaults.get("benign_split_strategy", "random"))
    parser.add_argument("--output_dir", type=str, default=defaults.get("output_dir", "outputs/nebula_train"))
    parser.add_argument("--model_out", type=str, default=defaults.get("model_out"))
    parser.add_argument("--summary_out", type=str, default=defaults.get("summary_out"))
    parser.add_argument("--tokenizer", choices=["bpe", "whitespace"], default=defaults.get("tokenizer", "bpe"))
    parser.add_argument("--seq_len", type=int, default=defaults.get("seq_len", 512))
    parser.add_argument("--limit_per_class", type=int, default=defaults.get("limit_per_class"))
    parser.add_argument("--val_ratio", type=float, default=defaults.get("val_ratio", 0.1))
    parser.add_argument("--test_ratio", type=float, default=defaults.get("test_ratio", 0.2))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--seeds", nargs="*", default=defaults.get("seeds"))
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 5))
    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 96))
    parser.add_argument("--learning_rate", type=float, default=defaults.get("learning_rate", 3e-4))
    parser.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay", 1e-2))
    parser.add_argument("--time_budget_min", type=float, default=defaults.get("time_budget_min"))
    parser.add_argument("--device", type=str, default=defaults.get("device"))
    parser.add_argument("--cache_npz", type=str, default=defaults.get("cache_npz"))
    parser.add_argument("--progress", action="store_true", default=defaults.get("progress", False))
    parser.add_argument("--target_fpr", type=float, default=defaults.get("target_fpr", 0.01))
    return parser


def parse_config() -> TrainConfig:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, default=None)
    base_args, remaining = base_parser.parse_known_args()
    cfg_dict = load_config_file(base_args.config)
    parser = build_parser(cfg_dict)
    args = parser.parse_args(remaining)
    merged = {**cfg_dict, **{k: v for k, v in vars(args).items() if v is not None}}
    merged["config_path"] = base_args.config
    seeds_value = merged.get("seeds")
    if seeds_value:
        if isinstance(seeds_value, str):
            parsed = [seed.strip() for seed in seeds_value.split(",")]
            merged["seeds"] = [int(seed) for seed in parsed if seed]
        elif isinstance(seeds_value, list):
            merged["seeds"] = [int(seed) for seed in seeds_value]
        else:
            raise ValueError("--seeds 参数解析失败")
    cfg = TrainConfig(**merged)
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------- #
# 数据准备
# --------------------------------------------------------------------------- #


def list_json_files(directory: str) -> List[Path]:
    return sorted(path for path in Path(directory).rglob("*.json") if path.is_file())


def select_subset(paths: List[Path], limit: Optional[int]) -> List[Path]:
    if limit is None or limit >= len(paths):
        return paths
    return paths[:limit]


def subsample_per_class(
    paths: Sequence[Path],
    labels: Sequence[int],
    limit_per_class: Optional[int],
) -> Tuple[List[Path], List[int]]:
    path_list = list(paths)
    label_list = [int(label) for label in labels]
    if limit_per_class is None:
        return path_list, label_list

    kept_paths: List[Path] = []
    kept_labels: List[int] = []
    counts = {0: 0, 1: 0}
    for path, label in zip(path_list, label_list):
        if label not in counts:
            raise ValueError(f"标签必须是 0/1，收到 {label!r}")
        if counts[label] >= limit_per_class:
            continue
        kept_paths.append(path)
        kept_labels.append(label)
        counts[label] += 1
    return kept_paths, kept_labels


def ensure_binary_split(name: str, labels: Sequence[int], *, stage: str) -> None:
    if len(labels) == 0:
        raise RuntimeError(f"{name} 在 {stage} 后为空，无法训练/评估。")
    unique = sorted({int(label) for label in labels})
    if unique != [0, 1]:
        raise RuntimeError(f"{name} 在 {stage} 后只剩类别 {unique}，无法进行二分类训练/评估。")


def infer_label(path: Path) -> Optional[int]:
    label_path = path.with_suffix(".label")
    if label_path.exists():
        try:
            return int(label_path.read_text(encoding="utf-8").strip())
        except Exception:
            return None
    parts = [part.lower() for part in path.parts]
    if any(token in part for part in parts for token in ("malicious", "malware", "black")):
        return 1
    if any(token in part for part in parts for token in ("benign", "goodware", "white")):
        return 0
    return None


def load_json(path: Path) -> dict:
    data = path.read_bytes()
    if orjson is not None:
        return orjson.loads(data)
    return json.loads(data)


def convert_to_nebula_report(report: dict, source: Path) -> Optional[list]:
    """Convert a CAPE report to a Speakeasy entry_points list.

    Returns a LIST so that Nebula.preprocess() correctly triggers
    filter_and_normalize_report() instead of raw-encoding the dict.
    """
    if not isinstance(report, dict):
        LOGGER.warning("编码失败 %s: JSON 顶层不是对象", source.name)
        return None
    if "entry_points" in report:
        ep = report["entry_points"]
        return ep if isinstance(ep, list) else None
    converted = cape_to_speakeasy_format(report)
    if converted is None:
        LOGGER.warning("编码失败 %s: 未提取到可用 API 调用", source.name)
        return None
    return converted


def encode_reports(
    nebula_model: Nebula,
    paths: Sequence[Path],
    label: int,
    *,
    progress: bool = False,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    features: List[np.ndarray] = []
    labels: List[int] = []
    ids: List[str] = []

    iterator = paths
    if progress:
        iterator = tqdm(iterator, desc=f"Encoding label={label}", total=len(paths))

    for path in iterator:
        try:
            report = load_json(path)
            nebula_report = convert_to_nebula_report(report, path)
            if nebula_report is None:
                continue
            arr = nebula_model.preprocess(nebula_report)
            if arr.ndim == 2:
                arr = arr[0]
        except Exception as exc:
            LOGGER.warning("编码失败 %s: %s", path.name, exc)
            continue
        features.append(arr.astype(np.int32, copy=False))
        labels.append(label)
        ids.append(path.stem)
    return features, labels, ids


def encode_labeled_reports(
    nebula_model: Nebula,
    paths: Sequence[Path],
    labels: Sequence[int],
    *,
    progress: bool = False,
    desc: str = "Encoding",
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    features: List[np.ndarray] = []
    kept_labels: List[int] = []
    ids: List[str] = []

    iterator = list(zip(paths, labels))
    if progress:
        iterator = tqdm(iterator, desc=desc, total=len(iterator))

    for path, label in iterator:
        try:
            report = load_json(path)
            nebula_report = convert_to_nebula_report(report, path)
            if nebula_report is None:
                continue
            arr = nebula_model.preprocess(nebula_report)
            if arr.ndim == 2:
                arr = arr[0]
        except Exception as exc:
            LOGGER.warning("编码失败 %s: %s", path.name, exc)
            continue
        features.append(arr.astype(np.int32, copy=False))
        kept_labels.append(int(label))
        ids.append(path.stem)
    return features, kept_labels, ids


def _dataset_digest(
    cfg: TrainConfig,
    paths: Sequence[Path],
    labels: Sequence[int],
    *,
    split_name: Optional[str],
) -> str:
    payload = [
        f"tokenizer={cfg.tokenizer}",
        f"seq_len={cfg.seq_len}",
        f"split={split_name or 'dataset'}",
    ]
    payload.extend(
        f"{Path(path).resolve()}::{int(label)}" for path, label in zip(paths, labels)
    )
    return hashlib.md5("\n".join(payload).encode("utf-8")).hexdigest()[:12]


def _cache_path_for_split(
    cache_npz: Optional[str],
    split_name: Optional[str],
    dataset_digest: str,
) -> Optional[Path]:
    if not cache_npz:
        return None
    base = Path(cache_npz)
    suffix = base.suffix or ".npz"
    suffix_parts = [base.stem]
    if split_name:
        suffix_parts.append(split_name)
    suffix_parts.append(dataset_digest)
    return base.with_name("_".join(suffix_parts) + suffix)


def build_dataset_from_paths(
    cfg: TrainConfig,
    paths: Sequence[Path],
    labels: Sequence[int],
    *,
    split_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    dataset_digest = _dataset_digest(cfg, paths, labels, split_name=split_name)
    cache_path = _cache_path_for_split(cfg.cache_npz, split_name, dataset_digest)
    if cache_path and cache_path.exists():
        LOGGER.info("从缓存加载特征：%s", cache_path)
        data = np.load(cache_path, allow_pickle=False)
        return data["features"], data["labels"], data["ids"].tolist()

    nebula_model = Nebula(
        vocab_size=50000,
        seq_len=cfg.seq_len,
        tokenizer=cfg.tokenizer,
    )
    nebula_model.model.eval()

    features, kept_labels, ids = encode_labeled_reports(
        nebula_model,
        paths,
        labels,
        progress=cfg.progress,
        desc=f"Encoding {split_name or 'dataset'}",
    )
    if not features:
        raise RuntimeError(f"{split_name or 'dataset'} 中没有成功编码的样本。")

    feature_arr = np.vstack(features)
    label_arr = np.array(kept_labels, dtype=np.int8)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, features=feature_arr, labels=label_arr, ids=np.array(ids))
        LOGGER.info("特征已缓存至 %s", cache_path)
    return feature_arr, label_arr, ids


def load_holdout_paths(cfg: TrainConfig) -> Tuple[List[Path], List[int]]:
    assert cfg.mal_dir and cfg.benign_dir
    mal_paths = select_subset(list_json_files(cfg.mal_dir), cfg.limit_per_class)
    ben_paths = select_subset(list_json_files(cfg.benign_dir), cfg.limit_per_class)
    LOGGER.info("读取恶意样本 %d 个，良性样本 %d 个。", len(mal_paths), len(ben_paths))
    return mal_paths + ben_paths, [1] * len(mal_paths) + [0] * len(ben_paths)


def load_labeled_json_dir(directory: str, limit: Optional[int]) -> Tuple[List[Path], List[int]]:
    paths: List[Path] = []
    labels: List[int] = []
    for path in list_json_files(directory):
        label = infer_label(path)
        if label is None:
            continue
        paths.append(path)
        labels.append(int(label))
    paths, labels = subsample_per_class(paths, labels, limit)
    if not paths:
        raise RuntimeError(f"{directory} 中未找到可识别标签的 JSON 报告。")
    return paths, labels


def load_explicit_split_paths(
    cfg: TrainConfig,
) -> Tuple[List[Path], List[int], List[Path], List[int], List[Path], List[int]]:
    if cfg.split_mode == "dir":
        assert cfg.train_dir and cfg.val_dir and cfg.test_dir
        train_paths, train_labels = load_labeled_json_dir(cfg.train_dir, cfg.limit_per_class)
        val_paths, val_labels = load_labeled_json_dir(cfg.val_dir, cfg.limit_per_class)
        test_paths, test_labels = load_labeled_json_dir(cfg.test_dir, cfg.limit_per_class)
        for name, labels in (("train", train_labels), ("val", val_labels), ("test", test_labels)):
            ensure_binary_split(name, labels, stage="split preparation")
        return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

    if cfg.split_mode == "split_spec":
        assert cfg.split_spec_path and cfg.split_name and cfg.mal_dir and cfg.benign_dir
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = load_json_split_spec_splits(
            split_spec_path=cfg.split_spec_path,
            split_name=cfg.split_name,
            mal_dir=cfg.mal_dir,
            benign_dir=cfg.benign_dir,
        )
        train_paths, train_labels = subsample_per_class([Path(path) for path in train_paths], train_labels, cfg.limit_per_class)
        val_paths, val_labels = subsample_per_class([Path(path) for path in val_paths], val_labels, cfg.limit_per_class)
        test_paths, test_labels = subsample_per_class([Path(path) for path in test_paths], test_labels, cfg.limit_per_class)
        for name, labels in (("train", train_labels), ("val", val_labels), ("test", test_labels)):
            ensure_binary_split(name, labels, stage="split preparation")
        return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

    assert cfg.malicious_manifest_path and cfg.benign_manifest_path and cfg.mal_dir and cfg.benign_dir
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = load_json_time_ood_splits(
        malicious_manifest_path=cfg.malicious_manifest_path,
        benign_manifest_path=cfg.benign_manifest_path,
        mal_dir=cfg.mal_dir,
        benign_dir=cfg.benign_dir,
        train_end_date=cfg.train_end_date,
        val_end_date=cfg.val_end_date,
        benign_split_strategy=cfg.benign_split_strategy,
        seed=cfg.seed,
    )
    train_paths, train_labels = subsample_per_class([Path(path) for path in train_paths], train_labels, cfg.limit_per_class)
    val_paths, val_labels = subsample_per_class([Path(path) for path in val_paths], val_labels, cfg.limit_per_class)
    test_paths, test_labels = subsample_per_class([Path(path) for path in test_paths], test_labels, cfg.limit_per_class)
    for name, labels in (("train", train_labels), ("val", val_labels), ("test", test_labels)):
        ensure_binary_split(name, labels, stage="split preparation")
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def load_or_build_dataset(cfg: TrainConfig) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    paths, labels = load_holdout_paths(cfg)
    return build_dataset_from_paths(cfg, paths, labels)


def stratified_holdout_indices(y: np.ndarray, val_ratio: float, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(idx, y))
    val_rel = val_ratio / (1 - test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed + 1)
    train_idx_rel, val_idx_rel = next(sss2.split(train_val_idx, y[train_val_idx]))
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]
    return train_idx, val_idx, test_idx


# --------------------------------------------------------------------------- #
# 训练与评估
# --------------------------------------------------------------------------- #


def pick_metrics(fp_rates: Sequence[float], tprs: Sequence[float], f1s: Sequence[float], target_fpr: float) -> Dict[str, float]:
    idx = min(range(len(fp_rates)), key=lambda i: abs(fp_rates[i] - target_fpr))
    return {
        "target_fpr": float(target_fpr),
        "nearest_fpr": float(fp_rates[idx]),
        "tpr": float(tprs[idx]),
        "f1": float(f1s[idx]),
        "index": idx,
    }


def prepare_device(device: Optional[str]) -> str:
    if device:
        dev = device.lower()
        if dev == "cpu":
            return dev
        if dev == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("指定使用 MPS 但当前环境不可用。")
            return dev
        if dev.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("指定使用 CUDA 但当前环境不可用。")
            return dev
        raise ValueError("device 仅支持 cpu / cuda[:idx] / mps / auto")
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def train_and_evaluate(cfg: TrainConfig) -> Dict:
    if cfg.split_mode == "holdout":
        features, labels, ids = load_or_build_dataset(cfg)
        LOGGER.info("总样本数：%d", len(labels))
        ensure_binary_split("holdout", labels, stage="encoding")
        train_idx, val_idx, test_idx = stratified_holdout_indices(labels, cfg.val_ratio, cfg.test_ratio, cfg.seed)
        X_train = features[train_idx]
        y_train = labels[train_idx].astype(np.float32)
        X_val = features[val_idx]
        y_val = labels[val_idx].astype(np.float32)
        X_test = features[test_idx]
        y_test = labels[test_idx].astype(np.float32)
        dataset_summary = {
            "mode": "holdout",
            "total": int(len(labels)),
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        }
    else:
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = load_explicit_split_paths(cfg)
        X_train, y_train_arr, train_ids = build_dataset_from_paths(cfg, train_paths, train_labels, split_name="train")
        X_val, y_val_arr, val_ids = build_dataset_from_paths(cfg, val_paths, val_labels, split_name="val")
        X_test, y_test_arr, test_ids = build_dataset_from_paths(cfg, test_paths, test_labels, split_name="test")
        ensure_binary_split("train", y_train_arr, stage="encoding")
        ensure_binary_split("val", y_val_arr, stage="encoding")
        ensure_binary_split("test", y_test_arr, stage="encoding")
        y_train = y_train_arr.astype(np.float32)
        y_val = y_val_arr.astype(np.float32)
        y_test = y_test_arr.astype(np.float32)
        ids = train_ids + val_ids + test_ids
        dataset_summary = {
            "mode": cfg.split_mode,
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        }

    LOGGER.info("划分数据 => train: %d, val: %d, test: %d", len(X_train), len(X_val), len(X_test))

    device = prepare_device(cfg.device)
    LOGGER.info("使用设备：%s", device)

    nebula_model = Nebula(
        vocab_size=50000,
        seq_len=cfg.seq_len,
        tokenizer=cfg.tokenizer,
    )
    model = nebula_model.model  # TransformerEncoderChunks

    trainer = ModelTrainer(
        model=model,
        device=torch.device(device),
        loss_function=BCEWithLogitsLoss(),
        optimizer_class=AdamW,
        optimizer_config={"lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
        optim_scheduler=None,
        batchSize=cfg.batch_size,
        verbosity_n_batches=100,
        outputFolder=cfg.output_dir,
    )

    requested_time_budget = int(cfg.time_budget_min * 60) if cfg.time_budget_min else None
    time_budget = None
    train_epochs = cfg.epochs
    if requested_time_budget is not None:
        LOGGER.warning(
            "Nebula upstream 的 time_budget 训练路径当前不可用；"
            "已回退到按 epochs=%d 训练。",
            cfg.epochs,
        )
    trainer.train(
        X_train,
        y_train,
        epochs=train_epochs,
        time_budget=time_budget,
    )

    val_loss, val_tprs, val_f1s, val_auc = trainer.evaluate(X_val, y_val)
    test_loss, test_tprs, test_f1s, test_auc = trainer.evaluate(X_test, y_test)

    target_val = pick_metrics(trainer.fp_rates, val_tprs, val_f1s, cfg.target_fpr)
    target_test = pick_metrics(trainer.fp_rates, test_tprs, test_f1s, cfg.target_fpr)

    summary = {
        "mode": f"holdout-{cfg.split_mode}" if cfg.split_mode != "holdout" else "holdout",
        "config": {
            **{k: v for k, v in asdict(cfg).items() if k not in {"config_path"}},
            "device": device,
        },
        "dataset": dataset_summary,
        "train": {
            "epochs": train_epochs,
            "requested_epochs": cfg.epochs,
            "time_budget_sec": time_budget,
            "requested_time_budget_sec": requested_time_budget,
            "train_auc_last": float(trainer.auc[-1]) if trainer.auc else None,
            "fp_rates": [float(f) for f in trainer.fp_rates],
            "train_tpr_last": trainer.train_tprs[-1].tolist() if trainer.train_tprs.size else [],
            "train_f1_last": trainer.train_f1s[-1].tolist() if trainer.train_f1s.size else [],
        },
        "validation": {
            "loss_mean": float(np.mean(val_loss)),
            "auc": float(val_auc),
            "tprs": [float(x) for x in val_tprs],
            "f1s": [float(x) for x in val_f1s],
            "target": target_val,
        },
        "test": {
            "loss_mean": float(np.mean(test_loss)),
            "auc": float(test_auc),
            "tprs": [float(x) for x in test_tprs],
            "f1s": [float(x) for x in test_f1s],
            "target": target_test,
        },
        "best": {
            "f1": float(target_test["f1"]),
            "auroc": float(test_auc),
            "tpr_id": float(target_val["tpr"]),
            "fpr_id": float(target_val["nearest_fpr"]),
            "tpr_ood": float(target_test["tpr"]),
            "fpr_ood": float(target_test["nearest_fpr"]),
            "target_fpr": float(cfg.target_fpr),
        },
    }

    if cfg.split_mode == "split_spec":
        summary["split_name"] = cfg.split_name
        summary["split_spec_path"] = cfg.split_spec_path

    if cfg.summary_out:
        with open(cfg.summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        LOGGER.info("训练摘要已写入 %s", cfg.summary_out)

    if cfg.model_out:
        torch.save(model.state_dict(), cfg.model_out)
        LOGGER.info("模型权重已保存至 %s", cfg.model_out)

    return summary


def aggregate_seed_summaries(cfg: TrainConfig, summaries: List[Dict]) -> Dict:
    seeds = [summary["seed"] for summary in summaries]
    best_keys = set()
    for summary in summaries:
        best = summary.get("best", {})
        for key, value in best.items():
            if isinstance(value, (int, float)):
                best_keys.add(key)

    best_mean: Dict[str, float] = {}
    best_std: Dict[str, float] = {}
    for key in sorted(best_keys):
        values = [
            float(summary["best"][key])
            for summary in summaries
            if isinstance(summary.get("best", {}).get(key), (int, float))
        ]
        if values:
            best_mean[key] = float(np.mean(values))
            best_std[key] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    return {
        "mode": summaries[0].get("mode"),
        "tokenizer": cfg.tokenizer,
        "seeds": seeds,
        "per_seed": summaries,
        "best_mean": best_mean,
        "best_std": best_std,
        "config": {k: v for k, v in asdict(cfg).items() if k not in {"config_path", "seed", "seeds"}},
    }


def save_summary(summary: Dict, path: Optional[str]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    LOGGER.info("训练摘要已写入 %s", out_path)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def main() -> None:
    setup_logging()
    cfg = parse_config()
    LOGGER.info("训练配置：%s", cfg)
    if cfg.seeds:
        per_seed: List[Dict] = []
        for seed in cfg.seeds:
            LOGGER.info(" - 种子 %d", seed)
            seed_cfg = TrainConfig(**{**asdict(cfg), "seed": seed, "seeds": None, "summary_out": None})
            seed_summary = train_and_evaluate(seed_cfg)
            seed_summary["seed"] = seed
            per_seed.append(seed_summary)
        aggregated = aggregate_seed_summaries(cfg, per_seed)
        save_summary(aggregated, cfg.summary_out)
        print(json.dumps(aggregated, ensure_ascii=False, indent=2))
        return

    summary = train_and_evaluate(cfg)
    summary["seed"] = cfg.seed
    save_summary(summary, cfg.summary_out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
