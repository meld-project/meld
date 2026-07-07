#!/usr/bin/env python3
"""
Pretrained Nebula baseline for MELD experiments.

This runner mirrors the interface of the other experiment scripts so it can be
scheduled by `src/experiments/run_suite.py`. It loads the official Nebula
repository from `third_party/nebula`, performs inference on local CAPE JSON
reports, calibrates thresholds on validation data, and reports the same metric
schema as the TF-IDF and LEC runners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except Exception:  # pragma: no cover
    # Some remote environments ship a broken wandb dependency stack; in that
    # case we still want the baseline to run with logging disabled.
    WANDB_AVAILABLE = False
    wandb = None

ROOT_DIR = Path(__file__).resolve().parents[2]
NEBULA_REPO = ROOT_DIR / "third_party" / "nebula"
if not NEBULA_REPO.exists():
    raise RuntimeError(
        f"未找到 {NEBULA_REPO}。请先执行："
        " git clone https://github.com/dtrizna/nebula third_party/nebula"
    )

if str(NEBULA_REPO) not in sys.path:
    sys.path.insert(0, str(NEBULA_REPO))

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None

from nebula import Nebula

from src.baselines.cape_adapter import cape_to_speakeasy_format
from src.utils.json_splits import (
    load_json_split_spec_splits,
    load_json_time_ood_splits,
)


LOGGER = logging.getLogger("nebula_baseline")
_SCORE_CACHE: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray, List[str]]] = {}
_MODEL_CACHE: Dict[Tuple[str, int, str], Nebula] = {}
_FILE_SCORE_CACHE: Dict[Tuple[str, str, int], Tuple[float, str]] = {}


@dataclass
class TrainConfig:
    split_mode: str = "holdout"  # holdout / cv / dir / time_ood / split_spec
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
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    seed: int = 42
    seeds: Optional[List[int]] = None
    limit: Optional[int] = None
    n_splits: int = 5
    target_fpr: float = 0.01
    tokenizer: str = "bpe"
    seq_len: int = 512
    device: Optional[str] = None
    batch_size: int = 128
    cache_scores: bool = True
    progress: bool = False
    out: Optional[str] = None
    save_raw: bool = False

    use_wandb: bool = False
    wandb_project: str = "lec-experiments"
    wandb_entity: Optional[str] = None
    wandb_api_key: Optional[str] = None
    run_tag: str = ""

    config_path: Optional[str] = field(default=None, repr=False, compare=False)

    def validate(self) -> None:
        if self.split_mode not in {"holdout", "cv", "dir", "time_ood", "split_spec"}:
            raise ValueError("split_mode 必须是 holdout/cv/dir/time_ood/split_spec")
        if self.split_mode in {"holdout", "cv"}:
            missing = [name for name in ("mal_dir", "benign_dir") if not getattr(self, name)]
            if missing:
                raise ValueError(f"{self.split_mode} 模式需要提供 {missing}")
        if self.split_mode == "dir":
            missing = [name for name in ("train_dir", "val_dir", "test_dir") if not getattr(self, name)]
            if missing:
                raise ValueError(f"dir 模式需要提供 {missing}")
        if self.split_mode == "split_spec":
            missing = [
                name
                for name in ("split_spec_path", "split_name", "mal_dir", "benign_dir")
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(f"split_spec 模式需要提供 {missing}")
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
        if self.limit is not None and self.limit < 2:
            raise ValueError("limit 至少为 2")
        if not (0 < self.target_fpr < 1):
            raise ValueError("target_fpr 必须在 (0, 1) 内")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须为正")
        if self.seeds:
            unique = sorted(dict.fromkeys(int(seed) for seed in self.seeds))
            if not unique:
                raise ValueError("seeds 不能为空")
            self.seeds = unique


def load_config_file(path: Optional[str]) -> Dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("配置文件需为 JSON 对象")
    return data


def build_parser(defaults: Dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="JSON 配置文件路径")
    parser.add_argument("--split_mode", choices=["holdout", "cv", "dir", "time_ood", "split_spec"], default=defaults.get("split_mode", "holdout"))
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
    parser.add_argument("--val_ratio", type=float, default=defaults.get("val_ratio", 0.1))
    parser.add_argument("--test_ratio", type=float, default=defaults.get("test_ratio", 0.2))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--seeds", type=str, default=None, help="逗号分隔的随机种子，例如 40,41,42")
    parser.add_argument("--limit", type=int, default=defaults.get("limit"))
    parser.add_argument("--n_splits", type=int, default=defaults.get("n_splits", 5))
    parser.add_argument("--target_fpr", type=float, default=defaults.get("target_fpr", 0.01))
    parser.add_argument("--tokenizer", choices=["bpe", "whitespace"], default=defaults.get("tokenizer", "bpe"))
    parser.add_argument("--seq_len", type=int, default=defaults.get("seq_len", 512))
    parser.add_argument("--device", type=str, default=defaults.get("device"))
    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 128))
    parser.add_argument("--cache_scores", action="store_true", default=defaults.get("cache_scores", True))
    parser.add_argument("--no-cache_scores", action="store_false", dest="cache_scores")
    parser.add_argument("--progress", action="store_true", default=defaults.get("progress", False))
    parser.add_argument("--out", type=str, default=defaults.get("out"))
    parser.add_argument("--save_raw", action="store_true", default=defaults.get("save_raw", False))
    parser.add_argument("--use_wandb", action="store_true", default=defaults.get("use_wandb", False))
    parser.add_argument("--wandb_project", type=str, default=defaults.get("wandb_project", "lec-experiments"))
    parser.add_argument("--wandb_entity", type=str, default=defaults.get("wandb_entity"))
    parser.add_argument("--wandb_api_key", type=str, default=defaults.get("wandb_api_key"))
    parser.add_argument("--run_tag", type=str, default=defaults.get("run_tag", ""))
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
    seeds_value = merged.pop("seeds", None)
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


def _paths_digest(paths: Sequence[str]) -> str:
    joined = "\n".join(sorted(str(path) for path in paths))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def list_json_files(directory: str) -> List[Path]:
    return sorted(path for path in Path(directory).rglob("*.json") if path.is_file())


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
        LOGGER.warning("处理失败 %s: JSON 顶层不是对象", source.name)
        return None
    # Already in Speakeasy format (has entry_points list)
    if "entry_points" in report:
        ep = report["entry_points"]
        return ep if isinstance(ep, list) else None
    # CAPE format — convert to Speakeasy entry_points list
    converted = cape_to_speakeasy_format(report)
    if converted is None:
        LOGGER.warning("处理失败 %s: 未提取到可用 API 调用", source.name)
        return None
    return converted


def prepare_device(device: Optional[str]) -> str:
    if device:
        dev = device.lower()
        if dev == "cpu":
            return dev
        if dev == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():  # type: ignore[attr-defined]
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


def get_nebula_model(tokenizer: str, seq_len: int, device: str) -> Nebula:
    cache_key = (tokenizer, seq_len, device)
    model = _MODEL_CACHE.get(cache_key)
    if model is not None:
        return model

    nebula_model = Nebula(vocab_size=50000, seq_len=seq_len, tokenizer=tokenizer)
    nebula_model.model.to(torch.device(device))
    nebula_model.model.eval()
    _MODEL_CACHE[cache_key] = nebula_model
    return nebula_model


def predict_probabilities(
    nebula_model: Nebula,
    features_batch: Sequence[np.ndarray],
    *,
    device: str,
) -> np.ndarray:
    batch = np.stack(features_batch, axis=0)
    inputs = torch.as_tensor(batch, dtype=torch.long, device=torch.device(device))
    with torch.inference_mode():
        logits = nebula_model.model(inputs)
        probs = torch.sigmoid(logits)
    probs = probs.squeeze(-1) if probs.ndim > 1 else probs
    return probs.detach().cpu().numpy().astype(float, copy=False)


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


def subsample_paths_labels(
    paths: Sequence[str],
    labels: Sequence[int],
    limit: Optional[int],
    seed: int,
) -> Tuple[List[str], List[int]]:
    path_list = list(paths)
    label_list = list(labels)
    if limit is None or len(label_list) <= limit:
        return path_list, label_list

    y_arr = np.array(label_list, dtype=int)
    idx_all = np.arange(len(label_list))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=limit, random_state=seed)
    keep_idx, _ = next(sss.split(idx_all, y_arr))
    return [path_list[i] for i in keep_idx], [label_list[i] for i in keep_idx]


def load_holdout_dataset(cfg: TrainConfig) -> Tuple[List[str], List[int]]:
    assert cfg.mal_dir and cfg.benign_dir
    mal_paths = [str(path) for path in list_json_files(cfg.mal_dir)]
    ben_paths = [str(path) for path in list_json_files(cfg.benign_dir)]
    paths = mal_paths + ben_paths
    labels = [1] * len(mal_paths) + [0] * len(ben_paths)
    return subsample_paths_labels(paths, labels, cfg.limit, cfg.seed)


def load_labeled_json_dir(directory: str, limit: Optional[int], seed: int) -> Tuple[List[str], List[int]]:
    kept_paths: List[str] = []
    labels: List[int] = []
    for path in list_json_files(directory):
        label = infer_label(path)
        if label is None:
            continue
        kept_paths.append(str(path))
        labels.append(int(label))
    if not kept_paths:
        raise RuntimeError(f"{directory} 中未找到可识别标签的 JSON 报告。")
    return subsample_paths_labels(kept_paths, labels, limit, seed)


def load_dir_splits(cfg: TrainConfig) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    assert cfg.train_dir and cfg.val_dir and cfg.test_dir
    train_paths, train_labels = load_labeled_json_dir(cfg.train_dir, cfg.limit, cfg.seed)
    val_paths, val_labels = load_labeled_json_dir(cfg.val_dir, cfg.limit, cfg.seed + 1)
    test_paths, test_labels = load_labeled_json_dir(cfg.test_dir, cfg.limit, cfg.seed + 2)
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def load_split_spec_splits(cfg: TrainConfig) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    assert cfg.split_spec_path and cfg.split_name and cfg.mal_dir and cfg.benign_dir
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = load_json_split_spec_splits(
        split_spec_path=cfg.split_spec_path,
        split_name=cfg.split_name,
        mal_dir=cfg.mal_dir,
        benign_dir=cfg.benign_dir,
    )
    train_paths, train_labels = subsample_paths_labels(train_paths, train_labels, cfg.limit, cfg.seed)
    val_paths, val_labels = subsample_paths_labels(val_paths, val_labels, cfg.limit, cfg.seed + 1)
    test_paths, test_labels = subsample_paths_labels(test_paths, test_labels, cfg.limit, cfg.seed + 2)
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def load_time_ood_splits(cfg: TrainConfig) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    assert cfg.malicious_manifest_path and cfg.benign_manifest_path
    assert cfg.mal_dir and cfg.benign_dir
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
    train_paths, train_labels = subsample_paths_labels(train_paths, train_labels, cfg.limit, cfg.seed)
    val_paths, val_labels = subsample_paths_labels(val_paths, val_labels, cfg.limit, cfg.seed + 1)
    test_paths, test_labels = subsample_paths_labels(test_paths, test_labels, cfg.limit, cfg.seed + 2)
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def compute_nebula_scores(
    paths: Sequence[str],
    labels: Sequence[int],
    *,
    tokenizer: str,
    seq_len: int,
    device: str,
    batch_size: int,
    progress: bool = False,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    digest = _paths_digest(paths)
    cache_key = (digest, tokenizer, seq_len)
    if use_cache and cache_key in _SCORE_CACHE:
        scores, cached_labels, ids = _SCORE_CACHE[cache_key]
        return scores.copy(), cached_labels.copy(), list(ids)

    nebula_model = get_nebula_model(tokenizer, seq_len, device)

    scores: List[float] = []
    kept_labels: List[int] = []
    ids: List[str] = []
    pending_features: List[np.ndarray] = []
    pending_labels: List[int] = []
    pending_ids: List[str] = []
    pending_cache_keys: List[Tuple[str, str, int]] = []

    def flush_batch() -> None:
        if not pending_features:
            return
        probs = predict_probabilities(nebula_model, pending_features, device=device)
        prob_list = [float(prob) for prob in probs.tolist()]
        scores.extend(prob_list)
        kept_labels.extend(pending_labels)
        ids.extend(pending_ids)
        if use_cache:
            for cache_key_item, prob, sample_id in zip(pending_cache_keys, prob_list, pending_ids):
                _FILE_SCORE_CACHE[cache_key_item] = (prob, sample_id)
        pending_features.clear()
        pending_labels.clear()
        pending_ids.clear()
        pending_cache_keys.clear()

    iterator = list(zip(paths, labels))
    if progress:
        iterator = tqdm(iterator, desc="Nebula Inference", total=len(iterator))

    for raw_path, label in iterator:
        path = Path(raw_path)
        file_cache_key = (str(path.resolve()), tokenizer, seq_len)
        if use_cache:
            cached_file = _FILE_SCORE_CACHE.get(file_cache_key)
            if cached_file is not None:
                prob, sample_id = cached_file
                scores.append(float(prob))
                kept_labels.append(int(label))
                ids.append(sample_id)
                continue
        try:
            report = load_json(path)
            report = convert_to_nebula_report(report, path)
            if report is None:
                continue
            features = nebula_model.preprocess(report)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("处理失败 %s: %s", path.name, exc)
            continue
        if features is None:
            LOGGER.warning("处理失败 %s: 预处理结果为空", path.name)
            continue
        if getattr(features, "ndim", 1) == 2:
            features = features[0]
        pending_features.append(np.asarray(features, dtype=np.int64))
        pending_labels.append(int(label))
        pending_ids.append(path.stem)
        pending_cache_keys.append(file_cache_key)
        if len(pending_features) >= batch_size:
            flush_batch()

    flush_batch()

    if not scores:
        raise RuntimeError("未成功生成任何 Nebula 评分，请检查 JSON 格式或依赖。")

    scores_arr = np.array(scores, dtype=float)
    labels_arr = np.array(kept_labels, dtype=int)
    if use_cache:
        _SCORE_CACHE[cache_key] = (scores_arr.copy(), labels_arr.copy(), list(ids))
    return scores_arr, labels_arr, ids


def quantile_threshold(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> Dict[str, float]:
    mask_neg = y_true == 0
    mask_pos = y_true == 1
    if mask_neg.sum() == 0 or mask_pos.sum() == 0:
        raise ValueError("正负样本数量不足，无法计算阈值。")
    neg_scores = y_score[mask_neg]
    tau = float(np.quantile(neg_scores, 1 - target_fpr))
    preds = (y_score >= tau).astype(int)
    fpr = float(((preds == 1) & mask_neg).sum() / mask_neg.sum())
    tpr = float(((preds == 1) & mask_pos).sum() / mask_pos.sum())

    def _safe_stats(arr: np.ndarray) -> Tuple[float, float]:
        if arr.size == 0:
            return 0.0, 1.0
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        if std == 0:
            std = 1.0
        return float(arr.mean()), std

    mu_all, std_all = _safe_stats(y_score)
    z_all = (tau - mu_all) / std_all
    return {"tau": tau, "fpr": fpr, "tpr": tpr, "z_all": float(z_all)}


def evaluate_at_threshold(y_true: np.ndarray, y_score: np.ndarray, tau: float) -> Dict[str, float]:
    preds = (y_score >= tau).astype(int)
    macro_f1 = float(f1_score(y_true, preds, average="macro"))
    auroc = float(roc_auc_score(y_true, y_score))
    aupr = float(average_precision_score(y_true, y_score))
    accuracy = float(accuracy_score(y_true, preds))
    mask_neg = y_true == 0
    mask_pos = y_true == 1
    fpr = float(((preds == 1) & mask_neg).sum() / max(mask_neg.sum(), 1))
    tpr = float(((preds == 1) & mask_pos).sum() / max(mask_pos.sum(), 1))
    return {
        "macro_f1": macro_f1,
        "auroc": auroc,
        "aupr": aupr,
        "accuracy": accuracy,
        "fpr": fpr,
        "tpr": tpr,
    }


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


def evaluate_holdout(paths: Sequence[str], labels: Sequence[int], cfg: TrainConfig) -> Dict:
    labels_arr = np.array(labels, dtype=int)
    train_idx, val_idx, test_idx = stratified_holdout_indices(labels_arr, cfg.val_ratio, cfg.test_ratio, cfg.seed)
    device = prepare_device(cfg.device)
    scores, kept_labels, ids = compute_nebula_scores(
        paths,
        labels,
        tokenizer=cfg.tokenizer,
        seq_len=cfg.seq_len,
        device=device,
        batch_size=cfg.batch_size,
        progress=cfg.progress,
        use_cache=cfg.cache_scores,
    )
    if len(scores) != len(labels_arr):
        raise RuntimeError("Nebula 推理后样本数量发生变化，当前 holdout 模式要求所有样本均可解析。")
    id_stats = quantile_threshold(kept_labels[val_idx], scores[val_idx], cfg.target_fpr)
    val_metrics = evaluate_at_threshold(kept_labels[val_idx], scores[val_idx], id_stats["tau"])
    test_metrics = evaluate_at_threshold(kept_labels[test_idx], scores[test_idx], id_stats["tau"])

    summary = {
        "mode": "holdout",
        "tokenizer": cfg.tokenizer,
        "target_fpr": cfg.target_fpr,
        "val_ratio": cfg.val_ratio,
        "test_ratio": cfg.test_ratio,
        "best": {
            "threshold": id_stats["tau"],
            "tpr_id": id_stats["tpr"],
            "fpr_id": id_stats["fpr"],
            "z_id": id_stats["z_all"],
            "macro_f1": test_metrics["macro_f1"],
            "auroc": test_metrics["auroc"],
            "aupr": test_metrics["aupr"],
            "accuracy": test_metrics["accuracy"],
            "tpr_ood": test_metrics["tpr"],
            "fpr_ood": test_metrics["fpr"],
        },
        "indices": {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
        },
    }
    if cfg.save_raw:
        summary["raw_scores"] = {
            "ids": ids,
            "scores": scores.tolist(),
            "labels": kept_labels.tolist(),
        }
    return summary


def evaluate_cv(paths: Sequence[str], labels: Sequence[int], cfg: TrainConfig) -> Dict:
    device = prepare_device(cfg.device)
    scores, kept_labels, ids = compute_nebula_scores(
        paths,
        labels,
        tokenizer=cfg.tokenizer,
        seq_len=cfg.seq_len,
        device=device,
        batch_size=cfg.batch_size,
        progress=cfg.progress,
        use_cache=cfg.cache_scores,
    )
    label_arr = np.array(labels, dtype=int)
    if len(scores) != len(label_arr):
        raise RuntimeError("Nebula 推理后样本数量发生变化，当前 cv 模式要求所有样本均可解析。")
    splits = max(2, min(cfg.n_splits, int(np.bincount(label_arr).min())))
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=cfg.seed)
    fold_summaries: List[Dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(scores, label_arr)):
        id_stats = quantile_threshold(label_arr[train_idx], scores[train_idx], cfg.target_fpr)
        val_metrics = evaluate_at_threshold(label_arr[val_idx], scores[val_idx], id_stats["tau"])
        fold_summaries.append(
            {
                "fold": fold_idx,
                "threshold": id_stats["tau"],
                "tpr_id": id_stats["tpr"],
                "fpr_id": id_stats["fpr"],
                "z_id": id_stats["z_all"],
                "macro_f1": val_metrics["macro_f1"],
                "auroc": val_metrics["auroc"],
                "aupr": val_metrics["aupr"],
                "accuracy": val_metrics["accuracy"],
                "tpr_val": val_metrics["tpr"],
                "fpr_val": val_metrics["fpr"],
            }
        )

    best_keys = {
        key
        for fold in fold_summaries
        for key, value in fold.items()
        if key != "fold" and isinstance(value, (int, float))
    }
    best_mean: Dict[str, float] = {}
    best_std: Dict[str, float] = {}
    for key in sorted(best_keys):
        values = [float(fold[key]) for fold in fold_summaries if isinstance(fold.get(key), (int, float))]
        best_mean[key] = float(statistics.mean(values))
        best_std[key] = float(statistics.stdev(values)) if len(values) > 1 else 0.0

    summary = {
        "mode": "cv",
        "tokenizer": cfg.tokenizer,
        "target_fpr": cfg.target_fpr,
        "cv_splits": splits,
        "folds": fold_summaries,
        "best": best_mean,
        "best_std": best_std,
    }
    if cfg.save_raw:
        summary["raw_scores"] = {
            "ids": ids,
            "scores": scores.tolist(),
            "labels": kept_labels.tolist(),
        }
    return summary


def evaluate_explicit_split(
    train_paths: Sequence[str],
    train_labels: Sequence[int],
    val_paths: Sequence[str],
    val_labels: Sequence[int],
    test_paths: Sequence[str],
    test_labels: Sequence[int],
    cfg: TrainConfig,
    *,
    mode: str,
) -> Dict:
    device = prepare_device(cfg.device)
    train_scores, y_train, train_ids = compute_nebula_scores(
        train_paths,
        train_labels,
        tokenizer=cfg.tokenizer,
        seq_len=cfg.seq_len,
        device=device,
        batch_size=cfg.batch_size,
        progress=cfg.progress,
        use_cache=cfg.cache_scores,
    )
    val_scores, y_val, val_ids = compute_nebula_scores(
        val_paths,
        val_labels,
        tokenizer=cfg.tokenizer,
        seq_len=cfg.seq_len,
        device=device,
        batch_size=cfg.batch_size,
        progress=cfg.progress,
        use_cache=cfg.cache_scores,
    )
    test_scores, y_test, test_ids = compute_nebula_scores(
        test_paths,
        test_labels,
        tokenizer=cfg.tokenizer,
        seq_len=cfg.seq_len,
        device=device,
        batch_size=cfg.batch_size,
        progress=cfg.progress,
        use_cache=cfg.cache_scores,
    )

    id_stats = quantile_threshold(y_val, val_scores, cfg.target_fpr)
    test_metrics = evaluate_at_threshold(y_test, test_scores, id_stats["tau"])

    summary = {
        "mode": mode,
        "tokenizer": cfg.tokenizer,
        "device": device,
        "target_fpr": cfg.target_fpr,
        "num_samples_train": int(len(y_train)),
        "num_samples_val": int(len(y_val)),
        "num_samples_test": int(len(y_test)),
        "best": {
            "threshold": id_stats["tau"],
            "tpr_id": id_stats["tpr"],
            "fpr_id": id_stats["fpr"],
            "z_id": id_stats["z_all"],
            "macro_f1": test_metrics["macro_f1"],
            "auroc": test_metrics["auroc"],
            "aupr": test_metrics["aupr"],
            "accuracy": test_metrics["accuracy"],
            "tpr_ood": test_metrics["tpr"],
            "fpr_ood": test_metrics["fpr"],
            "num_samples_test": int(len(y_test)),
            "num_pos_samples_test": int((y_test == 1).sum()),
            "num_neg_samples_test": int((y_test == 0).sum()),
        },
    }
    if cfg.save_raw:
        summary["raw_scores"] = {
            "train": {"ids": train_ids, "scores": train_scores.tolist(), "labels": y_train.tolist()},
            "val": {"ids": val_ids, "scores": val_scores.tolist(), "labels": y_val.tolist()},
            "test": {"ids": test_ids, "scores": test_scores.tolist(), "labels": y_test.tolist()},
        }
    return summary


def init_wandb(cfg: TrainConfig) -> Optional[str]:
    if not cfg.use_wandb or not WANDB_AVAILABLE:
        return None
    try:
        if cfg.wandb_api_key:
            wandb.login(key=cfg.wandb_api_key)
        run_name = f"nebula-{cfg.split_mode}"
        if cfg.run_tag:
            run_name += f"-{cfg.run_tag}"
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=run_name,
            config=asdict(cfg),
            tags=[cfg.split_mode, "nebula_baseline"],
            reinit=True,
        )
        run = getattr(wandb, "run", None)
        return getattr(run, "id", None)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("W&B 初始化失败: %s", exc)
        return None


def get_wandb_run():
    if not WANDB_AVAILABLE or wandb is None:
        return None
    return getattr(wandb, "run", None)


def log_metrics_to_wandb(metrics: Dict, prefix: str = "") -> None:
    if get_wandb_run() is None:
        return
    payload = {}
    for key, value in metrics.items():
        payload[f"{prefix}/{key}" if prefix else key] = value
    wandb.log(payload)


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
            best_mean[key] = float(statistics.mean(values))
            best_std[key] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    return {
        "mode": summaries[0].get("mode"),
        "tokenizer": cfg.tokenizer,
        "seeds": seeds,
        "per_seed": summaries,
        "best_mean": best_mean,
        "best_std": best_std,
        "config": {k: v for k, v in asdict(cfg).items() if k not in {"config_path", "seed"}},
    }


def save_summary(summary: Dict, path: Optional[str]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    LOGGER.info("结果已保存至 %s", out_path)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def run_experiment(cfg: TrainConfig) -> Dict:
    init_wandb(cfg)

    if cfg.split_mode == "holdout":
        paths, labels = load_holdout_dataset(cfg)
        summary = evaluate_holdout(paths, labels, cfg)
    elif cfg.split_mode == "cv":
        paths, labels = load_holdout_dataset(cfg)
        summary = evaluate_cv(paths, labels, cfg)
    elif cfg.split_mode == "dir":
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = load_dir_splits(cfg)
        summary = evaluate_explicit_split(
            train_paths,
            train_labels,
            val_paths,
            val_labels,
            test_paths,
            test_labels,
            cfg,
            mode="holdout-dir",
        )
    elif cfg.split_mode == "split_spec":
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = load_split_spec_splits(cfg)
        summary = evaluate_explicit_split(
            train_paths,
            train_labels,
            val_paths,
            val_labels,
            test_paths,
            test_labels,
            cfg,
            mode="holdout-split_spec",
        )
        summary["split_name"] = cfg.split_name
        summary["split_spec_path"] = cfg.split_spec_path
    else:
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = load_time_ood_splits(cfg)
        summary = evaluate_explicit_split(
            train_paths,
            train_labels,
            val_paths,
            val_labels,
            test_paths,
            test_labels,
            cfg,
            mode="holdout-time_ood",
        )

    if "best" in summary:
        log_metrics_to_wandb(summary["best"], prefix=cfg.split_mode)
        run = get_wandb_run()
        if run is not None:
            run.summary.update(
                {
                    "best_macro_f1": summary["best"].get("macro_f1", 0),
                    "best_auroc": summary["best"].get("auroc", 0),
                    "best_aupr": summary["best"].get("aupr", 0),
                    "best_threshold": summary["best"].get("threshold", 0),
                }
            )
            wandb.finish()
    return summary


def main() -> None:
    setup_logging()
    cfg = parse_config()
    LOGGER.info("实验配置：%s", cfg)
    if cfg.seeds:
        per_seed = []
        for seed in cfg.seeds:
            seed_cfg = replace(cfg, seed=seed, seeds=None, out=None)
            LOGGER.info("运行种子 %d", seed)
            seed_summary = run_experiment(seed_cfg)
            seed_summary["seed"] = seed
            per_seed.append(seed_summary)
        aggregated = aggregate_seed_summaries(cfg, per_seed)
        print(json.dumps(aggregated, ensure_ascii=False, indent=2))
        save_summary(aggregated, cfg.out)
    else:
        summary = run_experiment(cfg)
        summary["seed"] = cfg.seed
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        save_summary(summary, cfg.out)


if __name__ == "__main__":
    main()
