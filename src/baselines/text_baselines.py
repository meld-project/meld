#!/usr/bin/env python3
"""
Baseline text classifiers (TF-IDF word/char) for MELD experiments.

This script mirrors the interface of src/lec/train_lec.py so it can be
scheduled by the same wrapper. It supports holdout / CV splits, multiple
seeds, and exports JSON summaries with the same metric names.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field, replace, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import statistics

# W&B integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False
    wandb = None

LOGGER = logging.getLogger("text_baselines")


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    md_dir: Optional[str] = None
    encoder: str = "tfidf_word"  # tfidf_word / tfidf_char
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
    n_splits: int = 10
    target_fpr: float = 0.01
    out: Optional[str] = None
    save_raw: bool = False
    progress: bool = False

    # W&B 配置
    use_wandb: bool = False  # 是否启用 W&B 记录
    wandb_project: str = "lec-experiments"  # W&B 项目名
    wandb_entity: Optional[str] = None  # W&B 实体名（可选）
    wandb_api_key: Optional[str] = None  # W&B API 密钥
    run_tag: str = ""  # 运行标签

    config_path: Optional[str] = field(default=None, repr=False, compare=False)

    def validate(self) -> None:
        if self.split_mode not in {"holdout", "cv", "dir", "time_ood", "split_spec"}:
            raise ValueError("split_mode 必须是 holdout/cv/dir/time_ood/split_spec")
        if self.split_mode == "holdout":
            if not (0 < self.val_ratio < 1) or not (0 < self.test_ratio < 1):
                raise ValueError("val_ratio/test_ratio 必须在 (0,1) 内")
            if self.val_ratio + self.test_ratio >= 0.95:
                raise ValueError("val_ratio + test_ratio 过大")
        if self.split_mode in {"holdout", "cv"} and not self.md_dir:
            raise ValueError("holdout/cv 模式需要 md_dir")
        if self.split_mode == "dir":
            missing = [name for name in ("train_dir", "val_dir", "test_dir") if not getattr(self, name)]
            if missing:
                raise ValueError(f"dir 模式需要提供 {missing}")
        if self.split_mode == "split_spec":
            missing = [name for name in ("split_spec_path", "split_name") if not getattr(self, name)]
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
            raise ValueError("target_fpr 必须在 (0,1) 内")
        if self.seeds:
            cleaned = sorted(dict.fromkeys(int(s) for s in self.seeds))
            if not cleaned:
                raise ValueError("seeds 不能为空")
            self.seeds = cleaned


def load_config_file(path: Optional[str]) -> Dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("配置文件需为 JSON 对象")
    return data


def build_parser(defaults: Dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="JSON 配置文件路径")
    parser.add_argument("--md_dir", type=str, default=defaults.get("md_dir"))
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
    parser.add_argument("--encoder", choices=["tfidf_word", "tfidf_char"], default=defaults.get("encoder", "tfidf_word"))
    parser.add_argument("--split_mode", choices=["holdout", "cv", "dir", "time_ood", "split_spec"], default=defaults.get("split_mode", "holdout"))
    parser.add_argument("--val_ratio", type=float, default=defaults.get("val_ratio", 0.1))
    parser.add_argument("--test_ratio", type=float, default=defaults.get("test_ratio", 0.2))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--seeds", type=str, default=None, help="逗号分隔的多随机种子，例如 40,41,42")
    parser.add_argument("--limit", type=int, default=defaults.get("limit"))
    parser.add_argument("--n_splits", type=int, default=defaults.get("n_splits", 10))
    parser.add_argument("--target_fpr", type=float, default=defaults.get("target_fpr", 0.01))
    parser.add_argument("--out", type=str, default=defaults.get("out"))
    parser.add_argument("--save_raw", action="store_true", default=defaults.get("save_raw", False))
    parser.add_argument("--progress", action="store_true", default=defaults.get("progress", False))
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
            parsed = [s.strip() for s in seeds_value.split(",")]
            seed_list = [int(s) for s in parsed if s]
        elif isinstance(seeds_value, list):
            seed_list = [int(s) for s in seeds_value]
        else:
            raise ValueError("--seeds 参数解析失败")
        merged["seeds"] = seed_list
        merged.setdefault("seed", seed_list[0])
    config = TrainConfig(**merged)
    config.validate()
    return config


# --------------------------------------------------------------------------- #
# 数据集处理
# --------------------------------------------------------------------------- #


def load_md_reports(md_dir: str, limit: Optional[int] = None) -> Tuple[List[str], List[int], List[str]]:
    md_dir_path = Path(md_dir)
    if not md_dir_path.exists():
        raise FileNotFoundError(f"Markdown 目录不存在：{md_dir}")

    md_paths: List[Path] = []
    for path in md_dir_path.rglob("*.md"):
        md_paths.append(path)
    md_paths.sort()

    texts: List[str] = []
    labels: List[int] = []
    kept_paths: List[str] = []
    skipped_unknown = 0
    for md_path in md_paths:
        label_path = md_path.with_suffix(".label")
        label: Optional[int] = None
        if label_path.exists():
            try:
                with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
                    label = int(f.read().strip())
            except Exception:
                label = None
        if label is None:
            parts = [p.lower() for p in md_path.parts]
            if any("malicious" in p or "black" in p for p in parts):
                label = 1
            elif any("benign" in p or "white" in p for p in parts):
                label = 0
            elif "unknown" in parts:
                skipped_unknown += 1
                continue
            else:
                continue
        with open(md_path, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
        labels.append(label)
        kept_paths.append(str(md_path))

    if skipped_unknown:
        LOGGER.info("跳过 unknown 样本 %d 个。", skipped_unknown)

    if limit is not None and len(labels) > limit:
        LOGGER.info("限制样本数至 %d，进行分层抽样。", limit)
        y_arr = np.array(labels, dtype=int)
        idx_all = np.arange(len(labels))
        sss = StratifiedShuffleSplit(n_splits=1, train_size=limit, random_state=42)
        keep_idx, _ = next(sss.split(idx_all, y_arr))
        texts = [texts[i] for i in keep_idx]
        labels = [labels[i] for i in keep_idx]
        kept_paths = [kept_paths[i] for i in keep_idx]

    return texts, labels, kept_paths


def subsample_dataset(
    texts: Sequence[str],
    labels: Sequence[int],
    paths: Sequence[str],
    limit: Optional[int],
    seed: int,
) -> Tuple[List[str], List[int], List[str]]:
    texts_list = list(texts)
    labels_list = list(labels)
    paths_list = list(paths)
    if limit is None or len(labels_list) <= limit:
        return texts_list, labels_list, paths_list

    LOGGER.info("限制样本数至 %d，进行分层抽样。", limit)
    y_arr = np.array(labels_list, dtype=int)
    idx_all = np.arange(len(labels_list))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=limit, random_state=seed)
    keep_idx, _ = next(sss.split(idx_all, y_arr))
    return (
        [texts_list[i] for i in keep_idx],
        [labels_list[i] for i in keep_idx],
        [paths_list[i] for i in keep_idx],
    )


def load_texts_from_paths(paths: Sequence[str]) -> List[str]:
    texts: List[str] = []
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            texts.append(fh.read())
    return texts


# --------------------------------------------------------------------------- #
# 特征与模型
# --------------------------------------------------------------------------- #


def build_pipeline(encoder: str, seed: int) -> Pipeline:
    if encoder == "tfidf_word":
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=200000,
            token_pattern=r"(?u)\b\w+\b",
        )
    elif encoder == "tfidf_char":
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=200000,
        )
    else:
        raise ValueError(f"未知编码器：{encoder}")

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
    )
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", clf),
    ])
    return pipeline


def predict_proba(pipeline: Pipeline, texts: Sequence[str]) -> np.ndarray:
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipeline.predict_proba(texts)[:, 1]
    scores = pipeline.decision_function(texts)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores


# --------------------------------------------------------------------------- #
# FPR / TPR 计算
# --------------------------------------------------------------------------- #


def quantile_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float) -> Dict[str, float]:
    mask_neg = y_true == 0
    mask_pos = y_true == 1
    if mask_neg.sum() == 0 or mask_pos.sum() == 0:
        raise ValueError("正负样本数量不足，无法计算阈值。")
    neg_scores = y_prob[mask_neg]
    tau = float(np.quantile(neg_scores, 1 - target_fpr))
    preds = (y_prob >= tau).astype(int)
    fpr = float(((preds == 1) & mask_neg).sum() / mask_neg.sum())
    tpr = float(((preds == 1) & mask_pos).sum() / mask_pos.sum())
    mu_neg, std_neg = float(neg_scores.mean()), float(neg_scores.std(ddof=1) + 1e-8)
    mu_all, std_all = float(y_prob.mean()), float(y_prob.std(ddof=1) + 1e-8)
    z_neg = (tau - mu_neg) / std_neg
    z_all = (tau - mu_all) / std_all
    return {"tau": tau, "fpr": fpr, "tpr": tpr, "z_neg": float(z_neg), "z_all": float(z_all)}


def _evaluate_fixed_threshold(y_true: np.ndarray, y_prob: np.ndarray, tau: float) -> Dict[str, float]:
    mask_neg = y_true == 0
    mask_pos = y_true == 1
    preds = (y_prob >= tau).astype(int)
    fpr = float(((preds == 1) & mask_neg).sum() / max(mask_neg.sum(), 1))
    tpr = float(((preds == 1) & mask_pos).sum() / max(mask_pos.sum(), 1))
    mu, std = float(y_prob.mean()), float(y_prob.std(ddof=1) + 1e-8)
    return {"tpr": tpr, "fpr": fpr, "z_all": (tau - mu) / std}


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #


def evaluate_holdout(
    texts: Sequence[str],
    labels: Sequence[int],
    cfg: TrainConfig,
) -> Dict:
    texts = np.array(texts)
    labels = np.array(labels, dtype=int)
    train_idx, val_idx, test_idx = stratified_holdout_indices(labels, cfg.val_ratio, cfg.test_ratio, cfg.seed)

    pipeline = build_pipeline(cfg.encoder, cfg.seed)
    pipeline.fit(texts[train_idx], labels[train_idx])

    prob_val = predict_proba(pipeline, texts[val_idx])
    prob_test = predict_proba(pipeline, texts[test_idx])

    id_stats = quantile_threshold(labels[val_idx], prob_val, cfg.target_fpr)
    ood_stats = _evaluate_fixed_threshold(labels[test_idx], prob_test, id_stats["tau"])

    preds = (prob_test >= id_stats["tau"]).astype(int)
    macro_f1 = float(f1_score(labels[test_idx], preds, average="macro"))
    auroc = float(roc_auc_score(labels[test_idx], prob_test))
    aupr = float(average_precision_score(labels[test_idx], prob_test))
    accuracy = float(accuracy_score(labels[test_idx], preds))

    return {
        "mode": "holdout",
        "encoder": cfg.encoder,
        "val_ratio": cfg.val_ratio,
        "test_ratio": cfg.test_ratio,
        "target_fpr": cfg.target_fpr,
        "best": {
            "threshold": id_stats["tau"],
            "tpr_id": id_stats["tpr"],
            "fpr_id": id_stats["fpr"],
            "z_id": id_stats["z_all"],
            "tpr_ood": ood_stats["tpr"],
            "fpr_ood": ood_stats["fpr"],
            "z_ood": ood_stats["z_all"],
            "macro_f1": macro_f1,
            "auroc": auroc,
            "aupr": aupr,
            "accuracy": accuracy,
        },
    }


def evaluate_explicit_split(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    test_texts: Sequence[str],
    test_labels: Sequence[int],
    cfg: TrainConfig,
    *,
    mode: str,
) -> Dict:
    train_arr = np.array(train_texts)
    val_arr = np.array(val_texts)
    test_arr = np.array(test_texts)
    y_train = np.array(train_labels, dtype=int)
    y_val = np.array(val_labels, dtype=int)
    y_test = np.array(test_labels, dtype=int)

    pipeline = build_pipeline(cfg.encoder, cfg.seed)
    pipeline.fit(train_arr, y_train)

    prob_val = predict_proba(pipeline, val_arr)
    prob_test = predict_proba(pipeline, test_arr)

    id_stats = quantile_threshold(y_val, prob_val, cfg.target_fpr)
    ood_stats = _evaluate_fixed_threshold(y_test, prob_test, id_stats["tau"])

    preds = (prob_test >= id_stats["tau"]).astype(int)
    macro_f1 = float(f1_score(y_test, preds, average="macro"))
    auroc = float(roc_auc_score(y_test, prob_test))
    aupr = float(average_precision_score(y_test, prob_test))
    accuracy = float(accuracy_score(y_test, preds))

    return {
        "mode": mode,
        "encoder": cfg.encoder,
        "target_fpr": cfg.target_fpr,
        "num_samples_train": int(len(y_train)),
        "num_samples_val": int(len(y_val)),
        "num_samples_test": int(len(y_test)),
        "best": {
            "threshold": id_stats["tau"],
            "tpr_id": id_stats["tpr"],
            "fpr_id": id_stats["fpr"],
            "z_id": id_stats["z_all"],
            "tpr_ood": ood_stats["tpr"],
            "fpr_ood": ood_stats["fpr"],
            "z_ood": ood_stats["z_all"],
            "macro_f1": macro_f1,
            "auroc": auroc,
            "aupr": aupr,
            "accuracy": accuracy,
            "num_samples_test": int(len(y_test)),
            "num_pos_samples_test": int((y_test == 1).sum()),
            "num_neg_samples_test": int((y_test == 0).sum()),
        },
    }


def evaluate_cv(texts: Sequence[str], labels: Sequence[int], cfg: TrainConfig) -> Dict:
    texts = np.array(texts)
    labels = np.array(labels, dtype=int)
    pipeline = build_pipeline(cfg.encoder, cfg.seed)
    splits = max(2, min(cfg.n_splits, int(np.bincount(labels).min())))

    probs = np.zeros_like(labels, dtype=float)
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=cfg.seed)
    for train_idx, val_idx in skf.split(texts, labels):
        pipeline.fit(texts[train_idx], labels[train_idx])
        probs[val_idx] = predict_proba(pipeline, texts[val_idx])

    id_stats = quantile_threshold(labels, probs, cfg.target_fpr)
    preds = (probs >= id_stats["tau"]).astype(int)
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    auroc = float(roc_auc_score(labels, probs))
    aupr = float(average_precision_score(labels, probs))
    accuracy = float(accuracy_score(labels, preds))

    return {
        "mode": "cv",
        "encoder": cfg.encoder,
        "cv_splits": splits,
        "target_fpr": cfg.target_fpr,
        "best": {
            "threshold": id_stats["tau"],
            "tpr_id": id_stats["tpr"],
            "fpr_id": id_stats["fpr"],
            "z_id": id_stats["z_all"],
            "macro_f1": macro_f1,
            "auroc": auroc,
            "aupr": aupr,
            "accuracy": accuracy,
        },
    }


def load_dir_splits(cfg: TrainConfig) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    assert cfg.train_dir and cfg.val_dir and cfg.test_dir
    train_texts, train_labels, train_paths = load_md_reports(cfg.train_dir, cfg.limit)
    val_texts, val_labels, val_paths = load_md_reports(cfg.val_dir, cfg.limit)
    test_texts, test_labels, test_paths = load_md_reports(cfg.test_dir, cfg.limit)
    if not train_texts or not val_texts or not test_texts:
        raise RuntimeError("dir 模式中存在空数据集。")
    return (
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        test_texts,
        test_labels,
    )


def load_split_spec_splits(cfg: TrainConfig) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    assert cfg.split_spec_path and cfg.split_name
    from src.utils.split_spec import load_named_split, slice_split_docs

    docs, fold, _protocol = load_named_split(cfg.split_spec_path, cfg.split_name)
    train_paths, train_labels = slice_split_docs(docs, fold["train_ids"])
    val_paths, val_labels = slice_split_docs(docs, fold["val_ids"])
    test_paths, test_labels = slice_split_docs(docs, fold["test_ids"])

    train_texts = load_texts_from_paths(train_paths)
    val_texts = load_texts_from_paths(val_paths)
    test_texts = load_texts_from_paths(test_paths)

    train_texts, train_labels, _ = subsample_dataset(train_texts, train_labels, train_paths, cfg.limit, cfg.seed)
    val_texts, val_labels, _ = subsample_dataset(val_texts, val_labels, val_paths, cfg.limit, cfg.seed + 1)
    test_texts, test_labels, _ = subsample_dataset(test_texts, test_labels, test_paths, cfg.limit, cfg.seed + 2)
    if not train_texts or not val_texts or not test_texts:
        raise RuntimeError("split_spec 模式中存在空数据集。")
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def load_time_ood_splits(cfg: TrainConfig) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    from src.utils.time_split import load_manifest_with_time, split_by_time

    assert cfg.malicious_manifest_path and cfg.benign_manifest_path
    assert cfg.mal_dir and cfg.benign_dir

    mal_records, ben_records = load_manifest_with_time(
        cfg.malicious_manifest_path,
        cfg.benign_manifest_path,
        mal_dir=cfg.mal_dir,
        benign_dir=cfg.benign_dir,
    )
    train_end_date = cfg.train_end_date or "2025-04-23 08:08:46"
    val_end_date = cfg.val_end_date or "2025-06-14 11:39:39"
    (
        train_paths,
        train_labels,
        _train_ids,
        val_paths,
        val_labels,
        _val_ids,
        test_paths,
        test_labels,
        _test_ids,
    ) = split_by_time(
        mal_records,
        ben_records,
        train_end_date,
        val_end_date,
        mal_dir=cfg.mal_dir,
        benign_dir=cfg.benign_dir,
        benign_split_strategy=cfg.benign_split_strategy,
        random_seed=cfg.seed,
    )

    train_texts = load_texts_from_paths(train_paths)
    val_texts = load_texts_from_paths(val_paths)
    test_texts = load_texts_from_paths(test_paths)

    train_texts, train_labels, _ = subsample_dataset(train_texts, train_labels, train_paths, cfg.limit, cfg.seed)
    val_texts, val_labels, _ = subsample_dataset(val_texts, val_labels, val_paths, cfg.limit, cfg.seed)
    test_texts, test_labels, _ = subsample_dataset(test_texts, test_labels, test_paths, cfg.limit, cfg.seed)
    if not train_texts or not val_texts or not test_texts:
        raise RuntimeError("time_ood 模式中存在空数据集。")
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


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
# W&B 集成
# --------------------------------------------------------------------------- #


def init_wandb(cfg: TrainConfig) -> Optional[str]:
    """初始化 W&B，返回 run_id"""
    if not cfg.use_wandb or not WANDB_AVAILABLE:
        LOGGER.info("W&B 未启用或不可用，跳过初始化")
        return None

    try:
        # 设置 API key（如果提供）
        if cfg.wandb_api_key:
            wandb.login(key=cfg.wandb_api_key)

        # 构建运行名称
        run_name = f"text-{cfg.encoder}-{cfg.split_mode}"
        if cfg.run_tag:
            run_name += f"-{cfg.run_tag}"

        # 初始化 W&B
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=run_name,
            config=asdict(cfg),
            tags=[cfg.split_mode, cfg.encoder, "text_baseline"],
            reinit=True,
        )

        LOGGER.info(f"W&B 初始化成功，项目: {cfg.wandb_project}, 运行: {run_name}")
        run = getattr(wandb, "run", None)
        return getattr(run, "id", None)

    except Exception as exc:
        LOGGER.warning(f"W&B 初始化失败: {exc}")
        return None


def get_wandb_run():
    if not WANDB_AVAILABLE or wandb is None:
        return None
    return getattr(wandb, "run", None)


def log_metrics_to_wandb(metrics: Dict, prefix: str = "") -> None:
    """将指标记录到 W&B"""
    if get_wandb_run() is None:
        return

    # 添加前缀
    wandb_metrics = {}
    for key, value in metrics.items():
        wandb_key = f"{prefix}/{key}" if prefix else key
        wandb_metrics[wandb_key] = value

    wandb.log(wandb_metrics)


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #


def aggregate_seed_summaries(cfg: TrainConfig, summaries: List[Dict]) -> Dict:
    seeds = [s["seed"] for s in summaries]
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
    aggregated = {
        "mode": summaries[0].get("mode"),
        "encoder": cfg.encoder,
        "seeds": seeds,
        "per_seed": summaries,
        "best_mean": best_mean,
        "best_std": best_std,
        "config": {k: v for k, v in asdict(cfg).items() if k not in {"config_path", "seed"}},
    }
    return aggregated


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def run_experiment(cfg: TrainConfig) -> Dict:
    # 初始化 W&B
    run_id = init_wandb(cfg)

    if cfg.split_mode in {"holdout", "cv"}:
        assert cfg.md_dir is not None
        texts, labels, _ = load_md_reports(cfg.md_dir, cfg.limit)
        if not texts:
            raise RuntimeError("未读取到任何样本，请检查 Markdown 目录。")
        if cfg.split_mode == "holdout":
            summary = evaluate_holdout(texts, labels, cfg)
        else:
            summary = evaluate_cv(texts, labels, cfg)
    elif cfg.split_mode == "dir":
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_dir_splits(cfg)
        summary = evaluate_explicit_split(
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            test_texts,
            test_labels,
            cfg,
            mode="holdout-dir",
        )
    elif cfg.split_mode == "split_spec":
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_split_spec_splits(cfg)
        summary = evaluate_explicit_split(
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            test_texts,
            test_labels,
            cfg,
            mode="holdout-split_spec",
        )
        summary["split_name"] = cfg.split_name
        summary["split_spec_path"] = cfg.split_spec_path
    else:
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_time_ood_splits(cfg)
        summary = evaluate_explicit_split(
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            test_texts,
            test_labels,
            cfg,
            mode="holdout-time_ood",
        )

    # 记录指标到 W&B
    if "best" in summary:
        log_metrics_to_wandb(summary["best"], prefix=cfg.split_mode)
        # 记录摘要
        run = get_wandb_run()
        if run is not None:
            run.summary.update({
                "best_macro_f1": summary["best"].get("macro_f1", 0),
                "best_auroc": summary["best"].get("auroc", 0),
                "best_aupr": summary["best"].get("aupr", 0),
                "best_threshold": summary["best"].get("threshold", 0),
            })
            wandb.finish()
            LOGGER.info("实验结果已记录到 W&B")

    return summary


def save_summary(summary: Dict, path: Optional[str]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    LOGGER.info("结果已保存至 %s", out_path)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def main() -> None:
    setup_logging()
    cfg = parse_config()
    LOGGER.info("实验配置：%s", cfg)

    if cfg.seeds:
        per_seed: List[Dict] = []
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
