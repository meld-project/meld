#!/usr/bin/env python3
"""Embedding-based baselines (Word2Vec / Transformer encoders).

This script mirrors the interface of `src/lec/train_lec.py` so it can be
invoked by the same wrapper. It supports holdout / CV splits, multiple seeds,
and exports JSON summaries compatible with MELD experiments.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, replace, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import statistics

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("需要安装 transformers 和 torch 才能运行 embedding baselines") from exc

try:
    from gensim.models import KeyedVectors
except ImportError:
    KeyedVectors = None  # type: ignore

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# W&B integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False
    wandb = None

LOGGER = logging.getLogger("embedding_baselines")


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    md_dir: str
    encoder: str = "bert"  # bert / bge / word2vec
    model_name: Optional[str] = None  # huggingface 模型名
    word2vec_path: Optional[str] = None
    split_mode: str = "holdout"  # holdout / cv
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    seed: int = 42
    seeds: Optional[List[int]] = None
    limit: Optional[int] = None
    n_splits: int = 10
    target_fpr: float = 0.01
    max_length: int = 1024
    batch_size: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    layerwise: bool = False  # 是否使用分层特征提取（类似 LEC）
    until_layer: Optional[int] = None  # 前向至第几层（1-based），None 表示所有层
    out: Optional[str] = None
    save_raw: bool = False

    # W&B 配置
    use_wandb: bool = False  # 是否启用 W&B 记录
    wandb_project: str = "lec-experiments"  # W&B 项目名
    wandb_entity: Optional[str] = None  # W&B 实体名（可选）
    wandb_api_key: Optional[str] = None  # W&B API 密钥
    run_tag: str = ""  # 运行标签

    config_path: Optional[str] = field(default=None, repr=False, compare=False)

    def validate(self) -> None:
        if self.split_mode not in {"holdout", "cv"}:
            raise ValueError("split_mode 必须是 holdout/cv")
        if self.split_mode == "holdout":
            if not (0 < self.val_ratio < 1) or not (0 < self.test_ratio < 1):
                raise ValueError("val_ratio/test_ratio 必须在 (0,1) 内")
            if self.val_ratio + self.test_ratio >= 0.95:
                raise ValueError("val_ratio + test_ratio 过大")
        if self.limit is not None and self.limit < 2:
            raise ValueError("limit 至少为 2")
        if not (0 < self.target_fpr < 1):
            raise ValueError("target_fpr 必须在 (0,1) 内")
        if self.seeds:
            cleaned = sorted(dict.fromkeys(int(s) for s in self.seeds))
            if not cleaned:
                raise ValueError("seeds 不能为空")
            self.seeds = cleaned
        if self.encoder == "word2vec" and not self.word2vec_path:
            raise ValueError("word2vec 模式需要提供 --word2vec_path")
        if self.encoder in {"bert", "bge"} and not self.model_name:
            raise ValueError("Transformer 模式需要提供 --model_name")


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
    parser.add_argument("--encoder", choices=["bert", "bge", "word2vec"], default=defaults.get("encoder", "bert"))
    parser.add_argument("--model_name", type=str, default=defaults.get("model_name"))
    parser.add_argument("--word2vec_path", type=str, default=defaults.get("word2vec_path"))
    parser.add_argument("--split_mode", choices=["holdout", "cv"], default=defaults.get("split_mode", "holdout"))
    parser.add_argument("--val_ratio", type=float, default=defaults.get("val_ratio", 0.1))
    parser.add_argument("--test_ratio", type=float, default=defaults.get("test_ratio", 0.2))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--seeds", type=str, default=None, help="逗号分隔的多随机种子，例如 40,41,42")
    parser.add_argument("--limit", type=int, default=defaults.get("limit"))
    parser.add_argument("--n_splits", type=int, default=defaults.get("n_splits", 10))
    parser.add_argument("--target_fpr", type=float, default=defaults.get("target_fpr", 0.01))
    parser.add_argument("--max_length", type=int, default=defaults.get("max_length", 1024))
    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 4))
    parser.add_argument("--device", type=str, default=defaults.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--layerwise", action="store_true", default=defaults.get("layerwise", False), help="使用分层特征提取（类似 LEC）")
    parser.add_argument("--until_layer", type=int, default=defaults.get("until_layer"), help="前向至第几层（1-based），None 表示所有层")
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
# 数据读取
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


# --------------------------------------------------------------------------- #
# 编码器
# --------------------------------------------------------------------------- #


class TransformerEncoder:
    def __init__(self, model_name: str, device: str, max_length: int, batch_size: int, layerwise: bool = False) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_kwargs = {}
        if layerwise:
            model_kwargs["output_hidden_states"] = True
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size
        self.layerwise = layerwise
        if hasattr(self.model.config, "num_hidden_layers"):
            self.num_layers = self.model.config.num_hidden_layers
        else:
            self.num_layers = 12  # BERT base default

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """编码为单层特征（最后一层）"""
        features: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state  # [B, L, H]
            mask = inputs["attention_mask"].unsqueeze(-1)  # [B, L, 1]
            token_counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = (hidden * mask).sum(dim=1) / token_counts
            features.append(pooled.cpu().numpy())
        return np.concatenate(features, axis=0)

    @torch.inference_mode()
    def encode_layers(self, texts: Sequence[str], until_layer: Optional[int] = None) -> np.ndarray:
        """编码为多层特征（类似 LEC）"""
        if not self.layerwise:
            raise ValueError("需要 layerwise=True 才能使用 encode_layers")
        
        all_layer_features: List[List[np.ndarray]] = []
        
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            
            # 获取所有层的 hidden states
            hidden_states = outputs.hidden_states[1:]  # 跳过 embedding 层
            if until_layer is not None:
                hidden_states = hidden_states[:until_layer]
            
            mask = inputs["attention_mask"].unsqueeze(-1)  # [B, L, 1]
            token_counts = mask.sum(dim=1).clamp(min=1.0)
            
            # 对每一层进行 mean pooling
            batch_layer_features = []
            for layer_hidden in hidden_states:  # [B, L, H]
                pooled = (layer_hidden * mask).sum(dim=1) / token_counts  # [B, H]
                batch_layer_features.append(pooled.cpu().numpy())
            
            if not all_layer_features:
                all_layer_features = [[] for _ in range(len(batch_layer_features))]
            for layer_idx, layer_feat in enumerate(batch_layer_features):
                all_layer_features[layer_idx].append(layer_feat)
        
        # 合并所有 batch，形状: [num_layers, num_samples, hidden_size]
        layer_features = []
        for layer_feats in all_layer_features:
            layer_features.append(np.concatenate(layer_feats, axis=0))
        
        # 转置为 [num_samples, num_layers, hidden_size]
        features = np.stack(layer_features, axis=1)
        return features


class Word2VecEncoder:
    def __init__(self, path: str) -> None:
        if KeyedVectors is None:
            raise RuntimeError("需要安装 gensim 才能使用 word2vec 编码器")
        self.model = KeyedVectors.load(path)
        self.dim = self.model.vector_size

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        features = np.zeros((len(texts), self.dim), dtype=np.float32)
        for idx, text in enumerate(texts):
            tokens = [tok for tok in text.split() if tok in self.model.key_to_index]
            if tokens:
                vecs = self.model[tokens]
                features[idx] = vecs.mean(axis=0)
            else:
                features[idx] = 0.0
        return features


def build_encoder(cfg: TrainConfig):
    if cfg.encoder in {"bert", "bge"}:
        model_name = cfg.model_name
        assert model_name is not None
        return TransformerEncoder(
            model_name, 
            cfg.device, 
            cfg.max_length, 
            cfg.batch_size,
            layerwise=cfg.layerwise
        )
    elif cfg.encoder == "word2vec":
        assert cfg.word2vec_path is not None
        return Word2VecEncoder(cfg.word2vec_path)
    else:  # pragma: no cover
        raise ValueError(f"未知编码器：{cfg.encoder}")


# --------------------------------------------------------------------------- #
# 评价指标
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
# 评估
# --------------------------------------------------------------------------- #


def evaluate_holdout(features: np.ndarray, labels: np.ndarray, cfg: TrainConfig) -> Dict:
    train_idx, val_idx, test_idx = stratified_holdout_indices(labels, cfg.val_ratio, cfg.test_ratio, cfg.seed)

    scaler = StandardScaler(with_mean=True, with_std=True)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg.seed, solver="lbfgs")
    scaler.fit(features[train_idx])
    X_tr = scaler.transform(features[train_idx])
    clf.fit(X_tr, labels[train_idx])

    def predict(split_idx: np.ndarray) -> np.ndarray:
        return clf.decision_function(scaler.transform(features[split_idx]))

    prob_val = predict(val_idx)
    prob_test = predict(test_idx)

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
        "model_name": cfg.model_name,
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


def evaluate_cv(features: np.ndarray, labels: np.ndarray, cfg: TrainConfig) -> Dict:
    splits = max(2, min(cfg.n_splits, int(np.bincount(labels).min())))
    scaler = StandardScaler(with_mean=True, with_std=True)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg.seed, solver="lbfgs")
    probs = np.zeros_like(labels, dtype=float)
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=cfg.seed)
    for train_idx, val_idx in skf.split(features, labels):
        scaler.fit(features[train_idx])
        clf.fit(scaler.transform(features[train_idx]), labels[train_idx])
        probs[val_idx] = clf.decision_function(scaler.transform(features[val_idx]))

    id_stats = quantile_threshold(labels, probs, cfg.target_fpr)
    preds = (probs >= id_stats["tau"]).astype(int)
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    auroc = float(roc_auc_score(labels, probs))
    aupr = float(average_precision_score(labels, probs))
    accuracy = float(accuracy_score(labels, preds))

    return {
        "mode": "cv",
        "encoder": cfg.encoder,
        "model_name": cfg.model_name,
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


def evaluate_cv_layers(
    X_layers: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
    progress: bool = False,
) -> Dict:
    """逐层评估（类似 LEC）"""
    from tqdm import tqdm
    
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("数据仅包含单一类别，无法进行训练/评估。")
    splits = max(2, min(cfg.n_splits, int(counts.min())))
    LOGGER.info("使用 %d 折交叉验证，评估 %d 层。", splits, X_layers.shape[1])

    results: List[Dict] = []
    best_result: Optional[Dict] = None
    layer_iter: Iterable[int] = range(X_layers.shape[1])
    if progress:
        layer_iter = tqdm(layer_iter, desc="Evaluating layers (CV)")

    for li in layer_iter:
        X = X_layers[:, li, :]
        probs = np.zeros_like(y, dtype=float)
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=cfg.seed)
        
        for train_idx, val_idx in skf.split(X, y):
            scaler = StandardScaler(with_mean=True, with_std=True)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg.seed, solver="lbfgs")
            scaler.fit(X[train_idx])
            clf.fit(scaler.transform(X[train_idx]), y[train_idx])
            probs[val_idx] = clf.decision_function(scaler.transform(X[val_idx]))

        id_stats = quantile_threshold(y, probs, cfg.target_fpr)
        preds = (probs >= id_stats["tau"]).astype(int)
        macro_f1 = float(f1_score(y, preds, average="macro"))
        auroc = float(roc_auc_score(y, probs))
        aupr = float(average_precision_score(y, probs))
        accuracy = float(accuracy_score(y, preds))

        metrics = {
            "layer_index": int(li + 1),
            "threshold": id_stats["tau"],
            "tpr_id": id_stats["tpr"],
            "fpr_id": id_stats["fpr"],
            "z_id": id_stats["z_all"],
            "macro_f1": macro_f1,
            "auroc": auroc,
            "aupr": aupr,
            "accuracy": accuracy,
        }

        log_metrics_to_wandb(metrics, prefix="cv_layers")
        results.append(metrics)

        if best_result is None or metrics["macro_f1"] > best_result["macro_f1"]:
            best_result = metrics

    assert best_result is not None
    
    # 记录层性能图表
    if get_wandb_run() is not None:
        try:
            from src.utils.wandb_viz import log_layer_performance_chart
            log_layer_performance_chart(
                results,
                title=f"BERT Layer Performance - {cfg.split_mode.upper()}"
            )
        except Exception as exc:
            LOGGER.warning(f"层性能图表记录失败: {exc}")
    
    return {
        "mode": "cv",
        "encoder": cfg.encoder,
        "model_name": cfg.model_name,
        "num_layers": int(X_layers.shape[1]),
        "hidden_size": int(X_layers.shape[2]),
        "cv_splits": splits,
        "target_fpr": cfg.target_fpr,
        "best": best_result,
        "all_layers": results,
    }


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
        values = [float(summary["best"][key]) for summary in summaries if key in summary.get("best", {})]
        if values:
            best_mean[key] = float(statistics.mean(values))
            best_std[key] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    aggregated = {
        "mode": summaries[0].get("mode"),
        "encoder": cfg.encoder,
        "model_name": cfg.model_name,
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
        run_name = f"embedding-{cfg.encoder}-{cfg.split_mode}"
        if cfg.run_tag:
            run_name += f"-{cfg.run_tag}"
        if cfg.model_name:
            model_short = cfg.model_name.split("/")[-1]
            run_name += f"-{model_short[:20]}"

        # 初始化 W&B
        tags = [cfg.split_mode, cfg.encoder, "embedding_baseline"]
        if cfg.layerwise:
            tags.append("layerwise")
        
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=run_name,
            config=asdict(cfg),
            tags=tags,
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


def run_experiment(cfg: TrainConfig) -> Dict:
    # 初始化 W&B
    run_id = init_wandb(cfg)

    texts, labels, _ = load_md_reports(cfg.md_dir, cfg.limit)
    if not texts:
        raise RuntimeError("未读取到任何样本，请检查 Markdown 目录。")

    encoder = build_encoder(cfg)
    labels_arr = np.array(labels, dtype=int)

    # 分层模式：类似 LEC 的逐层评估
    if cfg.layerwise and cfg.encoder in {"bert", "bge"}:
        LOGGER.info("使用分层特征提取模式，开始编码 %d 篇文档...", len(texts))
        features_layers = encoder.encode_layers(texts, until_layer=cfg.until_layer)
        LOGGER.info("特征形状: %s (samples, layers, hidden_size)", features_layers.shape)
        
        if cfg.split_mode == "holdout":
            # TODO: 实现 holdout 分层评估
            raise NotImplementedError("holdout 模式的分层评估尚未实现")
        else:
            summary = evaluate_cv_layers(features_layers, labels_arr, cfg, progress=True)
    else:
        # 单层模式：使用最后一层
        LOGGER.info("开始编码 %d 篇文档...", len(texts))
        features = encoder.encode(texts)
        
        if cfg.split_mode == "holdout":
            summary = evaluate_holdout(features, labels_arr, cfg)
        else:
            summary = evaluate_cv(features, labels_arr, cfg)

    # 记录指标到 W&B
    if "best" in summary:
        if cfg.layerwise:
            # 分层模式：记录最佳层信息
            run = get_wandb_run()
            if run is not None:
                run.summary.update({
                    "best_layer": summary["best"].get("layer_index"),
                    "best_macro_f1": summary["best"].get("macro_f1", 0),
                    "best_auroc": summary["best"].get("auroc", 0),
                    "best_aupr": summary["best"].get("aupr", 0),
                    "best_threshold": summary["best"].get("threshold", 0),
                })
        else:
            log_metrics_to_wandb(summary["best"], prefix=cfg.split_mode)
            run = get_wandb_run()
            if run is not None:
                run.summary.update({
                    "best_macro_f1": summary["best"].get("macro_f1", 0),
                    "best_auroc": summary["best"].get("auroc", 0),
                    "best_aupr": summary["best"].get("aupr", 0),
                    "best_threshold": summary["best"].get("threshold", 0),
                })
                # 添加单层实验的详细可视化
                try:
                    from src.utils.wandb_viz import (
                        log_roc_curve,
                        log_pr_curve,
                        log_confusion_matrix,
                        log_score_distribution,
                    )
                    # 需要重新计算预测概率用于可视化
                    if cfg.split_mode == "cv":
                        # CV 模式下需要重新计算
                        scaler = StandardScaler(with_mean=True, with_std=True)
                        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg.seed, solver="lbfgs")
                        probs = np.zeros_like(labels_arr, dtype=float)
                        skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
                        for train_idx, val_idx in skf.split(features, labels_arr):
                            scaler.fit(features[train_idx])
                            clf.fit(scaler.transform(features[train_idx]), labels_arr[train_idx])
                            probs[val_idx] = clf.decision_function(scaler.transform(features[val_idx]))
                        
                        threshold = summary["best"].get("threshold", 0)
                        y_pred = (probs >= threshold).astype(int)
                        
                        log_roc_curve(labels_arr, probs, title=f"{cfg.encoder.upper()} - ROC Curve")
                        log_pr_curve(labels_arr, probs, title=f"{cfg.encoder.upper()} - PR Curve")
                        log_confusion_matrix(labels_arr, y_pred, title=f"{cfg.encoder.upper()} - Confusion Matrix")
                        log_score_distribution(labels_arr, probs, threshold, title=f"{cfg.encoder.upper()} - Score Distribution")
                except Exception as exc:
                    LOGGER.warning(f"详细可视化记录失败: {exc}")
        
        if get_wandb_run() is not None:
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
