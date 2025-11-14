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
from src.utils.optional_imports import WANDB, GENSIM
WANDB_AVAILABLE = WANDB.is_available()
wandb = WANDB.safe_import()

# Gensim for Word2Vec (optional)
KeyedVectors = GENSIM.safe_import().KeyedVectors if GENSIM.is_available() else None

LOGGER = logging.getLogger("embedding_baselines")


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    mal_dir: Optional[str] = None  # 用于 holdout/cv/time_ood 模式
    benign_dir: Optional[str] = None  # 用于 holdout/cv/time_ood 模式
    train_dir: Optional[str] = None  # 用于 dir 模式
    val_dir: Optional[str] = None  # 用于 dir 模式
    test_dir: Optional[str] = None  # 用于 dir 模式
    encoder: str = "bert"  # bert / bge / gemma / word2vec
    model_name: Optional[str] = None  # huggingface 模型名
    word2vec_path: Optional[str] = None
    split_mode: str = "holdout"  # holdout / cv / dir
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
        if self.split_mode not in {"holdout", "cv", "dir", "time_ood"}:
            raise ValueError("split_mode 必须是 holdout/cv/dir/time_ood")
        
        if self.split_mode == "dir":
            # dir 模式需要 train_dir, val_dir, test_dir
            for key, name in [
                (self.train_dir, "train_dir"),
                (self.val_dir, "val_dir"),
                (self.test_dir, "test_dir"),
            ]:
                if not key:
                    raise ValueError(f"dir 模式需要提供 {name}")
                path = Path(key)
                if not path.exists() or not path.is_dir():
                    raise FileNotFoundError(f"{name} 目录不存在：{path}")
        elif self.split_mode == "time_ood":
            # time_ood 模式需要 manifest 路径和目录
            required_fields = [
                ("malicious_manifest_path", "malicious_manifest_path"),
                ("benign_manifest_path", "benign_manifest_path"),
                ("mal_dir", "mal_dir"),
                ("benign_dir", "benign_dir"),
            ]
            for attr_name, field_name in required_fields:
                value = getattr(self, attr_name, None)
                if not value:
                    raise ValueError(f"time_ood 模式需要提供 {field_name}")
                if attr_name.endswith("_path"):
                    path = Path(value)
                    if not path.exists() or not path.is_file():
                        raise FileNotFoundError(f"{field_name} 文件不存在：{path}")
                elif attr_name.endswith("_dir"):
                    path = Path(value)
                    if not path.exists() or not path.is_dir():
                        raise FileNotFoundError(f"{field_name} 目录不存在：{path}")
        else:
            # holdout/cv 模式：必须使用 mal_dir + benign_dir
            if not self.mal_dir or not self.benign_dir:
                raise ValueError("holdout/cv 模式需要提供 mal_dir 和 benign_dir")
            
            # 验证 mal_dir 和 benign_dir
            mal_path = Path(self.mal_dir)
            ben_path = Path(self.benign_dir)
            if not mal_path.exists() or not mal_path.is_dir():
                raise FileNotFoundError(f"mal_dir 目录不存在：{mal_path}")
            if not ben_path.exists() or not ben_path.is_dir():
                raise FileNotFoundError(f"benign_dir 目录不存在：{ben_path}")
            
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
        if self.encoder in {"bert", "bge", "gemma"} and not self.model_name:
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
    parser.add_argument("--mal_dir", type=str, default=defaults.get("mal_dir"), help="Malicious markdown directory")
    parser.add_argument("--benign_dir", type=str, default=defaults.get("benign_dir"), help="Benign markdown directory")
    parser.add_argument("--train_dir", type=str, default=defaults.get("train_dir"))
    parser.add_argument("--val_dir", type=str, default=defaults.get("val_dir"))
    parser.add_argument("--test_dir", type=str, default=defaults.get("test_dir"))
    parser.add_argument("--encoder", choices=["bert", "bge", "gemma", "word2vec"], default=defaults.get("encoder", "bert"))
    parser.add_argument("--model_name", type=str, default=defaults.get("model_name"))
    parser.add_argument("--word2vec_path", type=str, default=defaults.get("word2vec_path"))
    parser.add_argument("--split_mode", choices=["holdout", "cv", "dir", "time_ood"], default=defaults.get("split_mode", "holdout"))
    parser.add_argument("--malicious_manifest_path", type=str, default=defaults.get("malicious_manifest_path"), help="Path to malicious manifest CSV (for time_ood mode)")
    parser.add_argument("--benign_manifest_path", type=str, default=defaults.get("benign_manifest_path"), help="Path to benign manifest CSV (for time_ood mode)")
    parser.add_argument("--train_end_date", type=str, default=defaults.get("train_end_date"), help="Train end date for time_ood mode")
    parser.add_argument("--val_end_date", type=str, default=defaults.get("val_end_date"), help="Val end date for time_ood mode")
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


def load_md_reports_from_dir(data_dir: str, progress: bool = False) -> Tuple[List[str], List[int], List[str]]:
    """Load Markdown reports from a directory with black/white subdirectories.
    
    Args:
        data_dir: Directory containing black/ and white/ subdirectories
        progress: Show progress bar
    
    Returns:
        Tuple of (texts, labels, ids)
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"目录不存在: {data_dir}")
    
    # Try black/white structure
    mal_dir = data_path / "black"
    ben_dir = data_path / "white"
    
    # Try alternative names
    if not mal_dir.exists():
        mal_dir = data_path / "cape_reports_malicious_md"
    if not ben_dir.exists():
        ben_dir = data_path / "cape_reports_benign_md"
    
    mal_files = sorted(mal_dir.glob("*.md")) if mal_dir.exists() else []
    ben_files = sorted(ben_dir.glob("*.md")) if ben_dir.exists() else []
    
    texts = []
    labels = []
    ids = []
    
    all_files = [(f, 1) for f in mal_files] + [(f, 0) for f in ben_files]
    
    iterator = all_files
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"Loading from {data_dir}", total=len(all_files))
    
    for md_file, label in iterator:
        try:
            text = md_file.read_text(encoding="utf-8", errors="ignore")
            texts.append(text)
            labels.append(label)
            ids.append(md_file.stem)
        except Exception as e:
            LOGGER.warning(f"Failed to load {md_file}: {e}")
    
    LOGGER.info(f"从 {data_dir} 加载了 {len(texts)} 个报告 (恶意: {len(mal_files)}, 良性: {len(ben_files)})")
    
    return texts, labels, ids


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
    if cfg.encoder in {"bert", "bge", "gemma"}:
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


def evaluate_holdout_with_predefined_indices(
    features: np.ndarray,
    labels: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    cfg: TrainConfig,
) -> Dict:
    """Evaluate with predefined train/val/test split (dir mode)."""
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)

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
        "_test_idx": test_idx,  # Store for visualization
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
    if WANDB_AVAILABLE and wandb.run:
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
        
        # 结束之前的 run（如果存在），避免并行执行时的 run 名称混乱
        if wandb.run is not None:
            wandb.finish()
        
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=run_name,
            config=asdict(cfg),
            tags=tags,
            reinit=True,  # 允许重新初始化，确保每个基线都有独立的 run
        )

        LOGGER.info(f"W&B 初始化成功，项目: {cfg.wandb_project}, 运行: {run_name}")
        return wandb.run.id

    except Exception as exc:
        LOGGER.warning(f"W&B 初始化失败: {exc}")
        return None


def log_metrics_to_wandb(metrics: Dict, prefix: str = "") -> None:
    """将指标记录到 W&B"""
    if not WANDB_AVAILABLE:
        return
    
    # 检查 W&B run 是否活跃
    if not wandb.run:
        return
    
    try:
        # 检查 run 是否已结束
        if hasattr(wandb.run, 'settings') and wandb.run.settings._run_id is None:
            return
        
        # 添加前缀
        wandb_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                wandb_key = f"{prefix}/{key}" if prefix else key
                wandb_metrics[wandb_key] = value
        
        if wandb_metrics:
            try:
                wandb.log(wandb_metrics)
            except Exception:
                # W&B run 可能已结束，静默忽略
                pass
    except Exception as exc:
        # 如果 W&B run 已结束，静默忽略
        if "finished" in str(exc).lower() or "active run" in str(exc).lower():
            pass
        else:
            LOGGER.warning(f"W&B 记录失败: {exc}")


def run_experiment(cfg: TrainConfig) -> Dict:
    # 初始化 W&B
    run_id = init_wandb(cfg)

    # Load data
    if cfg.split_mode == "dir":
        # dir 模式：从预定义的 train/val/test 目录加载数据
        train_texts, train_labels, train_ids = load_md_reports_from_dir(
            cfg.train_dir, progress=False
        )
        val_texts, val_labels, val_ids = load_md_reports_from_dir(
            cfg.val_dir, progress=False
        )
        test_texts, test_labels, test_ids = load_md_reports_from_dir(
            cfg.test_dir, progress=False
        )
        
        # 合并所有数据
        texts = train_texts + val_texts + test_texts
        labels = train_labels + val_labels + test_labels
        ids = train_ids + val_ids + test_ids
        
        # 保存划分信息用于后续评估
        n_train = len(train_texts)
        n_val = len(val_texts)
        n_test = len(test_texts)
    elif cfg.split_mode == "time_ood":
        # time_ood 模式：基于时间切分数据
        from src.utils.time_split import load_manifest_with_time, split_by_time
        
        LOGGER.info("Loading data with time-based split...")
        mal_df, ben_df = load_manifest_with_time(
            cfg.malicious_manifest_path,
            cfg.benign_manifest_path,
            mal_dir=cfg.mal_dir,
            benign_dir=cfg.benign_dir,
        )
        
        train_end_date = getattr(cfg, 'train_end_date', '2025-04-23 08:08:46')
        val_end_date = getattr(cfg, 'val_end_date', '2025-06-14 11:39:39')
        
        (
            train_paths, train_labels_list, train_ids_list,
            val_paths, val_labels_list, val_ids_list,
            test_paths, test_labels_list, test_ids_list,
        ) = split_by_time(
            mal_df,
            ben_df,
            train_end_date,
            val_end_date,
            mal_dir=cfg.mal_dir,
            benign_dir=cfg.benign_dir,
            benign_split_strategy="random",
            random_seed=getattr(cfg, 'seed', 42),
        )
        
        # 加载文本内容
        def load_texts_from_paths(paths):
            texts = []
            for path in paths:
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        texts.append(f.read())
                except Exception as e:
                    LOGGER.warning(f"Failed to load {path}: {e}")
                    texts.append("")
            return texts
        
        train_texts = load_texts_from_paths(train_paths)
        val_texts = load_texts_from_paths(val_paths)
        test_texts = load_texts_from_paths(test_paths)
        
        # 合并所有数据
        texts = train_texts + val_texts + test_texts
        labels = train_labels_list + val_labels_list + test_labels_list
        ids = train_ids_list + val_ids_list + test_ids_list
        
        # 保存划分信息用于后续评估
        n_train = len(train_texts)
        n_val = len(val_texts)
        n_test = len(test_texts)
    else:
        # 原有的 holdout/cv 模式：使用 mal_dir + benign_dir 加载数据
        mal_dir_path = Path(cfg.mal_dir)
        ben_dir_path = Path(cfg.benign_dir)
        
        mal_files = sorted(mal_dir_path.glob("*.md"))
        ben_files = sorted(ben_dir_path.glob("*.md"))
        
        texts = []
        labels = []
        ids = []
        
        all_files = [(f, 1) for f in mal_files] + [(f, 0) for f in ben_files]
        iterator = all_files
        if cfg.progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Loading reports", total=len(all_files))
        
        for md_file, label in iterator:
            try:
                text = md_file.read_text(encoding="utf-8", errors="ignore")
                texts.append(text)
                labels.append(label)
                ids.append(md_file.stem)
            except Exception as e:
                LOGGER.warning(f"Failed to load {md_file}: {e}")
        
        LOGGER.info(f"加载了 {len(texts)} 个报告 (恶意: {len(mal_files)}, 良性: {len(ben_files)})")
        
        # 如果设置了 limit，进行分层抽样
        if cfg.limit and len(texts) > cfg.limit:
            import numpy as np
            from sklearn.model_selection import StratifiedShuffleSplit
            LOGGER.info("限制样本数至 %d，进行分层抽样。", cfg.limit)
            y_arr = np.array(labels, dtype=int)
            idx_all = np.arange(len(labels))
            sss = StratifiedShuffleSplit(n_splits=1, train_size=cfg.limit, random_state=42)
            keep_idx, _ = next(sss.split(idx_all, y_arr))
            texts = [texts[i] for i in keep_idx]
            labels = [labels[i] for i in keep_idx]
            ids = [ids[i] for i in keep_idx]
        
        n_train = n_val = n_test = None
    
    if not texts:
        raise RuntimeError("未读取到任何样本，请检查 Markdown 目录。")

    encoder = build_encoder(cfg)
    labels_arr = np.array(labels, dtype=int)

    # 分层模式：类似 LEC 的逐层评估
    if cfg.layerwise and cfg.encoder in {"bert", "bge", "gemma"}:
        LOGGER.info("使用分层特征提取模式，开始编码 %d 篇文档...", len(texts))
        features_layers = encoder.encode_layers(texts, until_layer=cfg.until_layer)
        LOGGER.info("特征形状: %s (samples, layers, hidden_size)", features_layers.shape)
        
        if cfg.split_mode == "dir":
            # TODO: 实现 dir 模式的分层评估
            raise NotImplementedError("dir 模式的分层评估尚未实现")
        elif cfg.split_mode == "holdout":
            # TODO: 实现 holdout 分层评估
            raise NotImplementedError("holdout 模式的分层评估尚未实现")
        else:
            summary = evaluate_cv_layers(features_layers, labels_arr, cfg, progress=True)
    else:
        # 单层模式：使用最后一层
        LOGGER.info("开始编码 %d 篇文档...", len(texts))
        features = encoder.encode(texts)
        
        if cfg.split_mode in {"dir", "time_ood"}:
            summary = evaluate_holdout_with_predefined_indices(
                features, labels_arr, n_train, n_val, n_test, cfg
            )
        elif cfg.split_mode == "holdout":
            summary = evaluate_holdout(features, labels_arr, cfg)
        else:
            summary = evaluate_cv(features, labels_arr, cfg)

    # Update config_digest for dir mode
    if cfg.split_mode == "dir":
        summary["config_digest"] = {
            "train_dir": str(Path(cfg.train_dir).resolve()),
            "val_dir": str(Path(cfg.val_dir).resolve()),
            "test_dir": str(Path(cfg.test_dir).resolve()),
            "encoder": cfg.encoder,
            "model_name": cfg.model_name,
        }
    else:
        summary["config_digest"] = {
            "md_dir": str(Path(cfg.md_dir).resolve()),
            "encoder": cfg.encoder,
            "model_name": cfg.model_name,
        }
    
    # 记录指标到 W&B
    if "best" in summary:
        if cfg.layerwise:
            # 分层模式：记录最佳层信息
            if WANDB_AVAILABLE and wandb.run:
                wandb.run.summary.update({
                    "best_layer": summary["best"].get("layer_index"),
                    "best_macro_f1": summary["best"].get("macro_f1", 0),
                    "best_auroc": summary["best"].get("auroc", 0),
                    "best_aupr": summary["best"].get("aupr", 0),
                    "best_threshold": summary["best"].get("threshold", 0),
                })
        else:
            log_metrics_to_wandb(summary["best"], prefix=cfg.split_mode)
            if WANDB_AVAILABLE and wandb.run:
                wandb.run.summary.update({
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
    
    # 确保 W&B run 被正确结束（无论是否成功）
    if WANDB_AVAILABLE and wandb.run is not None:
        try:
            wandb.finish()
            LOGGER.info("实验结果已记录到 W&B")
        except Exception as exc:
            LOGGER.warning(f"W&B finish 失败: {exc}")

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

