#!/usr/bin/env python3
"""
逐层 LEC 训练脚本（不集成 Venn-Abers）
主要改进：
  - 统一的配置加载与日志输出，支持 --config JSON 文件；
  - LayerwiseFeatureExtractor 仅实例化一次，并正确处理 CUDA Tensor -> NumPy；
  - 训练/评估流程模块化，避免 cv/holdout/dir 三处重复代码；
  - 输出目录、安全检查与结果 JSON 管理更加严谨。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import numpy as np
import statistics
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    from .feature_extractor import LayerwiseFeatureExtractor
except ImportError:
    try:
        from src.lec.feature_extractor import LayerwiseFeatureExtractor
    except ImportError:
        import sys
        lec_dir = Path(__file__).parent
        if str(lec_dir) not in sys.path:
            sys.path.insert(0, str(lec_dir))
        from feature_extractor import LayerwiseFeatureExtractor

# W&B integration (optional)
from src.utils.optional_imports import WANDB, MATPLOTLIB
WANDB_AVAILABLE = WANDB.is_available()
wandb = WANDB.safe_import()


LOGGER = logging.getLogger("train_lec")

LOGREG_GAMMAS = (0.01, 0.1, 1.0)


# --------------------------------------------------------------------------- #
# 数据与配置处理
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    model_dir: str
    split_mode: str = "cv"  # cv / holdout / dir / time_ood / split_spec
    mal_dir: Optional[str] = None  # 用于 cv/holdout/time_ood 模式
    benign_dir: Optional[str] = None  # 用于 cv/holdout/time_ood 模式
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
    max_tokens: int = 1024
    stride: int = 256
    clf: str = "logreg"  # logreg / ridge
    n_splits: int = 10
    until_layer: Optional[int] = None  # 前向至第几层（1-based）
    out: Optional[str] = None
    gpu: int = 0
    progress: bool = False
    save_raw: bool = False
    run_tag: str = ""
    cache_dir: Optional[str] = None  # 若指定，将特征缓存至该目录
    target_fpr: float = 0.01
    lambda_penalty: float = 0.1
    bootstrap_samples: int = 0
    trust_remote_code: bool = False
    cache_up_to_layer: Optional[int] = None
    layer_pooling: str = "single"  # single / mean
    candidate_layers: Optional[List[int]] = None
    selection_strategy: str = "drift"  # drift / min_tpr / tpr_id / macro_f1

    # W&B 配置
    use_wandb: bool = False  # 是否启用 W&B 记录
    wandb_project: str = "lec-experiments"  # W&B 项目名
    wandb_entity: Optional[str] = None  # W&B 实体名（可选）
    wandb_api_key: Optional[str] = None  # W&B API 密钥

    # 仅 CLI 使用，不进入结果 JSON
    config_path: Optional[str] = field(default=None, repr=False, compare=False)

    def validate(self) -> None:
        if self.split_mode not in {"cv", "holdout", "dir", "time_ood", "split_spec"}:
            raise ValueError(f"split_mode 必须是 cv/holdout/dir/time_ood/split_spec 之一，当前为 {self.split_mode}")
        if self.split_mode == "dir":
            missing = [k for k in ("train_dir", "val_dir", "test_dir") if not getattr(self, k)]
            if missing:
                raise ValueError(f"dir 模式需要提供 {missing}")
        elif self.split_mode == "split_spec":
            missing = [k for k in ("split_spec_path", "split_name") if not getattr(self, k)]
            if missing:
                raise ValueError(f"split_spec 模式需要提供 {missing}")
            split_path = Path(str(self.split_spec_path))
            if not split_path.exists() or not split_path.is_file():
                raise FileNotFoundError(f"split_spec_path 文件不存在：{split_path}")
        elif self.split_mode == "time_ood":
            required_fields = [
                ("malicious_manifest_path", "malicious_manifest_path"),
                ("benign_manifest_path", "benign_manifest_path"),
                ("mal_dir", "mal_dir"),
                ("benign_dir", "benign_dir"),
            ]
            missing = []
            for attr_name, field_name in required_fields:
                value = getattr(self, attr_name, None)
                if not value:
                    missing.append(field_name)
                else:
                    if attr_name.endswith("_path"):
                        path = Path(value)
                        if not path.exists() or not path.is_file():
                            raise FileNotFoundError(f"{field_name} 文件不存在：{path}")
                    elif attr_name.endswith("_dir"):
                        path = Path(value)
                        if not path.exists() or not path.is_dir():
                            raise FileNotFoundError(f"{field_name} 目录不存在：{path}")
            if missing:
                raise ValueError(f"time_ood 模式需要提供 {missing}")
            if self.benign_split_strategy not in {"random", "time"}:
                raise ValueError("benign_split_strategy 必须是 random 或 time")
        elif not self.mal_dir or not self.benign_dir:
            raise ValueError("cv/holdout 模式需要指定 --mal_dir 和 --benign_dir")
        if self.limit is not None and self.limit < 2:
            raise ValueError("limit 至少为 2 才有意义")
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        if self.cache_up_to_layer is not None:
            if self.cache_up_to_layer < 1:
                raise ValueError("cache_up_to_layer 必须 >= 1")
            if self.until_layer is not None and self.cache_up_to_layer < self.until_layer:
                raise ValueError("cache_up_to_layer 不能小于 until_layer")
        if not (0.0 < self.target_fpr < 1.0):
            raise ValueError("target_fpr 必须在 (0, 1) 之间")
        if self.lambda_penalty < 0:
            raise ValueError("lambda_penalty 不能为负")
        if self.bootstrap_samples < 0:
            raise ValueError("bootstrap_samples 不能为负")
        if self.layer_pooling not in {"single", "mean"}:
            raise ValueError("layer_pooling 必须是 single 或 mean")
        if self.selection_strategy not in {"drift", "min_tpr", "tpr_id", "macro_f1"}:
            raise ValueError("selection_strategy 必须是 drift/min_tpr/tpr_id/macro_f1 之一")
        if self.candidate_layers is not None:
            if not self.candidate_layers:
                raise ValueError("candidate_layers 不能为空列表")
            if not all(isinstance(layer, int) for layer in self.candidate_layers):
                raise ValueError("candidate_layers 必须是整数列表")
        if self.seeds:
            if not all(isinstance(seed, int) for seed in self.seeds):
                raise ValueError("seeds 必须是整数列表")
            unique_seeds = sorted(dict.fromkeys(self.seeds))
            if not unique_seeds:
                raise ValueError("seeds 列表不能为空")
            self.seeds = unique_seeds


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #


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
    parser.add_argument("--model_dir", type=str, default=defaults.get("model_dir"))
    parser.add_argument("--mal_dir", type=str, default=defaults.get("mal_dir"), help="Malicious markdown directory (for cv/holdout/time_ood mode)")
    parser.add_argument("--benign_dir", type=str, default=defaults.get("benign_dir"), help="Benign markdown directory (for cv/holdout/time_ood mode)")
    parser.add_argument("--split_mode", choices=["cv", "holdout", "dir", "time_ood", "split_spec"], default=defaults.get("split_mode", "cv"))
    parser.add_argument("--split_spec_path", type=str, default=defaults.get("split_spec_path"))
    parser.add_argument("--split_name", type=str, default=defaults.get("split_name"))
    parser.add_argument("--malicious_manifest_path", type=str, default=defaults.get("malicious_manifest_path"), help="Path to malicious manifest CSV (for time_ood mode)")
    parser.add_argument("--benign_manifest_path", type=str, default=defaults.get("benign_manifest_path"), help="Path to benign manifest CSV (for time_ood mode)")
    parser.add_argument("--train_end_date", type=str, default=defaults.get("train_end_date"), help="Train end date for time_ood mode")
    parser.add_argument("--val_end_date", type=str, default=defaults.get("val_end_date"), help="Val end date for time_ood mode")
    parser.add_argument("--benign_split_strategy", choices=["random", "time"], default=defaults.get("benign_split_strategy", "random"),
                        help="Benign split strategy for time_ood mode")
    parser.add_argument("--val_ratio", type=float, default=defaults.get("val_ratio", 0.1))
    parser.add_argument("--test_ratio", type=float, default=defaults.get("test_ratio", 0.2))
    parser.add_argument("--train_dir", type=str, default=defaults.get("train_dir"))
    parser.add_argument("--val_dir", type=str, default=defaults.get("val_dir"))
    parser.add_argument("--test_dir", type=str, default=defaults.get("test_dir"))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--seeds", type=str, default=None, help="逗号分隔的多随机种子，例如 40,41,42")
    parser.add_argument("--limit", type=int, default=defaults.get("limit"))
    parser.add_argument("--max_tokens", type=int, default=defaults.get("max_tokens", 1024))
    parser.add_argument("--stride", type=int, default=defaults.get("stride", 256))
    parser.add_argument("--clf", choices=["logreg", "ridge"], default=defaults.get("clf", "logreg"))
    parser.add_argument("--n_splits", type=int, default=defaults.get("n_splits", 10))
    parser.add_argument("--until_layer", type=int, default=defaults.get("until_layer"))
    parser.add_argument("--out", type=str, default=defaults.get("out"))
    parser.add_argument("--gpu", type=int, default=defaults.get("gpu", 0))
    parser.add_argument("--progress", action="store_true", default=defaults.get("progress", False))
    parser.add_argument("--save_raw", action="store_true", default=defaults.get("save_raw", False))
    parser.add_argument("--run_tag", type=str, default=defaults.get("run_tag", ""))
    parser.add_argument("--cache_dir", type=str, default=defaults.get("cache_dir"))
    parser.add_argument("--cache_up_to_layer", type=int, default=defaults.get("cache_up_to_layer"), help="缓存特征时计算到的最大层数（>= until_layer）")
    parser.add_argument("--target_fpr", type=float, default=defaults.get("target_fpr", 0.01))
    parser.add_argument("--lambda_penalty", type=float, default=defaults.get("lambda_penalty", 0.1))
    parser.add_argument("--bootstrap_samples", type=int, default=defaults.get("bootstrap_samples", 0))
    parser.add_argument("--layer_pooling", choices=["single", "mean"], default=defaults.get("layer_pooling", "single"))
    parser.add_argument("--candidate_layers", type=str, default=None, help="候选层列表，例如 11,12,13,14,15 或 -1")
    parser.add_argument("--selection_strategy", choices=["drift", "min_tpr", "tpr_id", "macro_f1"], default=defaults.get("selection_strategy", "drift"))
    parser.add_argument("--trust_remote_code", action="store_true", default=defaults.get("trust_remote_code", False),
                        help="允许加载包含自定义代码的模型（transformers trust_remote_code）")
    parser.add_argument("--use_wandb", action="store_true", default=defaults.get("use_wandb", False))
    parser.add_argument("--wandb_project", type=str, default=defaults.get("wandb_project", "lec-experiments"))
    parser.add_argument("--wandb_entity", type=str, default=defaults.get("wandb_entity"))
    parser.add_argument("--wandb_api_key", type=str, default=defaults.get("wandb_api_key"))
    return parser


def parse_config() -> TrainConfig:
    # 先解析 --config
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, default=None)
    base_args, remaining = base_parser.parse_known_args()
    cfg_dict = load_config_file(base_args.config)
    parser = build_parser(cfg_dict)
    args = parser.parse_args(remaining)
    merged = {**cfg_dict, **{k: v for k, v in vars(args).items() if v is not None}}
    merged.pop("th_step", None)
    merged["config_path"] = base_args.config
    seeds_value = merged.pop("seeds", None)
    if seeds_value:
        if isinstance(seeds_value, str):
            parsed = [s.strip() for s in seeds_value.split(",")]
            seed_list = [int(s) for s in parsed if s]
        elif isinstance(seeds_value, list):
            seed_list = [int(s) for s in seeds_value if isinstance(s, (int, str))]
        else:
            raise ValueError("--seeds 参数解析失败")
        if not seed_list:
            raise ValueError("--seeds 至少提供一个整数种子")
        merged["seeds"] = seed_list
        merged.setdefault("seed", seed_list[0])
    candidate_layers_value = merged.pop("candidate_layers", None)
    if candidate_layers_value is not None:
        if isinstance(candidate_layers_value, str):
            merged["candidate_layers"] = [
                int(part.strip())
                for part in candidate_layers_value.split(",")
                if part.strip()
            ]
        elif isinstance(candidate_layers_value, list):
            merged["candidate_layers"] = [int(value) for value in candidate_layers_value]
        else:
            raise ValueError("--candidate_layers 参数解析失败")
    config = TrainConfig(**merged)
    config.validate()
    return config


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


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
        run_name = f"lec-{cfg.split_mode}"
        if cfg.run_tag:
            run_name += f"-{cfg.run_tag}"
        if cfg.until_layer:
            run_name += f"-L{cfg.until_layer}"

        # 初始化 W&B
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=run_name,
            config=asdict(cfg),
            tags=[cfg.split_mode, cfg.clf, f"model_{os.path.basename(cfg.model_dir)}"],
            reinit=True,
        )

        LOGGER.info(f"W&B 初始化成功，项目: {cfg.wandb_project}, 运行: {run_name}")
        return wandb.run.id

    except Exception as exc:
        LOGGER.warning(f"W&B 初始化失败: {exc}")
        return None


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #


def load_md_reports(md_dir: str, limit: Optional[int] = None) -> Tuple[List[str], List[int], List[str]]:
    """
    读取 Markdown 文本与标签，同时返回对应文件路径，便于缓存与追踪。
    标签优先顺序：
      1) 同名 .label；
      2) 父目录包含 'black/white/malicious/benign'；
      3) 'unknown' 目录跳过。
    """
    md_paths: List[str] = []
    for root, _, fnames in os.walk(md_dir, followlinks=True):
        for fn in fnames:
            if fn.endswith(".md"):
                md_paths.append(os.path.join(root, fn))
    md_paths.sort()

    texts, labels, kept_paths = [], [], []
    skipped_unknown = 0
    for path in md_paths:
        label_path = os.path.splitext(path)[0] + ".label"
        label: Optional[int] = None
        if os.path.exists(label_path):
            try:
                with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
                    label = int(f.read().strip())
            except Exception:
                LOGGER.warning("读取标签失败，改用目录推断：%s", label_path)
                label = None
        if label is None:
            parts_lower = [p.lower() for p in os.path.normpath(path).split(os.sep)]
            if any("black" in p or "malicious" in p for p in parts_lower):
                label = 1
            elif any("white" in p or "benign" in p for p in parts_lower):
                label = 0
            elif "unknown" in parts_lower:
                skipped_unknown += 1
                continue
            else:
                LOGGER.debug("无法确定标签，跳过：%s", path)
                continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
        labels.append(label)
        kept_paths.append(path)

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


def extract_metadata_from_paths(paths: List[str], texts: List[str]) -> Dict[str, List]:
    """
    从文件路径中提取元数据（家族、时间等）
    返回字典，包含 families, timestamps, lengths, ioc_coverage 等
    """
    families = []
    timestamps = []
    lengths = [len(text) for text in texts]
    
    for path in paths:
        # 尝试从路径中提取家族信息
        parts = path.split(os.sep)
        family = "Unknown"
        for part in parts:
            if any(keyword in part.lower() for keyword in ['family', 'malware', 'trojan', 'virus', 'worm']):
                family = part
                break
        families.append(family)
        
        # 尝试从文件名或路径中提取时间戳
        # 假设文件名可能包含日期格式 YYYY-MM-DD 或类似格式
        import re
        date_match = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2})', path)
        if date_match:
            timestamps.append(date_match.group(1))
        else:
            # 使用文件修改时间作为备选
            try:
                import time
                mtime = os.path.getmtime(path)
                timestamps.append(time.strftime('%Y-%m-%d', time.localtime(mtime)))
            except:
                timestamps.append(None)
    
    # 计算 IOC 覆盖率（简单启发式：查找 IP、URL、hash 等模式）
    ioc_coverage = []
    for text in texts:
        ioc_count = 0
        # 简单的 IOC 模式匹配
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        url_pattern = r'https?://[^\s]+'
        hash_pattern = r'\b[a-fA-F0-9]{32,64}\b'
        ioc_count += len(re.findall(ip_pattern, text))
        ioc_count += len(re.findall(url_pattern, text))
        ioc_count += len(re.findall(hash_pattern, text))
        # 归一化到 [0, 1]
        coverage = min(ioc_count / max(len(text) / 100, 1), 1.0)
        ioc_coverage.append(coverage)
    
    return {
        'families': families,
        'timestamps': timestamps,
        'lengths': lengths,
        'ioc_coverage': ioc_coverage,
    }


# --------------------------------------------------------------------------- #
# 特征提取与缓存
# --------------------------------------------------------------------------- #


def build_extractor(
    model_dir: str,
    gpu: int,
    dtype: Optional[str] = "float16",
    trust_remote_code: bool = False,
) -> LayerwiseFeatureExtractor:
    if torch.cuda.is_available():
        device = f"cuda:{gpu}"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    LOGGER.info("使用设备 %s 进行特征提取。", device)
    return LayerwiseFeatureExtractor(
        model_dir=model_dir,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )


def texts_to_features(
    texts: Sequence[str],
    extractor: LayerwiseFeatureExtractor,
    max_tokens: int,
    stride: int,
    until_layer: Optional[int],
    progress: bool = False,
) -> np.ndarray:
    iterator: Iterable[int] = range(len(texts))
    if progress:
        iterator = tqdm(iterator, total=len(texts), desc="Encoding docs")
    feats: List[np.ndarray] = []
    for idx in iterator:
        tensor = extractor.encode_document_layers(
            text=texts[idx],
            max_tokens=max_tokens,
            stride=stride,
            until_layer=until_layer,
        )
        if tensor.is_cuda:
            tensor = tensor.detach().cpu()
        feat_np = tensor.numpy()
        # 清理 inf 和 nan 值，转换为 float32
        feat_np = np.nan_to_num(feat_np, nan=0.0, posinf=1e6, neginf=-1e6)
        feat_np = feat_np.astype(np.float32)  # 转换为 float32 避免 float16 溢出
        feats.append(feat_np)
        # 每个文档处理完后立即清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    features = np.stack(feats, axis=0)
    # 最终清理：确保所有特征都是有效的 float32
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    features = features.astype(np.float32)
    return features


def maybe_cache_features(
    cache_dir: Optional[str],
    key: str,
    compute_fn,
    required_layers: Optional[int] = None,
) -> np.ndarray:
    if not cache_dir:
        return compute_fn()
    cache_path = os.path.join(cache_dir, f"{key}.npy")
    if os.path.exists(cache_path):
        LOGGER.info("命中特征缓存：%s", cache_path)
        cached = np.load(cache_path)
        if required_layers is not None:
            if cached.shape[1] < required_layers:
                LOGGER.info("缓存层数不足（%d < %d），重新计算。", cached.shape[1], required_layers)
            else:
                return cached[:, :required_layers, :]
        else:
            return cached
    features = compute_fn()
    np.save(cache_path, features)
    LOGGER.info("特征已缓存到：%s", cache_path)
    if required_layers is not None:
        return features[:, :required_layers, :]
    return features


def normalise_layer_indices(
    candidate_layers: Optional[Sequence[int]],
    total_layers: int,
) -> List[int]:
    if total_layers < 1:
        raise ValueError("total_layers 必须 >= 1")
    if not candidate_layers:
        return list(range(total_layers))

    resolved: List[int] = []
    for raw in candidate_layers:
        if raw == 0:
            raise ValueError("candidate_layers 使用 1-based 索引，不能包含 0")
        idx = total_layers + int(raw) if raw < 0 else int(raw) - 1
        if idx < 0 or idx >= total_layers:
            raise ValueError(f"候选层 {raw} 超出范围（模型共有 {total_layers} 层）")
        if idx not in resolved:
            resolved.append(idx)
    return resolved


def describe_layer_group(layer_indices: Sequence[int], pooling: str) -> str:
    layers_1based = [idx + 1 for idx in layer_indices]
    if len(layers_1based) == 1:
        return f"L{layers_1based[0]}"
    if all(curr - prev == 1 for prev, curr in zip(layers_1based, layers_1based[1:])):
        span = f"L{layers_1based[0]}-L{layers_1based[-1]}"
    else:
        span = ",".join(f"L{layer}" for layer in layers_1based)
    return span if pooling == "single" else f"mean({span})"


def build_feature_views(
    X_layers: np.ndarray,
    *,
    layer_pooling: str,
    candidate_layers: Optional[Sequence[int]],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    total_layers = int(X_layers.shape[1])
    resolved = normalise_layer_indices(candidate_layers, total_layers)

    if layer_pooling == "single":
        specs = [
            {
                "view_index": int(pos + 1),
                "layer_index": int(layer_idx + 1),
                "layer_indices": [int(layer_idx + 1)],
                "view_label": describe_layer_group([layer_idx], pooling="single"),
            }
            for pos, layer_idx in enumerate(resolved)
        ]
        return X_layers[:, resolved, :], specs

    pooled = X_layers[:, resolved, :].mean(axis=1, keepdims=True)
    spec = {
        "view_index": 1,
        "layer_indices": [int(layer_idx + 1) for layer_idx in resolved],
        "view_label": describe_layer_group(resolved, pooling="mean"),
    }
    return pooled, [spec]


def compute_selection_score(metrics: Dict[str, Any], strategy: str) -> float:
    if strategy == "drift":
        return float(metrics.get("drift_score", metrics.get("tpr_id", float("-inf"))))
    if strategy == "min_tpr":
        if "tpr_ood" in metrics:
            return float(min(metrics.get("tpr_id", float("-inf")), metrics.get("tpr_ood", float("-inf"))))
        return float(metrics.get("tpr_id", float("-inf")))
    if strategy == "tpr_id":
        return float(metrics.get("tpr_id", float("-inf")))
    if strategy == "macro_f1":
        return float(metrics.get("macro_f1", float("-inf")))
    raise ValueError(f"未知 selection_strategy: {strategy}")


def results_have_numeric_layers(results: Sequence[Dict[str, Any]]) -> bool:
    if not results:
        return False
    return all(isinstance(entry.get("layer_index"), int) for entry in results)


# --------------------------------------------------------------------------- #
# 模型与评估工具
# --------------------------------------------------------------------------- #


def build_classifier(name: str, gamma: Optional[float] = None) -> Pipeline:
    if name == "logreg":
        if gamma is not None and gamma <= 0:
            raise ValueError("gamma must be positive when provided.")
        reg_strength = gamma if gamma is not None else 1.0
        clf = LogisticRegression(
            C=1.0 / reg_strength,
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
    else:
        clf = RidgeClassifier(class_weight="balanced")
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", clf),
    ])


def select_logreg_gamma(
    X: np.ndarray,
    y: np.ndarray,
    gammas: Sequence[float] = LOGREG_GAMMAS,
    cv_splits: int = 5,
) -> float:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return float(gammas[-1])
    splits = min(cv_splits, int(counts.min()))
    if splits < 2:
        return float(gammas[-1])

    best_gamma = float(gammas[-1])
    best_score = -np.inf
    for gamma in gammas:
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
        scores: List[float] = []
        for train_idx, val_idx in skf.split(X, y):
            pipeline = build_classifier("logreg", gamma=gamma)
            pipeline.fit(X[train_idx], y[train_idx])
            preds = pipeline.predict(X[val_idx])
            score = f1_score(y[val_idx], preds, average="macro")
            scores.append(score)
        mean_score = float(np.mean(scores)) if scores else -np.inf
        if mean_score > best_score or (np.isclose(mean_score, best_score) and gamma < best_gamma):
            best_score = mean_score
            best_gamma = float(gamma)
    return best_gamma



def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.ndim == 1:
        scores = scores[:, None]
    s = scores.squeeze()
    s_min, s_max = s.min(), s.max()
    if s_max - s_min < 1e-8:
        return np.full_like(s, fill_value=0.5, dtype=float)
    return (s - s_min) / (s_max - s_min)


def predict_proba(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    model = pipeline.named_steps["clf"]
    if hasattr(model, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    scores = pipeline.decision_function(X)
    return normalize_scores(scores)


def quantile_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr: float = 0.01,
    adjust_threshold: bool = True,
) -> Dict:
    """
    计算分位数阈值，确保 FPR ≤ target_fpr
    
    Args:
        y_true: 真实标签
        y_prob: 预测概率/分数
        target_fpr: 目标 FPR
        adjust_threshold: 如果 FPR > target_fpr，是否微调阈值（往右调约 0.001）
    
    Returns:
        包含阈值、FPR、TPR 等指标的字典
    """
    mask_neg = y_true == 0
    mask_pos = y_true == 1
    if mask_neg.sum() == 0 or mask_pos.sum() == 0:
        raise ValueError("需要包含正负样本才能计算 FPR/TPR。")

    neg_scores = y_prob[mask_neg]
    tau = float(np.quantile(neg_scores, 1 - target_fpr))
    
    # 如果启用微调且 FPR 超过目标值，稍微提高阈值
    if adjust_threshold:
        y_pred = (y_prob >= tau).astype(int)
        fpr = float(((y_pred == 1) & mask_neg).sum() / mask_neg.sum())
        
        # 如果 FPR > target_fpr，微调阈值（往右调）
        if fpr > target_fpr:
            # 尝试微调：增加阈值约 0.001 * std，或使用更严格的分位数
            std_neg = float(neg_scores.std(ddof=1) + 1e-8)
            adjustment = min(0.001 * std_neg, std_neg * 0.01)  # 最多调整 1% std
            
            # 尝试多个微调值，找到满足 FPR ≤ target_fpr 的最小阈值
            for adj_factor in [1.0, 1.5, 2.0, 3.0, 5.0]:
                tau_adjusted = tau + adjustment * adj_factor
                y_pred_adj = (y_prob >= tau_adjusted).astype(int)
                fpr_adj = float(((y_pred_adj == 1) & mask_neg).sum() / mask_neg.sum())
                if fpr_adj <= target_fpr:
                    tau = tau_adjusted
                    break
            else:
                # 如果微调失败，使用更严格的分位数
                tau = float(np.quantile(neg_scores, 1 - target_fpr * 0.9))
    
    y_pred = (y_prob >= tau).astype(int)
    fpr = float(((y_pred == 1) & mask_neg).sum() / mask_neg.sum())
    tpr = float(((y_pred == 1) & mask_pos).sum() / mask_pos.sum())

    mu_neg, std_neg = float(neg_scores.mean()), float(neg_scores.std(ddof=1) + 1e-8)
    mu_all, std_all = float(y_prob.mean()), float(y_prob.std(ddof=1) + 1e-8)
    z_neg = (tau - mu_neg) / std_neg
    z_all = (tau - mu_all) / std_all

    return {"tau": tau, "fpr": fpr, "tpr": tpr, "z_neg": float(z_neg), "z_all": float(z_all)}


def _evaluate_fixed_threshold(y_true: np.ndarray, y_prob: np.ndarray, tau: float) -> Dict:
    mask_neg = y_true == 0
    mask_pos = y_true == 1
    if mask_neg.sum() == 0 or mask_pos.sum() == 0:
        raise ValueError("需要包含正负样本才能计算 FPR/TPR。")
    preds = (y_prob >= tau).astype(int)
    fpr = float(((preds == 1) & mask_neg).sum() / mask_neg.sum())
    tpr = float(((preds == 1) & mask_pos).sum() / mask_pos.sum())
    mu, std = float(y_prob.mean()), float(y_prob.std(ddof=1) + 1e-8)
    return {"tpr": tpr, "fpr": fpr, "z_all": (tau - mu) / std}


def _histogram_probabilities(values: np.ndarray, bins: int = 50) -> Optional[np.ndarray]:
    if values.size == 0:
        return None
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0), density=False)
    total = hist.sum()
    if total == 0:
        return None
    probs = hist.astype(np.float64) / float(total)
    probs = np.clip(probs, 1e-12, None)
    probs /= probs.sum()
    return probs


def compute_kl_divergence(
    baseline_scores: np.ndarray,
    shifted_scores: np.ndarray,
    bins: int = 50,
) -> Optional[float]:
    """估算两段分数分布之间的 KL 散度（以基准分布为 P，漂移分布为 Q）。"""
    p = _histogram_probabilities(baseline_scores, bins=bins)
    q = _histogram_probabilities(shifted_scores, bins=bins)
    if p is None or q is None:
        return None
    # KL(P || Q)
    div = float(np.sum(p * np.log(p / q)))
    if np.isnan(div) or np.isinf(div):
        return None
    return div


def compute_layer_kl_metrics(
    val_scores: np.ndarray,
    y_val: np.ndarray,
    test_scores: np.ndarray,
    y_test: np.ndarray,
    bins: int = 50,
) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "kl_all": compute_kl_divergence(val_scores, test_scores, bins=bins)
    }
    for label, key in [(0, "kl_neg"), (1, "kl_pos")]:
        val_mask = y_val == label
        test_mask = y_test == label
        metrics[key] = compute_kl_divergence(val_scores[val_mask], test_scores[test_mask], bins=bins)
    return metrics


def annotate_kl_alerts(
    layer_metrics: List[Dict],
    key: str = "kl_neg",
    sigma_factor: float = 1.5,
) -> Optional[Dict]:
    values = [m.get(key) for m in layer_metrics if isinstance(m.get(key), (int, float))]
    finite_values = [float(v) for v in values if np.isfinite(v)]
    if not finite_values:
        return None
    mean = float(np.mean(finite_values))
    std = float(np.std(finite_values, ddof=1)) if len(finite_values) > 1 else 0.0
    threshold = mean + sigma_factor * std
    history = []
    for entry in layer_metrics:
        val = entry.get(key)
        alert = bool(val is not None and np.isfinite(val) and val > threshold)
        entry["kl_alert"] = alert
        entry["kl_threshold"] = threshold
        history.append({
            "layer_index": entry.get("layer_index"),
            "value": val,
            "delta_tau": entry.get("delta_tau"),
            "alert": alert,
        })
    return {
        "metric": key,
        "mean": mean,
        "std": std,
        "sigma_factor": sigma_factor,
        "threshold": threshold,
        "history": history,
    }


def bootstrap_interval(
    scores: np.ndarray,
    tau: float,
    n_samples: int,
    seed: int,
) -> Tuple[float, float]:
    if n_samples <= 0 or scores.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_samples):
        sample = rng.choice(scores, size=scores.size, replace=True)
        draws.append(float((sample >= tau).mean()))
    lower = float(np.quantile(draws, 0.025))
    upper = float(np.quantile(draws, 0.975))
    return lower, upper


def log_metrics_to_wandb(metrics: Dict, prefix: str = "") -> None:
    """将指标记录到 W&B"""
    if not WANDB_AVAILABLE or not wandb.run:
        return

    # 添加前缀
    wandb_metrics = {}
    for key, value in metrics.items():
        wandb_key = f"{prefix}/{key}" if prefix else key
        wandb_metrics[wandb_key] = value

    wandb.log(wandb_metrics)


def log_layer_performance_chart(
    results: List[Dict],
    title: str,
    target_fpr: float = 0.01,
) -> None:
    """记录各层性能图表到 W&B（增强版：多子图）"""
    if not WANDB_AVAILABLE or not wandb.run or not results:
        return

    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # 提取数据
        layers = [r["layer_index"] for r in results]
        macro_f1 = [r["macro_f1"] for r in results]
        aupr = [r["aupr"] for r in results]
        auroc = [r["auroc"] for r in results]
        accuracy = [r.get("accuracy", 0) for r in results]
        tpr = [r.get("tpr_id", 0) for r in results]
        fpr = [r.get("fpr_id", 0) for r in results]
        thresholds = [r.get("threshold", 0) for r in results]
        
        # 混淆矩阵元素（如果存在）
        has_cm = all(k in results[0] for k in ["confusion_matrix_tn", "confusion_matrix_fp", 
                                                "confusion_matrix_fn", "confusion_matrix_tp"])
        if has_cm:
            tn = [r.get("confusion_matrix_tn", 0) for r in results]
            fp = [r.get("confusion_matrix_fp", 0) for r in results]
            fn = [r.get("confusion_matrix_fn", 0) for r in results]
            tp = [r.get("confusion_matrix_tp", 0) for r in results]

        # 创建多子图
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 子图1: 主要性能指标
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(layers, macro_f1, 'b-o', label='Macro F1', linewidth=2, markersize=6)
        ax1.plot(layers, aupr, 'r-s', label='AUPR', linewidth=2, markersize=6)
        ax1.plot(layers, auroc, 'g-^', label='AUROC', linewidth=2, markersize=6)
        if any(accuracy):
            ax1.plot(layers, accuracy, 'm-d', label='Accuracy', linewidth=2, markersize=6)
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Main Performance Metrics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: TPR/FPR
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(layers, tpr, 'g-o', label='TPR', linewidth=2, markersize=6)
        ax2.plot(layers, fpr, 'r-s', label='FPR', linewidth=2, markersize=6)
        ax2.axhline(y=target_fpr, color='k', linestyle='--', alpha=0.5, label=f'Target FPR ({target_fpr:.2%})')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Rate')
        ax2.set_title('TPR/FPR by Layer')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 子图3: 阈值变化
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(layers, thresholds, 'purple', marker='o', linewidth=2, markersize=6)
        ax3.set_xlabel('Layer Index')
        ax3.set_ylabel('Threshold')
        ax3.set_title('Threshold by Layer')
        ax3.grid(True, alpha=0.3)

        # 子图4: 混淆矩阵元素（如果存在）
        if has_cm:
            ax4 = fig.add_subplot(gs[1, :2])
            ax4.plot(layers, tp, 'g-o', label='TP', linewidth=2, markersize=6)
            ax4.plot(layers, tn, 'b-s', label='TN', linewidth=2, markersize=6)
            ax4.plot(layers, fp, 'r-^', label='FP', linewidth=2, markersize=6)
            ax4.plot(layers, fn, 'm-v', label='FN', linewidth=2, markersize=6)
            ax4.set_xlabel('Layer Index')
            ax4.set_ylabel('Count')
            ax4.set_title('Confusion Matrix Elements by Layer')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 子图5: 性能对比（归一化）
        ax5 = fig.add_subplot(gs[1, 2])
        # 归一化到 [0, 1] 以便对比
        metrics_normalized = {
            'Macro F1': np.array(macro_f1),
            'AUPR': np.array(aupr),
            'AUROC': np.array(auroc),
        }
        if any(accuracy):
            metrics_normalized['Accuracy'] = np.array(accuracy)
        
        for name, values in metrics_normalized.items():
            if values.max() > values.min():
                normalized = (values - values.min()) / (values.max() - values.min())
            else:
                normalized = values
            ax5.plot(layers, normalized, marker='o', label=name, linewidth=2, markersize=6)
        ax5.set_xlabel('Layer Index')
        ax5.set_ylabel('Normalized Score')
        ax5.set_title('Normalized Performance Comparison')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        # 记录到 W&B
        wandb.log({"layer_performance_comprehensive": wandb.Image(plt)})
        plt.close()

        # 同时记录简化版（向后兼容）
        plt.figure(figsize=(10, 6))
        plt.plot(layers, macro_f1, 'b-o', label='Macro F1', linewidth=2, markersize=6)
        plt.plot(layers, aupr, 'r-s', label='AUPR', linewidth=2, markersize=6)
        plt.plot(layers, auroc, 'g-^', label='AUROC', linewidth=2, markersize=6)
        if any(accuracy):
            plt.plot(layers, accuracy, 'm-d', label='Accuracy', linewidth=2, markersize=6)
        plt.xlabel('Layer Index')
        plt.ylabel('Performance Score')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        wandb.log({"layer_performance": wandb.Image(plt)})
        plt.close()

    except ImportError:
        LOGGER.warning("matplotlib 不可用，跳过图表生成")
    except Exception as exc:
        LOGGER.warning(f"图表生成失败: {exc}")


def log_kl_monitor_chart(
    results: List[Dict],
    monitor: Dict,
    title: str,
    total_layers: Optional[int] = None,
) -> None:
    if not WANDB_AVAILABLE or not wandb.run or not results or not monitor:
        return
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        layers = [int(entry.get("layer_index")) for entry in results]
        max_layer = total_layers or (max(layers) if layers else 0)
        x_axis = list(range(1, max_layer + 1))
        kl_map = {int(entry.get("layer_index")): entry.get("kl_neg") for entry in results}
        kl_values = [kl_map.get(idx, np.nan) for idx in x_axis]

        threshold = monitor.get("threshold")
        mean = monitor.get("mean")
        std = monitor.get("std", 0.0)
        sigma_factor = monitor.get("sigma_factor", 1.5)
        lower_band = mean - sigma_factor * std if mean is not None else None
        upper_band = mean + sigma_factor * std if mean is not None else None

        plt.figure(figsize=(10, 5))
        if lower_band is not None and upper_band is not None:
            plt.axhspan(
                lower_band,
                upper_band,
                color="red",
                alpha=0.08,
                label=f"Mean ± {sigma_factor}σ",
            )
        plt.plot(x_axis, kl_values, marker="o", label="KL (ID vs OOD negatives)")
        if threshold is not None:
            plt.axhline(threshold, color="r", linestyle="--", label=f"Alert threshold ({threshold:.4f})")

        alert_layers = [entry.get("layer_index") for entry in results if entry.get("kl_alert")]
        alert_values = [entry.get("kl_neg") for entry in results if entry.get("kl_alert")]
        if alert_layers:
            plt.scatter(alert_layers, alert_values, color="red", marker="x", s=80, label="Alerts")

        def annotate_layer(layer_idx: int, text: str) -> None:
            if layer_idx in kl_map and kl_map[layer_idx] is not None:
                plt.annotate(
                    text,
                    xy=(layer_idx, kl_map[layer_idx]),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="blue",
                )

        annotate_layer(15, f"L15: {kl_map.get(15):.3f}" if kl_map.get(15) is not None else "L15")
        if max_layer:
            annotate_layer(max_layer, f"L{max_layer}: {kl_map.get(max_layer):.3f}" if kl_map.get(max_layer) is not None else f"L{max_layer}")

        plt.xlabel("Layer Index")
        plt.ylabel("KL Divergence")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        wandb.log({"kl_monitor": wandb.Image(plt)})
        plt.close()
    except ImportError:
        LOGGER.warning("matplotlib 不可用，跳过 KL 监控图表")
    except Exception as exc:
        LOGGER.warning(f"KL 监控图表生成失败: {exc}")


def log_layer_drift_overview(
    results: List[Dict],
    drift_stats: Optional[List[Dict]],
    monitor: Optional[Dict],
    title: str,
    total_layers: Optional[int],
    stable_band: float = 0.1,
) -> None:
    if not WANDB_AVAILABLE or not wandb.run or not results or not drift_stats:
        return
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        max_layer = total_layers or max(r.get("layer_index", 0) for r in results)
        layers = list(range(1, max_layer + 1))

        delta_map = {int(d.get("layer_index")): d.get("delta_tau") for d in drift_stats}
        delta_vals = [delta_map.get(idx, np.nan) for idx in layers]

        kl_history = monitor.get("history") if monitor else []
        kl_map = {int(h.get("layer_index")): h.get("value") for h in kl_history}
        kl_vals = [kl_map.get(idx, np.nan) for idx in layers]
        threshold = monitor.get("threshold") if monitor else None
        mean = monitor.get("mean") if monitor else None
        std = monitor.get("std") if monitor else None
        sigma_factor = monitor.get("sigma_factor", 1.5) if monitor else 1.5
        lower_band = mean - sigma_factor * std if (mean is not None and std is not None) else None
        upper_band = mean + sigma_factor * std if (mean is not None and std is not None) else None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

        if stable_band is not None:
            ax1.axhspan(-stable_band, stable_band, color="green", alpha=0.1, label=f"Stable band ±{stable_band:.2f}")
        ax1.axhline(0, color="gray", linewidth=1, linestyle="--")
        ax1.bar(layers, delta_vals, color="#1f77b4")
        ax1.set_ylabel("Δτ (val→test)")
        ax1.set_title("Threshold Shift by Layer")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")

        if lower_band is not None and upper_band is not None:
            ax2.axhspan(lower_band, upper_band, color="red", alpha=0.08, label=f"Mean ± {sigma_factor}σ")
        ax2.plot(layers, kl_vals, marker="o", label="KL (ID vs OOD negatives)")
        if threshold is not None:
            ax2.axhline(threshold, color="r", linestyle="--", label=f"Alert threshold ({threshold:.3f})")
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("KL Divergence")
        ax2.set_title("KL Drift by Layer")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")

        fig.text(
            0.02,
            0.01,
            "Mid-layer stability (≈layers 11–18) aligns with Tenney/Jawahar's observation that intermediate semantics are more generalizable.",
            fontsize=8,
            color="dimgray",
        )
        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        wandb.log({"layer_drift_overview": wandb.Image(fig)})
        plt.close(fig)
    except ImportError:
        LOGGER.warning("matplotlib 不可用，跳过 drift 概览图表")
    except Exception as exc:
        LOGGER.warning(f"drift 图表生成失败: {exc}")


def log_best_layer_detailed_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    layer_idx: int,
    prefix: str = "best_layer"
) -> None:
    """记录最佳层的详细指标（ROC/PR 曲线、混淆矩阵等）"""
    if not WANDB_AVAILABLE or not wandb.run:
        return

    try:
        from src.utils.wandb_viz import (
            log_roc_curve,
            log_pr_curve,
            log_confusion_matrix,
            log_score_distribution,
        )

        y_pred = (y_prob >= threshold).astype(int)
        
        log_roc_curve(y_true, y_prob, title=f"{prefix} - ROC Curve (Layer {layer_idx})")
        log_pr_curve(y_true, y_prob, title=f"{prefix} - PR Curve (Layer {layer_idx})")
        log_confusion_matrix(y_true, y_pred, title=f"{prefix} - Confusion Matrix (Layer {layer_idx})")
        log_score_distribution(y_true, y_prob, threshold, title=f"{prefix} - Score Distribution (Layer {layer_idx})")
        
    except ImportError:
        LOGGER.warning("wandb_viz 模块不可用，跳过详细可视化")
    except Exception as exc:
        LOGGER.warning(f"详细指标记录失败: {exc}")


def log_best_model_summary(summary: Dict) -> None:
    """记录最佳模型摘要到 W&B"""
    if not WANDB_AVAILABLE or not wandb.run:
        return

    best = summary.get("best", {})
    if best:
        # 创建最佳模型摘要表格
        # 注意：W&B表格要求数值列必须是数字类型，不能是字符串
        layer_idx = best.get("layer_index")
        # 将层索引转换为整数，如果为None则使用-1作为占位符
        layer_value = int(layer_idx) if layer_idx is not None else -1
        table_data = [
            ["最佳层", layer_value],
            ["阈值", float(best.get('threshold', 0))],
            ["TPR (ID)", float(best.get('tpr_id', 0))],
            ["TPR (OOD)", float(best.get('tpr_ood', 0))],
            ["Drift Score", float(best.get('drift_score', 0))],
            ["Macro F1", float(best.get('macro_f1', 0))],
            ["AUPR", float(best.get('aupr', 0))],
            ["AUROC", float(best.get('auroc', 0))],
            ["Accuracy", float(best.get('accuracy', 0))],
        ]
        
        # 添加样本计数（如果存在）
        if 'num_samples' in summary:
            table_data.append(["总样本数", int(summary.get('num_samples', 0))])
        if 'num_samples_test' in best:
            table_data.append(["测试集样本数", int(best.get('num_samples_test', 0))])
        if 'num_pos_samples' in best:
            table_data.append(["正样本数", int(best.get('num_pos_samples', 0))])
        if 'num_neg_samples' in best:
            table_data.append(["负样本数", int(best.get('num_neg_samples', 0))])
        if 'num_pos_samples_test' in best:
            table_data.append(["测试集正样本数", int(best.get('num_pos_samples_test', 0))])
        if 'num_neg_samples_test' in best:
            table_data.append(["测试集负样本数", int(best.get('num_neg_samples_test', 0))])

        table = wandb.Table(
            columns=["指标", "值"],
            data=table_data
        )
        wandb.log({"best_model_summary": table})

        # 记录关键指标作为摘要
        wandb.run.summary["best_layer"] = best.get("layer_index")
        wandb.run.summary["best_macro_f1"] = best.get("macro_f1")
        wandb.run.summary["best_drift_score"] = best.get("drift_score")
        wandb.run.summary["best_threshold"] = best.get("threshold")
        wandb.run.summary["best_aupr"] = best.get("aupr")
        wandb.run.summary["best_auroc"] = best.get("auroc")
        wandb.run.summary["best_accuracy"] = best.get("accuracy", 0)
        
        # 记录样本计数到摘要
        if 'num_samples' in summary:
            wandb.run.summary["num_samples"] = summary.get('num_samples')
        if 'num_samples_test' in best:
            wandb.run.summary["num_samples_test"] = best.get('num_samples_test')
        if 'num_pos_samples' in best:
            wandb.run.summary["num_pos_samples"] = best.get('num_pos_samples')
        if 'num_neg_samples' in best:
            wandb.run.summary["num_neg_samples"] = best.get('num_neg_samples')
        if 'num_pos_samples_test' in best:
            wandb.run.summary["num_pos_samples_test"] = best.get('num_pos_samples_test')
        if 'num_neg_samples_test' in best:
            wandb.run.summary["num_neg_samples_test"] = best.get('num_neg_samples_test')


def evaluate_cv_layers(
    X_layers: np.ndarray,
    y: np.ndarray,
    clf_name: str,
    n_splits: int,
    target_fpr: float,
    lambda_penalty: float,
    bootstrap_samples: int,
    seed: int,
    progress: bool,
    view_specs: Optional[List[Dict[str, Any]]] = None,
    selection_strategy: str = "drift",
) -> Dict:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("数据仅包含单一类别，无法进行训练/评估。")
    splits = max(2, min(n_splits, int(counts.min())))
    LOGGER.info("使用 %d 折交叉验证。", splits)

    results: List[Dict] = []
    best_result: Optional[Dict] = None
    if view_specs is None:
        view_specs = [
            {
                "view_index": int(li + 1),
                "layer_index": int(li + 1),
                "layer_indices": [int(li + 1)],
                "view_label": f"L{li + 1}",
            }
            for li in range(X_layers.shape[1])
        ]

    layer_iter: Iterable[int] = range(X_layers.shape[1])
    if progress:
        layer_iter = tqdm(layer_iter, desc="Evaluating layers (CV)")

    for li in layer_iter:
        X = X_layers[:, li, :]
        view_spec = view_specs[li]
        preds = np.zeros_like(y, dtype=float)
        gamma = None
        if clf_name == "logreg":
            gamma = select_logreg_gamma(X, y)
            LOGGER.debug("Layer %d gamma selected: %s", li + 1, gamma)
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
        for train_idx, val_idx in skf.split(X, y):
            pipeline = build_classifier(clf_name, gamma=gamma)
            pipeline.fit(X[train_idx], y[train_idx])
            preds[val_idx] = predict_proba(pipeline, X[val_idx])

        id_stats = quantile_threshold(y, preds, target_fpr=target_fpr, adjust_threshold=True)
        y_pred = (preds >= id_stats["tau"]).astype(int)
        macro_f1 = float(f1_score(y, y_pred, average="macro"))
        aupr = float(average_precision_score(y, preds))
        auroc = float(roc_auc_score(y, preds))
        accuracy = float(accuracy_score(y, y_pred))
        
        # 计算混淆矩阵
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # CV 仅有 ID 数据，视作 OOD 与 ID 一致
        drift_score = id_stats["tpr"] - lambda_penalty * 0.0

        # 样本计数
        num_samples = int(len(y))
        num_pos = int((y == 1).sum())
        num_neg = int((y == 0).sum())

        metrics = {
            "threshold": id_stats["tau"],
            "tpr_id": id_stats["tpr"],
            "fpr_id": id_stats["fpr"],
            "z_id": id_stats["z_all"],
            "tpr_ood": id_stats["tpr"],
            "fpr_ood": id_stats["fpr"],
            "z_ood": id_stats["z_all"],
            "macro_f1": macro_f1,
            "aupr": aupr,
            "auroc": auroc,
            "accuracy": accuracy,
            "drift_score": drift_score,
            "reg_gamma": float(gamma) if gamma is not None else None,
            "num_samples": num_samples,
            "num_pos_samples": num_pos,
            "num_neg_samples": num_neg,
            # 混淆矩阵元素
            "confusion_matrix_tn": int(tn),
            "confusion_matrix_fp": int(fp),
            "confusion_matrix_fn": int(fn),
            "confusion_matrix_tp": int(tp),
            "view_index": int(view_spec["view_index"]),
            "view_label": view_spec["view_label"],
            "layer_indices": list(view_spec["layer_indices"]),
        }
        if view_spec.get("layer_index") is not None:
            metrics["layer_index"] = int(view_spec["layer_index"])

        if bootstrap_samples > 0:
            rng_seed = seed + li
            pos_scores = preds[y == 1]
            neg_scores = preds[y == 0]
            if pos_scores.size > 0:
                ci = bootstrap_interval(pos_scores, id_stats["tau"], bootstrap_samples, rng_seed)
                metrics["tpr_id_ci"] = [float(ci[0]), float(ci[1])]
            if neg_scores.size > 0:
                ci = bootstrap_interval(neg_scores, id_stats["tau"], bootstrap_samples, rng_seed + 1)
                metrics["fpr_id_ci"] = [float(ci[0]), float(ci[1])]
            metrics["tpr_ood_ci"] = metrics.get("tpr_id_ci")
            metrics["fpr_ood_ci"] = metrics.get("fpr_id_ci")

        metrics["selection_score"] = compute_selection_score(metrics, selection_strategy)
        log_metrics_to_wandb(metrics, "cv_layers")
        results.append(metrics)

        if best_result is None or metrics["selection_score"] > best_result["selection_score"]:
            best_result = metrics

    assert best_result is not None
    return {
        "mode": "cv",
        "num_samples": int(len(y)),
        "num_layers": int(X_layers.shape[1]),
        "hidden_size": int(X_layers.shape[2]),
        "clf": clf_name,
        "cv_splits": splits,
        "target_fpr": target_fpr,
        "lambda_penalty": lambda_penalty,
        "bootstrap_samples": bootstrap_samples,
        "selection_strategy": selection_strategy,
        "best": best_result,
        "all_layers": results,
    }


def evaluate_holdout_layers(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    clf_name: str,
    target_fpr: float,
    lambda_penalty: float,
    bootstrap_samples: int,
    seed: int,
    progress: bool,
    val_ood_ratio: float = 0.5,  # 验证集中用于OOD评估的比例
    view_specs: Optional[List[Dict[str, Any]]] = None,
    selection_strategy: str = "drift",
) -> Dict:
    """
    Layer selection using validation set only (Algorithm 2 in paper).

    The validation set is partitioned into val_id and val_ood for layer selection.
    Test set (X_te/y_te) is used ONLY for final reporting, NOT for layer selection.
    """
    results: List[Dict] = []
    best_result: Optional[Dict] = None
    if view_specs is None:
        view_specs = [
            {
                "view_index": int(li + 1),
                "layer_index": int(li + 1),
                "layer_indices": [int(li + 1)],
                "view_label": f"L{li + 1}",
            }
            for li in range(X_tr.shape[1])
        ]

    layer_iter: Iterable[int] = range(X_tr.shape[1])
    if progress:
        layer_iter = tqdm(layer_iter, desc="Evaluating layers (holdout)")

    layer_drift_table: List[Dict[str, float]] = []

    # 将验证集划分为 val_id 和 val_ood（与论文 Algorithm 2 一致）
    # 使用分层采样保持类别比例
    n_va = len(y_va)
    n_va_ood = int(n_va * val_ood_ratio)
    n_va_id = n_va - n_va_ood

    # 分层划分验证集
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ood_ratio, random_state=seed)
    va_id_idx, va_ood_idx = next(sss.split(np.arange(n_va), y_va))

    X_va_id, y_va_id = X_va[va_id_idx], y_va[va_id_idx]
    X_va_ood, y_va_ood = X_va[va_ood_idx], y_va[va_ood_idx]

    LOGGER.info("Validation set partitioned: val_id=%d, val_ood=%d (ratio=%.2f)",
                len(y_va_id), len(y_va_ood), val_ood_ratio)

    for li in layer_iter:
        view_spec = view_specs[li]
        gamma = None
        if clf_name == "logreg":
            gamma = select_logreg_gamma(X_tr[:, li, :], y_tr)
            LOGGER.debug("Holdout layer %d gamma selected: %s", li + 1, gamma)
        pipeline = build_classifier(clf_name, gamma=gamma)
        pipeline.fit(X_tr[:, li, :], y_tr)

        train_prob = predict_proba(pipeline, X_tr[:, li, :])

        # 层选择使用 val_id 和 val_ood（不使用测试集）
        va_id_prob = predict_proba(pipeline, X_va_id[:, li, :])
        va_ood_prob = predict_proba(pipeline, X_va_ood[:, li, :])

        # ID stats 来自 val_id
        id_stats = quantile_threshold(y_va_id, va_id_prob, target_fpr=target_fpr)
        # OOD stats 来自 val_ood（使用 ID 阈值评估）
        ood_stats = _evaluate_fixed_threshold(y_va_ood, va_ood_prob, tau=id_stats["tau"])

        # drift_score 仅基于验证集（与论文一致）
        drift_score = min(id_stats["tpr"], ood_stats["tpr"]) - lambda_penalty * abs(
            id_stats["z_all"] - ood_stats["z_all"]
        )

        # 测试集仅用于最终评估报告（不参与层选择）
        te_prob = predict_proba(pipeline, X_te[:, li, :])
        try:
            recal_stats = quantile_threshold(y_te, te_prob, target_fpr=target_fpr)
        except ValueError:
            recal_stats = {"tau": id_stats["tau"]}

        te_pred = (te_prob >= id_stats["tau"]).astype(int)
        macro_f1 = float(f1_score(y_te, te_pred, average="macro"))
        aupr = float(average_precision_score(y_te, te_prob))
        auroc = float(roc_auc_score(y_te, te_prob))
        accuracy = float(accuracy_score(y_te, te_pred))
        delta_tau = float(recal_stats["tau"] - id_stats["tau"])
        # KL stats: 比较 val_id 和 val_ood 的分布
        kl_stats = compute_layer_kl_metrics(va_id_prob, y_va_id, va_ood_prob, y_va_ood)

        # 样本计数
        num_samples_tr = int(len(y_tr))
        num_samples_va_id = int(len(y_va_id))
        num_samples_va_ood = int(len(y_va_ood))
        num_samples_te = int(len(y_te))
        num_pos_te = int((y_te == 1).sum())
        num_neg_te = int((y_te == 0).sum())

        metrics = {
            "threshold": float(id_stats["tau"]),
            "tpr_id": float(id_stats["tpr"]),
            "fpr_id": float(id_stats["fpr"]),
            "z_id": float(id_stats["z_all"]),
            "tpr_ood": float(ood_stats["tpr"]),
            "fpr_ood": float(ood_stats["fpr"]),
            "z_ood": float(ood_stats["z_all"]),
            "tau_recalibrated": float(recal_stats["tau"]),
            "delta_tau": delta_tau,
            "kl_neg": kl_stats.get("kl_neg"),
            "kl_pos": kl_stats.get("kl_pos"),
            "kl_all": kl_stats.get("kl_all"),
            "macro_f1": macro_f1,
            "aupr": aupr,
            "auroc": auroc,
            "accuracy": accuracy,
            "drift_score": float(drift_score),
            "train_score_mean": float(train_prob.mean()),
            "train_score_std": float(train_prob.std()),
            "reg_gamma": float(gamma) if gamma is not None else None,
            "num_samples_train": num_samples_tr,
            "num_samples_val_id": num_samples_va_id,
            "num_samples_val_ood": num_samples_va_ood,
            "num_samples_test": num_samples_te,
            "num_pos_samples_test": num_pos_te,
            "num_neg_samples_test": num_neg_te,
            "view_index": int(view_spec["view_index"]),
            "view_label": view_spec["view_label"],
            "layer_indices": list(view_spec["layer_indices"]),
        }
        if view_spec.get("layer_index") is not None:
            metrics["layer_index"] = int(view_spec["layer_index"])

        hist_counts, hist_edges = np.histogram(train_prob, bins=50, range=(0.0, 1.0), density=False)
        metrics["train_score_hist"] = hist_counts.astype(int).tolist()
        metrics["train_score_bins"] = hist_edges.tolist()

        if bootstrap_samples > 0:
            rng_seed = seed + li * 10
            # Bootstrap CI 使用 val_id 和 val_ood
            pos_va_id = va_id_prob[y_va_id == 1]
            neg_va_id = va_id_prob[y_va_id == 0]
            pos_va_ood = va_ood_prob[y_va_ood == 1]
            neg_va_ood = va_ood_prob[y_va_ood == 0]
            if pos_va_id.size > 0:
                ci = bootstrap_interval(pos_va_id, id_stats["tau"], bootstrap_samples, rng_seed)
                metrics["tpr_id_ci"] = [float(ci[0]), float(ci[1])]
            if neg_va_id.size > 0:
                ci = bootstrap_interval(neg_va_id, id_stats["tau"], bootstrap_samples, rng_seed + 1)
                metrics["fpr_id_ci"] = [float(ci[0]), float(ci[1])]
            if pos_va_ood.size > 0:
                ci = bootstrap_interval(pos_va_ood, id_stats["tau"], bootstrap_samples, rng_seed + 2)
                metrics["tpr_ood_ci"] = [float(ci[0]), float(ci[1])]
            if neg_va_ood.size > 0:
                ci = bootstrap_interval(neg_va_ood, id_stats["tau"], bootstrap_samples, rng_seed + 3)
                metrics["fpr_ood_ci"] = [float(ci[0]), float(ci[1])]

        metrics["selection_score"] = compute_selection_score(metrics, selection_strategy)
        log_metrics_to_wandb(metrics, "holdout_layers")
        results.append(metrics)
        layer_drift_table.append({
            "layer_index": metrics.get("layer_index"),
            "view_label": metrics.get("view_label"),
            "tau_id": metrics["threshold"],
            "tau_ood": metrics["tau_recalibrated"],
            "delta_tau": metrics["delta_tau"],
            "kl_neg": metrics["kl_neg"],
            "kl_pos": metrics["kl_pos"],
            "kl_all": metrics["kl_all"],
        })

        if best_result is None or metrics["selection_score"] > best_result["selection_score"]:
            best_result = metrics

    kl_monitor = annotate_kl_alerts(results, key="kl_neg")

    assert best_result is not None
    return {
        "mode": "holdout",
        "num_layers": int(X_tr.shape[1]),
        "hidden_size": int(X_tr.shape[2]),
        "clf": clf_name,
        "target_fpr": target_fpr,
        "lambda_penalty": lambda_penalty,
        "bootstrap_samples": bootstrap_samples,
        "val_ood_ratio": val_ood_ratio,
        "num_val_id": len(y_va_id),
        "num_val_ood": len(y_va_ood),
        "selection_strategy": selection_strategy,
        "best": best_result,
        "all_layers": results,
        "layer_drift_stats": layer_drift_table,
        "kl_monitor": kl_monitor,
    }


def stratified_holdout_indices(
    y: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    if len(np.unique(y)) < 2:
        raise ValueError("样本中仅有单一类别，无法划分 holdout。")
    if val_ratio + test_ratio >= 0.95:
        raise ValueError("val_ratio + test_ratio 过大，建议调小。")
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(idx, y))
    val_size_rel = val_ratio / (1 - test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size_rel, random_state=seed)
    train_idx_rel, val_idx_rel = next(sss2.split(train_val_idx, y[train_val_idx]))
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]
    return train_idx, val_idx, test_idx


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def ensure_parent_dir(path: Optional[str]) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def save_summary(summary: Dict, path: Optional[str]) -> None:
    if not path:
        return
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    LOGGER.info("结果已保存至 %s", path)


def make_cache_key(paths: Sequence[str], suffix: str) -> str:
    import hashlib

    joined = "\n".join(paths).encode("utf-8")
    digest = hashlib.md5(joined).hexdigest()
    return f"{digest}_{suffix}"


def load_texts_with_paths(paths: Sequence[str]) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    paths_str: List[str] = []
    for raw_path in paths:
        path = str(raw_path)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                texts.append(fh.read())
                paths_str.append(path)
        except Exception as exc:
            LOGGER.warning("Failed to load %s: %s", path, exc)
            texts.append("")
            paths_str.append(path)
    return texts, paths_str


def run_experiment(cfg: TrainConfig) -> Dict:
    # 初始化 W&B
    run_id = init_wandb(cfg)

    extractor = build_extractor(
        cfg.model_dir,
        cfg.gpu,
        trust_remote_code=cfg.trust_remote_code,
    )
    model_total_layers = extractor.num_model_layers

    if cfg.split_mode == "dir":
        datasets = {}
        for name, directory in (("train", cfg.train_dir), ("val", cfg.val_dir), ("test", cfg.test_dir)):
            assert directory is not None
            texts, labels, paths = load_md_reports(directory, cfg.limit)
            if not texts:
                raise RuntimeError(f"{name} 集为空：{directory}")
            # 缓存key包含所有影响特征提取的参数
            cache_suffix = f"{name}_layer{cfg.until_layer or 'all'}_mt{cfg.max_tokens}_s{cfg.stride}"
            cache_key = make_cache_key(paths, cache_suffix)
            features = maybe_cache_features(
                cfg.cache_dir,
                cache_key,
                lambda texts=texts: texts_to_features(
                    texts=texts,
                    extractor=extractor,
                    max_tokens=cfg.max_tokens,
                    stride=cfg.stride,
                    until_layer=cfg.until_layer,
                    progress=cfg.progress,
                ),
            )
            datasets[name] = (features, np.array(labels, dtype=int))

        X_train_views, view_specs = build_feature_views(
            datasets["train"][0],
            layer_pooling=cfg.layer_pooling,
            candidate_layers=cfg.candidate_layers,
        )
        X_val_views, _ = build_feature_views(
            datasets["val"][0],
            layer_pooling=cfg.layer_pooling,
            candidate_layers=cfg.candidate_layers,
        )
        X_test_views, _ = build_feature_views(
            datasets["test"][0],
            layer_pooling=cfg.layer_pooling,
            candidate_layers=cfg.candidate_layers,
        )

        summary = evaluate_holdout_layers(
            X_tr=X_train_views,
            y_tr=datasets["train"][1],
            X_va=X_val_views,
            y_va=datasets["val"][1],
            X_te=X_test_views,
            y_te=datasets["test"][1],
            clf_name=cfg.clf,
            target_fpr=cfg.target_fpr,
            lambda_penalty=cfg.lambda_penalty,
            bootstrap_samples=cfg.bootstrap_samples,
            seed=cfg.seed,
            progress=cfg.progress,
            view_specs=view_specs,
            selection_strategy=cfg.selection_strategy,
        )
        summary.update({
            "mode": "holdout-dir",
            "until_layer": int(cfg.until_layer) if cfg.until_layer is not None else None,
            "target_fpr": cfg.target_fpr,
            "lambda_penalty": cfg.lambda_penalty,
            "bootstrap_samples": cfg.bootstrap_samples,
        })
        summary["model_total_layers"] = model_total_layers

        if "all_layers" in summary and results_have_numeric_layers(summary["all_layers"]):
            mode_name = "Dir Mode"
            log_layer_performance_chart(
                summary["all_layers"],
                f"Holdout - {cfg.clf.upper()} - {mode_name}",
                target_fpr=cfg.target_fpr,
            )
            if summary.get("kl_monitor"):
                log_kl_monitor_chart(
                    summary["all_layers"],
                    summary["kl_monitor"],
                    f"KL Monitor - {mode_name}",
                    total_layers=summary.get("model_total_layers"),
                )
            if summary.get("layer_drift_stats"):
                log_layer_drift_overview(
                    summary["all_layers"],
                    summary["layer_drift_stats"],
                    summary.get("kl_monitor"),
                    f"Drift Overview - {mode_name}",
                    total_layers=summary.get("model_total_layers"),
                )
        log_best_model_summary(summary)
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
            benign_split_strategy=cfg.benign_split_strategy,
            random_seed=getattr(cfg, 'seed', 42),
        )
        
        train_texts, train_paths_str = load_texts_with_paths(train_paths)
        val_texts, val_paths_str = load_texts_with_paths(val_paths)
        test_texts, test_paths_str = load_texts_with_paths(test_paths)
        
        datasets = {}
        cache_target_layers = cfg.cache_up_to_layer or cfg.until_layer or extractor.num_model_layers
        cache_target_layers = min(cache_target_layers, extractor.num_model_layers)
        required_layers = cfg.until_layer
        for name, texts_list, labels_list, paths_str in [
            ("train", train_texts, train_labels_list, train_paths_str),
            ("val", val_texts, val_labels_list, val_paths_str),
            ("test", test_texts, test_labels_list, test_paths_str),
        ]:
            if not texts_list:
                raise RuntimeError(f"{name} 集为空")
            # 缓存key包含所有影响特征提取的参数
            cache_layer_label = cache_target_layers if cache_target_layers else "all"
            cache_suffix = f"{name}_layer{cache_layer_label}_mt{cfg.max_tokens}_s{cfg.stride}"
            cache_key = make_cache_key(paths_str, cache_suffix)
            features = maybe_cache_features(
                cfg.cache_dir,
                cache_key,
                lambda texts=texts_list: texts_to_features(
                    texts=texts,
                    extractor=extractor,
                    max_tokens=cfg.max_tokens,
                    stride=cfg.stride,
                    until_layer=cache_target_layers,
                    progress=cfg.progress,
                ),
                required_layers=required_layers,
            )
            datasets[name] = (features, np.array(labels_list, dtype=int))

        X_train_views, view_specs = build_feature_views(
            datasets["train"][0],
            layer_pooling=cfg.layer_pooling,
            candidate_layers=cfg.candidate_layers,
        )
        X_val_views, _ = build_feature_views(
            datasets["val"][0],
            layer_pooling=cfg.layer_pooling,
            candidate_layers=cfg.candidate_layers,
        )
        X_test_views, _ = build_feature_views(
            datasets["test"][0],
            layer_pooling=cfg.layer_pooling,
            candidate_layers=cfg.candidate_layers,
        )
        summary = evaluate_holdout_layers(
            X_tr=X_train_views,
            y_tr=datasets["train"][1],
            X_va=X_val_views,
            y_va=datasets["val"][1],
            X_te=X_test_views,
            y_te=datasets["test"][1],
            clf_name=cfg.clf,
            target_fpr=cfg.target_fpr,
            lambda_penalty=cfg.lambda_penalty,
            bootstrap_samples=cfg.bootstrap_samples,
            seed=cfg.seed,
            progress=cfg.progress,
            view_specs=view_specs,
            selection_strategy=cfg.selection_strategy,
        )
        summary.update({
            "mode": "holdout-dir" if cfg.split_mode == "dir" else "holdout-time_ood",
            "until_layer": int(cfg.until_layer) if cfg.until_layer is not None else None,
            "target_fpr": cfg.target_fpr,
            "lambda_penalty": cfg.lambda_penalty,
            "bootstrap_samples": cfg.bootstrap_samples,
        })
        summary["model_total_layers"] = model_total_layers

        # 记录图表和摘要
        if "all_layers" in summary and results_have_numeric_layers(summary["all_layers"]):
            mode_name = "Dir Mode" if cfg.split_mode == "dir" else "Time-OOD Mode"
            log_layer_performance_chart(
                summary["all_layers"],
                f"Holdout - {cfg.clf.upper()} - {mode_name}",
                target_fpr=cfg.target_fpr,
            )
            if summary.get("kl_monitor"):
                log_kl_monitor_chart(
                    summary["all_layers"],
                    summary["kl_monitor"],
                    f"KL Monitor - {mode_name}",
                    total_layers=summary.get("model_total_layers"),
                )
            if summary.get("layer_drift_stats"):
                log_layer_drift_overview(
                    summary["all_layers"],
                    summary["layer_drift_stats"],
                    summary.get("kl_monitor"),
                    f"Drift Overview - {mode_name}",
                    total_layers=summary.get("model_total_layers"),
                )
        log_best_model_summary(summary)
    elif cfg.split_mode == "split_spec":
        from src.utils.split_spec import load_named_split

        assert cfg.split_spec_path and cfg.split_name
        docs, fold, protocol = load_named_split(cfg.split_spec_path, cfg.split_name)
        all_paths = [str(item["path"]) for item in docs]
        all_labels = np.array([int(item["label"]) for item in docs], dtype=int)
        if not all_paths:
            raise RuntimeError("split_spec 模式未读取到任何路径。")

        cache_target_layers = cfg.cache_up_to_layer or cfg.until_layer or extractor.num_model_layers
        cache_target_layers = min(cache_target_layers, extractor.num_model_layers)
        required_layers = cfg.until_layer
        cache_layer_label = cache_target_layers if cache_target_layers else "all"
        cache_suffix = f"split_spec_layer{cache_layer_label}_mt{cfg.max_tokens}_s{cfg.stride}"
        cache_key = make_cache_key(all_paths, cache_suffix)
        X_layers = maybe_cache_features(
            cfg.cache_dir,
            cache_key,
            lambda paths=all_paths: texts_to_features(
                texts=load_texts_with_paths(paths)[0],
                extractor=extractor,
                max_tokens=cfg.max_tokens,
                stride=cfg.stride,
                until_layer=cache_target_layers,
                progress=cfg.progress,
            ),
            required_layers=required_layers,
        )

        train_ids = np.array(fold["train_ids"], dtype=int)
        val_ids = np.array(fold["val_ids"], dtype=int)
        test_ids = np.array(fold["test_ids"], dtype=int)
        X_train_views, view_specs = build_feature_views(
            X_layers[train_ids],
            layer_pooling=cfg.layer_pooling,
            candidate_layers=cfg.candidate_layers,
        )
        X_val_views, _ = build_feature_views(
            X_layers[val_ids],
            layer_pooling=cfg.layer_pooling,
            candidate_layers=cfg.candidate_layers,
        )
        X_test_views, _ = build_feature_views(
            X_layers[test_ids],
            layer_pooling=cfg.layer_pooling,
            candidate_layers=cfg.candidate_layers,
        )
        summary = evaluate_holdout_layers(
            X_tr=X_train_views,
            y_tr=all_labels[train_ids],
            X_va=X_val_views,
            y_va=all_labels[val_ids],
            X_te=X_test_views,
            y_te=all_labels[test_ids],
            clf_name=cfg.clf,
            target_fpr=cfg.target_fpr,
            lambda_penalty=cfg.lambda_penalty,
            bootstrap_samples=cfg.bootstrap_samples,
            seed=cfg.seed,
            progress=cfg.progress,
            view_specs=view_specs,
            selection_strategy=cfg.selection_strategy,
        )
        summary.update({
            "mode": "holdout-split_spec",
            "until_layer": int(cfg.until_layer) if cfg.until_layer is not None else None,
            "target_fpr": cfg.target_fpr,
            "lambda_penalty": cfg.lambda_penalty,
            "bootstrap_samples": cfg.bootstrap_samples,
            "split_name": cfg.split_name,
            "split_spec_path": cfg.split_spec_path,
            "held_out_family": fold.get("held_out_family"),
            "split_counts": fold.get("counts"),
            "protocol": protocol,
        })
        summary["model_total_layers"] = model_total_layers
        if "all_layers" in summary and results_have_numeric_layers(summary["all_layers"]):
            log_layer_performance_chart(
                summary["all_layers"],
                f"Holdout - {cfg.clf.upper()} - Split {cfg.split_name}",
                target_fpr=cfg.target_fpr,
            )
        log_best_model_summary(summary)
    else:
        # holdout/cv 模式：使用 mal_dir + benign_dir 加载数据
        mal_dir_path = Path(cfg.mal_dir)
        ben_dir_path = Path(cfg.benign_dir)
        
        mal_files = sorted(mal_dir_path.glob("*.md"))
        ben_files = sorted(ben_dir_path.glob("*.md"))
        
        texts = []
        labels = []
        paths = []
        
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
                paths.append(str(md_file))
            except Exception as e:
                LOGGER.warning(f"Failed to load {md_file}: {e}")
        
        LOGGER.info(f"加载了 {len(texts)} 个报告 (恶意: {len(mal_files)}, 良性: {len(ben_files)})")
        
        # 如果设置了 limit，进行分层抽样
        if cfg.limit and len(texts) > cfg.limit:
            from sklearn.model_selection import StratifiedShuffleSplit
            LOGGER.info("限制样本数至 %d，进行分层抽样。", cfg.limit)
            y_arr = np.array(labels, dtype=int)
            idx_all = np.arange(len(labels))
            sss = StratifiedShuffleSplit(n_splits=1, train_size=cfg.limit, random_state=42)
            keep_idx, _ = next(sss.split(idx_all, y_arr))
            texts = [texts[i] for i in keep_idx]
            labels = [labels[i] for i in keep_idx]
            paths = [paths[i] for i in keep_idx]
        
        if not texts:
            raise RuntimeError("未读取到有效样本，请检查数据目录。")
        
        # 提取元数据（用于高级可视化）
        metadata = extract_metadata_from_paths(paths, texts)
        # 缓存key包含所有影响特征提取的参数，确保参数变化时不会使用错误的缓存
        cache_target_layers = cfg.cache_up_to_layer or cfg.until_layer or extractor.num_model_layers
        cache_target_layers = min(cache_target_layers, extractor.num_model_layers)
        required_layers = cfg.until_layer
        cache_layer_label = cache_target_layers if cache_target_layers else "all"
        cache_suffix = f"full_layer{cache_layer_label}_mt{cfg.max_tokens}_s{cfg.stride}"
        cache_key = make_cache_key(paths, cache_suffix)
        X_layers = maybe_cache_features(
            cfg.cache_dir,
            cache_key,
            lambda texts=texts: texts_to_features(
                texts=texts,
                extractor=extractor,
                max_tokens=cfg.max_tokens,
                stride=cfg.stride,
                until_layer=cache_target_layers,
                progress=cfg.progress,
            ),
            required_layers=required_layers,
        )
        X_views, view_specs = build_feature_views(
            X_layers,
            layer_pooling=cfg.layer_pooling,
            candidate_layers=cfg.candidate_layers,
        )
        y = np.array(labels, dtype=int)
        if cfg.split_mode == "cv":
            summary = evaluate_cv_layers(
                X_layers=X_views,
                y=y,
                clf_name=cfg.clf,
                n_splits=cfg.n_splits,
                target_fpr=cfg.target_fpr,
                lambda_penalty=cfg.lambda_penalty,
                bootstrap_samples=cfg.bootstrap_samples,
                seed=cfg.seed,
                progress=cfg.progress,
                view_specs=view_specs,
                selection_strategy=cfg.selection_strategy,
            )
            summary["model_total_layers"] = model_total_layers

            # 记录图表和摘要
            if "all_layers" in summary and results_have_numeric_layers(summary["all_layers"]):
                log_layer_performance_chart(
                    summary["all_layers"],
                    f"CV - {cfg.clf.upper()} - {cfg.n_splits} Folds",
                    target_fpr=cfg.target_fpr,
                )
                # 记录最佳层的详细可视化
                best_layer_value = summary["best"].get("layer_index")
                best_layer_idx = int(best_layer_value) - 1 if isinstance(best_layer_value, int) else -1
                if best_layer_idx >= 0 and best_layer_idx < X_views.shape[1]:
                    # 重新计算最佳层的预测概率用于可视化
                    X_best = X_views[:, best_layer_idx, :]
                    probs_best = np.zeros_like(y, dtype=float)
                    classes, counts = np.unique(y, return_counts=True)
                    splits = max(2, min(cfg.n_splits, int(counts.min())))
                    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=cfg.seed)
                    for train_idx, val_idx in skf.split(X_best, y):
                        pipeline = build_classifier(cfg.clf)
                        pipeline.fit(X_best[train_idx], y[train_idx])
                        probs_best[val_idx] = predict_proba(pipeline, X_best[val_idx])
                    
                    log_best_layer_detailed_metrics(
                        y_true=y,
                        y_prob=probs_best,
                        threshold=summary["best"]["threshold"],
                        layer_idx=best_layer_idx + 1,
                        prefix="best_layer_cv"
                    )
                    
                    # 记录最佳层的综合性能面板
                    if WANDB_AVAILABLE and wandb.run:
                        try:
                            from src.utils.wandb_viz import log_best_layer_comprehensive_panel
                            log_best_layer_comprehensive_panel(
                                y_true=y,
                                y_prob=probs_best,
                                threshold=summary["best"]["threshold"],
                                layer_idx=best_layer_idx + 1,
                                metrics=summary["best"],
                                prefix="best_layer_cv"
                            )
                        except Exception as exc:
                            LOGGER.debug(f"最佳层综合面板记录失败: {exc}")
                        
                        # 记录高级可视化
                        try:
                            from src.utils.advanced_wandb_viz import (
                                log_score_distribution_with_divergence,
                                log_calibration_curve,
                                log_family_difference_heatmap,
                                log_false_positive_analysis,
                            )
                            
                            # 分数分布 + 距离指标
                            log_score_distribution_with_divergence(
                                y_true=y,
                                y_prob=probs_best,
                                threshold=summary["best"]["threshold"],
                                title=f"Best Layer {best_layer_idx + 1} - Score Distribution with Divergence"
                            )
                            
                            # 校准曲线
                            log_calibration_curve(
                                y_true=y,
                                y_prob=probs_best,
                                title=f"Best Layer {best_layer_idx + 1} - Calibration Curve"
                            )
                            
                            # 家族差异热力图（如果有多个家族）
                            if 'families' in metadata:
                                families_unique = list(set(metadata['families']))
                                if len(families_unique) > 1:
                                    scores_by_family = {}
                                    for i, family in enumerate(metadata['families']):
                                        if family not in scores_by_family:
                                            scores_by_family[family] = []
                                        scores_by_family[family].append(probs_best[i])
                                    scores_by_family = {k: np.array(v) for k, v in scores_by_family.items()}
                                    train_scores = probs_best  # 使用全部分数作为参考
                                    log_family_difference_heatmap(
                                        scores_by_family=scores_by_family,
                                        train_scores=train_scores,
                                        title=f"Best Layer {best_layer_idx + 1} - Family Difference Heatmap"
                                    )
                            
                            # 误报分析
                            y_pred_best = (probs_best >= summary["best"]["threshold"]).astype(int)
                            false_positives = []
                            for i in range(len(y)):
                                if y[i] == 0 and y_pred_best[i] == 1:  # False Positive
                                    fp_info = {
                                        'length': metadata.get('lengths', [0] * len(y))[i],
                                        'family': metadata.get('families', ['Unknown'] * len(y))[i],
                                        'ioc_coverage': metadata.get('ioc_coverage', [0] * len(y))[i],
                                        'score': float(probs_best[i]),
                                    }
                                    false_positives.append(fp_info)
                            
                            if len(false_positives) > 0:
                                log_false_positive_analysis(
                                    false_positives=false_positives,
                                    title=f"Best Layer {best_layer_idx + 1} - False Positive Root Cause Analysis"
                                )
                        except Exception as exc:
                            LOGGER.debug(f"高级可视化记录失败: {exc}")
            log_best_model_summary(summary)
        else:
            train_idx, val_idx, test_idx = stratified_holdout_indices(
                y=y,
                val_ratio=cfg.val_ratio,
                test_ratio=cfg.test_ratio,
                seed=cfg.seed,
            )
            summary = evaluate_holdout_layers(
                X_tr=X_views[train_idx],
                y_tr=y[train_idx],
                X_va=X_views[val_idx],
                y_va=y[val_idx],
                X_te=X_views[test_idx],
                y_te=y[test_idx],
                clf_name=cfg.clf,
                target_fpr=cfg.target_fpr,
                lambda_penalty=cfg.lambda_penalty,
                bootstrap_samples=cfg.bootstrap_samples,
                seed=cfg.seed,
                progress=cfg.progress,
                view_specs=view_specs,
                selection_strategy=cfg.selection_strategy,
            )
            summary.update({
                "mode": "holdout",
                "num_samples": int(len(y)),
                "val_ratio": cfg.val_ratio,
                "test_ratio": cfg.test_ratio,
                "target_fpr": cfg.target_fpr,
                "lambda_penalty": cfg.lambda_penalty,
                "bootstrap_samples": cfg.bootstrap_samples,
            })
            summary["model_total_layers"] = model_total_layers

            # 记录图表和摘要
            if "all_layers" in summary and results_have_numeric_layers(summary["all_layers"]):
                log_layer_performance_chart(
                    summary["all_layers"],
                    f"Holdout - {cfg.clf.upper()} - {cfg.val_ratio}/{cfg.test_ratio}",
                    target_fpr=cfg.target_fpr,
                )
                if summary.get("kl_monitor"):
                    log_kl_monitor_chart(
                        summary["all_layers"],
                        summary["kl_monitor"],
                        "KL Monitor - Holdout",
                        total_layers=summary.get("model_total_layers"),
                    )
                if summary.get("layer_drift_stats"):
                    log_layer_drift_overview(
                        summary["all_layers"],
                        summary["layer_drift_stats"],
                        summary.get("kl_monitor"),
                        "Drift Overview - Holdout",
                        total_layers=summary.get("model_total_layers"),
                    )
            log_best_model_summary(summary)

    summary.setdefault("model_total_layers", model_total_layers)
    if "evaluated_layers" not in summary:
        evaluated = summary.get("num_layers") or len(summary.get("all_layers", []))
        summary["evaluated_layers"] = int(evaluated)
    summary["config"] = {k: v for k, v in asdict(cfg).items() if k not in {"config_path"}}
    if cfg.save_raw:
        summary["save_raw"] = True
    if cfg.run_tag:
        summary["run_tag"] = cfg.run_tag

    # 完成实验，记录最终结果到 W&B
    if WANDB_AVAILABLE and wandb.run:
        if summary.get("kl_monitor"):
            history = summary["kl_monitor"].get("history") or []
            alert_count = sum(1 for entry in history if entry.get("alert"))
            wandb.run.summary["kl_alert_count"] = int(alert_count)
            stable_layers = [
                entry.get("layer_index")
                for entry in history
                if entry.get("alert") is False
            ]
            wandb.run.summary["stable_layer_span"] = stable_layers
        wandb.log({
            "experiment_completed": True,
            "final_best_f1": summary.get("best", {}).get("macro_f1", 0),
            "final_best_layer": summary.get("best", {}).get("layer_index", 0),
        })
        LOGGER.info("实验结果已记录到 W&B")
        wandb.finish()

    return summary


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
        "seeds": seeds,
        "per_seed": summaries,
        "best_mean": best_mean,
        "best_std": best_std,
        "config": {
            k: v
            for k, v in asdict(cfg).items()
            if k not in {"config_path", "seed"}
        },
    }
    return aggregated


def main() -> None:
    setup_logging()
    cfg = parse_config()
    LOGGER.info("实验配置：%s", cfg)
    if cfg.seeds:
        per_seed: List[Dict] = []
        for seed in cfg.seeds:
            seed_cfg = replace(cfg, seed=seed, seeds=None, out=None if cfg.out else None)
            LOGGER.info("开始运行种子 %d", seed)
            seed_summary = run_experiment(seed_cfg)
            seed_summary["seed"] = seed
            per_seed.append(seed_summary)
        aggregated = aggregate_seed_summaries(cfg, per_seed)
        print(json.dumps(aggregated, ensure_ascii=False, indent=2))
        save_summary(aggregated, cfg.out)
    else:
        summary = run_experiment(cfg)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        save_summary(summary, cfg.out)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        LOGGER.exception("运行失败：%s", exc)
        raise
