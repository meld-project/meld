"""Run Nebula BPE inference on the MELD CAPE JSON corpus."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# W&B integration (optional)
from src.utils.optional_imports import WANDB, ORJSON
from src.baselines.cape_adapter import cape_to_speakeasy_format
WANDB_AVAILABLE = WANDB.is_available()
wandb = WANDB.safe_import()

# ORJSON for fast JSON parsing (optional)
try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None

# --------------------------------------------------------------------------- #
# 依赖检查与路径配置
# --------------------------------------------------------------------------- #

ROOT_DIR = Path(__file__).resolve().parents[2]
NEBULA_REPO = ROOT_DIR / "third_party" / "nebula"
if not NEBULA_REPO.exists():
    raise RuntimeError(
        f"未找到 {NEBULA_REPO}。请先克隆 upstream 仓库："
        " git clone https://github.com/dtrizna/nebula third_party/nebula"
    )

if str(NEBULA_REPO) not in sys.path:
    sys.path.insert(0, str(NEBULA_REPO))

# ORJSON for fast JSON parsing (optional)
orjson = ORJSON.safe_import()

from nebula import Nebula

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("nebula_baseline")


def load_report(path: Path) -> dict:
    """Load a CAPE JSON report from disk."""
    data = path.read_bytes()
    if orjson is not None:
        return orjson.loads(data)
    return json.loads(data)


def predict_proba_gpu(nebula_model: Nebula, features: np.ndarray, device: str) -> float:
    """GPU 版本的 predict_proba，将输入张量移动到指定设备。
    
    Args:
        nebula_model: Nebula 模型实例
        features: 预处理后的特征数组
        device: 设备名称 ("cuda" 或 "cpu")
    
    Returns:
        预测概率值
    """
    dynamic_features = torch.Tensor(features).long().to(device)
    with torch.no_grad():
        logits = nebula_model.model(dynamic_features)
    return torch.sigmoid(logits).item()


def predict_proba_batch(nebula_model: Nebula, features_list: List[np.ndarray], device: str) -> List[float]:
    """批量 GPU 版本的 predict_proba，将多个输入合并成 batch 一起推理。
    
    Args:
        nebula_model: Nebula 模型实例
        features_list: 预处理后的特征数组列表
        device: 设备名称 ("cuda" 或 "cpu")
    
    Returns:
        预测概率值列表
    """
    if not features_list:
        return []
    
    # 将所有特征合并成 batch
    # 确保每个特征都是一维数组，然后堆叠成二维张量 [batch_size, seq_len]
    batch_features = torch.tensor(np.array(features_list), dtype=torch.long).to(device)
    
    # 确保维度正确：应该是 [batch_size, seq_len]
    if batch_features.dim() == 1:
        # 如果只有一个样本，添加 batch 维度
        batch_features = batch_features.unsqueeze(0)
    elif batch_features.dim() > 2:
        # 如果维度过多，可能需要 reshape
        batch_features = batch_features.view(batch_features.size(0), -1)
    
    with torch.no_grad():
        logits = nebula_model.model(batch_features)
        # 处理输出：如果 logits 是 [batch_size, 1]，需要 squeeze
        if logits.dim() > 1:
            logits = logits.squeeze(-1) if logits.size(-1) == 1 else logits
        probs = torch.sigmoid(logits).cpu().numpy()
    
    # 确保返回的是列表
    if isinstance(probs, np.ndarray):
        return probs.flatten().tolist()
    return [float(probs)] if not isinstance(probs, list) else probs


@dataclass
class TrainConfig:
    mal_dir: Optional[str] = None
    benign_dir: Optional[str] = None
    split_mode: str = "holdout"  # holdout / cv / dir / time_ood
    train_dir: Optional[str] = None
    val_dir: Optional[str] = None
    test_dir: Optional[str] = None
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    seed: int = 42
    seeds: Optional[List[int]] = None
    limit_per_class: Optional[int] = None
    n_splits: int = 5
    target_fpr: float = 0.01
    tokenizer: str = "bpe"
    seq_len: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_scores: bool = True
    progress: bool = False
    batch_size: int = 32  # 批处理大小
    out: Optional[str] = None
    save_raw: bool = False
    nebula_vocab_file: Optional[str] = None
    nebula_bpe_model_file: Optional[str] = None
    nebula_torch_model_file: Optional[str] = None
    nebula_model_config: Optional[str] = None

    # Time-OOD 模式参数
    train_end_date: Optional[str] = None  # Format: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"
    val_end_date: Optional[str] = None     # Format: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"
    malicious_manifest_path: Optional[str] = None  # Path to malicious manifest CSV
    benign_manifest_path: Optional[str] = None    # Path to benign manifest CSV

    # W&B 配置
    use_wandb: bool = False  # 是否启用 W&B 记录
    wandb_project: str = "lec-experiments"  # W&B 项目名
    wandb_entity: Optional[str] = None  # W&B 实体名（可选）
    wandb_api_key: Optional[str] = None  # W&B API 密钥
    run_tag: str = ""  # 运行标签

    config_path: Optional[str] = field(default=None, repr=False, compare=False)

    def validate(self) -> None:
        if self.split_mode not in {"holdout", "cv", "dir", "time_ood"}:
            raise ValueError(f"Invalid split_mode: {self.split_mode}")
        
        if self.split_mode == "time_ood":
            if not self.malicious_manifest_path:
                raise ValueError("malicious_manifest_path required for time_ood split mode")
            if not self.benign_manifest_path:
                raise ValueError("benign_manifest_path required for time_ood split mode")
            if not self.mal_dir or not self.benign_dir:
                raise ValueError("mal_dir and benign_dir required for time_ood split mode")
            # If dates not provided, use default optimal dates
            if not self.train_end_date:
                self.train_end_date = "2025-04-23 08:08:46"
            if not self.val_end_date:
                self.val_end_date = "2025-06-14 11:39:39"
        
        if self.split_mode not in {"holdout", "cv", "dir", "time_ood"}:
            raise ValueError("split_mode 必须是 holdout、cv、dir 或 time_ood")
        
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
            # holdout/cv 模式需要 mal_dir 和 benign_dir
            for key, name in [(self.mal_dir, "mal_dir"), (self.benign_dir, "benign_dir")]:
                if not key:
                    raise ValueError(f"{name} 必须指定 JSON 目录")
                path = Path(key)
                if not path.exists() or not path.is_dir():
                    raise FileNotFoundError(f"{name} 目录不存在：{path}")
        
        if self.split_mode == "holdout":
            if not (0 < self.val_ratio < 1) or not (0 < self.test_ratio < 1):
                raise ValueError("val_ratio/test_ratio 必须在 (0,1) 内")
            if self.val_ratio + self.test_ratio >= 0.95:
                raise ValueError("val_ratio + test_ratio 过大")
        if self.limit_per_class is not None and self.limit_per_class < 1:
            raise ValueError("limit_per_class 至少为 1")
        if not (0 < self.target_fpr < 1):
            raise ValueError("target_fpr 必须在 (0,1) 之间")
        if self.seeds:
            cleaned = sorted(dict.fromkeys(int(s) for s in self.seeds))
            if not cleaned:
                raise ValueError("seeds 不能为空")
            self.seeds = cleaned
        for attr, desc in [
            ("nebula_vocab_file", "nebula_vocab_file"),
            ("nebula_bpe_model_file", "nebula_bpe_model_file"),
            ("nebula_torch_model_file", "nebula_torch_model_file"),
            ("nebula_model_config", "nebula_model_config"),
        ]:
            value = getattr(self, attr)
            if value:
                path = Path(value)
                if not path.exists():
                    raise FileNotFoundError(f"{desc} 文件不存在：{path}")


def tpr_at_fpr(y_true: Iterable[int], scores: Iterable[float], target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    indices = np.where(fpr <= target_fpr)[0]
    if len(indices) == 0:
        return 0.0
    return float(tpr[indices[-1]])


def build_parser(defaults: Dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="JSON 配置文件路径")
    parser.add_argument("--mal_dir", type=str, default=defaults.get("mal_dir"))
    parser.add_argument("--benign_dir", type=str, default=defaults.get("benign_dir"))
    parser.add_argument("--split_mode", choices=["holdout", "cv"], default=defaults.get("split_mode", "holdout"))
    parser.add_argument("--val_ratio", type=float, default=defaults.get("val_ratio", 0.1))
    parser.add_argument("--test_ratio", type=float, default=defaults.get("test_ratio", 0.2))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--seeds", type=str, default=None, help="逗号分隔的随机种子，例如 40,41,42")
    parser.add_argument("--limit_per_class", type=int, default=defaults.get("limit_per_class"))
    parser.add_argument("--n_splits", type=int, default=defaults.get("n_splits", 5))
    parser.add_argument("--target_fpr", type=float, default=defaults.get("target_fpr", 0.01))
    parser.add_argument("--tokenizer", choices=["bpe", "whitespace"], default=defaults.get("tokenizer", "bpe"))
    parser.add_argument("--seq_len", type=int, default=defaults.get("seq_len", 512))
    parser.add_argument("--device", type=str, default=defaults.get("device", "cuda" if torch.cuda.is_available() else "cpu"), help="设备类型: cuda 或 cpu")
    parser.add_argument("--cache_scores", action="store_true", default=defaults.get("cache_scores", True))
    parser.add_argument("--no-cache_scores", action="store_false", dest="cache_scores")
    parser.add_argument("--progress", action="store_true", default=defaults.get("progress", False))
    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 32), help="批处理大小（用于加速推理）")
    parser.add_argument("--out", type=str, default=defaults.get("out"))
    parser.add_argument("--save_raw", action="store_true", default=defaults.get("save_raw", False))
    parser.add_argument("--use_wandb", action="store_true", default=defaults.get("use_wandb", False))
    parser.add_argument("--wandb_project", type=str, default=defaults.get("wandb_project", "lec-experiments"))
    parser.add_argument("--wandb_entity", type=str, default=defaults.get("wandb_entity"))
    parser.add_argument("--wandb_api_key", type=str, default=defaults.get("wandb_api_key"))
    parser.add_argument("--run_tag", type=str, default=defaults.get("run_tag", ""))
    parser.add_argument("--nebula_vocab_file", type=str, default=defaults.get("nebula_vocab_file"), help="Path to custom Nebula vocab JSON")
    parser.add_argument("--nebula_bpe_model_file", type=str, default=defaults.get("nebula_bpe_model_file"), help="Path to custom SentencePiece model")
    parser.add_argument("--nebula_torch_model_file", type=str, default=defaults.get("nebula_torch_model_file"), help="Path to custom Nebula torch weights")
    parser.add_argument("--nebula_model_config", type=str, default=defaults.get("nebula_model_config"), help="Path to custom Nebula model config JSON")
    return parser


def load_config_file(path: Optional[str]) -> Dict:
    """Load configuration from JSON file."""
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 JSON 对象，当前: {type(data)}")
    return data


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
# 数据加载与 Nebula 推理
# --------------------------------------------------------------------------- #

_SCORE_CACHE: Dict[
    Tuple[str, str, Optional[int], str, int, Tuple[Tuple[str, str], ...]],
    Tuple[np.ndarray, np.ndarray, List[str]]
] = {}
def _resolve_optional_path(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return None
    return str(Path(path_str).expanduser().resolve())


def _load_model_config(path_str: str) -> Dict:
    config_path = Path(path_str)
    if not config_path.exists():
        raise FileNotFoundError(f"nebula_model_config 文件不存在：{config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("nebula_model_config 必须是 JSON 对象")
    return data


def build_nebula_kwargs(cfg: TrainConfig) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    vocab_file = _resolve_optional_path(cfg.nebula_vocab_file)
    bpe_model_file = _resolve_optional_path(cfg.nebula_bpe_model_file)
    torch_model_file = _resolve_optional_path(cfg.nebula_torch_model_file)
    if vocab_file:
        kwargs["vocab_file"] = vocab_file
    if bpe_model_file:
        kwargs["bpe_model_file"] = bpe_model_file
    if torch_model_file:
        kwargs["torch_model_file"] = torch_model_file
    if cfg.nebula_model_config:
        kwargs["torch_model_config"] = _load_model_config(cfg.nebula_model_config)
    return kwargs


def _cache_key_for_kwargs(kwargs: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    items: List[Tuple[str, str]] = []
    for key, value in sorted(kwargs.items()):
        if isinstance(value, dict):
            items.append((key, json.dumps(value, sort_keys=True)))
        else:
            items.append((key, str(value)))
    return tuple(items)


def list_json_files(directory: str) -> List[Path]:
    paths = sorted(Path(directory).glob("*.json"))
    return [p for p in paths if p.is_file()]


def select_subset(paths: List[Path], limit: Optional[int]) -> List[Path]:
    if limit is None or limit >= len(paths):
        return paths
    return paths[:limit]


def load_json(path: Path) -> dict:
    """Backward compatibility wrapper using fast loader when available."""
    return load_report(path)


def _convert_to_speakeasy(report: Optional[dict], source: Path) -> Optional[dict]:
    """Ensure the report is in Speakeasy-compatible format for Nebula."""
    if not isinstance(report, dict):
        LOGGER.warning("报告 %s 不是有效的 JSON 对象，跳过", source.name)
        return None
    if "entry_points" in report:
        return report
    converted = cape_to_speakeasy_format(report)
    if converted is None:
        LOGGER.warning("报告 %s 缺少 entry_points，无法转换为 Speakeasy 格式", source.name)
        return None
    return converted


def load_speakeasy_report(path: Path) -> Optional[dict]:
    """Load CAPE JSON and convert to the Speakeasy structure Nebula expects."""
    try:
        raw_report = load_report(path)
    except Exception as exc:
        LOGGER.warning("读取 %s 失败: %s", path.name, exc)
        return None
    return _convert_to_speakeasy(raw_report, path)


def compute_nebula_scores(
    mal_dir: str,
    benign_dir: str,
    *,
    limit_per_class: Optional[int],
    tokenizer: str,
    seq_len: int,
    device: str = "cpu",
    progress: bool = False,
    use_cache: bool = True,
    batch_size: int = 32,
    nebula_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    nebula_kwargs = nebula_kwargs or {}
    cache_key = (
        mal_dir,
        benign_dir,
        limit_per_class,
        tokenizer,
        seq_len,
        device,
        _cache_key_for_kwargs(nebula_kwargs),
    )
    if use_cache and cache_key in _SCORE_CACHE:
        return _SCORE_CACHE[cache_key]

    nebula_model = Nebula(
        vocab_size=50000,
        seq_len=seq_len,
        tokenizer=tokenizer,
        **nebula_kwargs,
    )
    nebula_model.model.eval()
    nebula_model.model.to(device)
    LOGGER.info(f"使用设备: {device} 进行 Nebula 推理，批处理大小: {batch_size}")

    mal_paths = select_subset(list_json_files(mal_dir), limit_per_class)
    ben_paths = select_subset(list_json_files(benign_dir), limit_per_class)
    all_paths = [(path, 1) for path in mal_paths] + [(path, 0) for path in ben_paths]

    scores: List[float] = []
    labels: List[int] = []
    ids: List[str] = []

    # 批量处理以提高速度
    pbar = None
    if progress:
        pbar = tqdm(all_paths, desc="Nebula Inference (batch)", total=len(all_paths))
    iterator = pbar if pbar is not None else all_paths

    batch_items = []
    for idx, (path, label) in enumerate(iterator):
        batch_items.append((path, label))

        # 当达到 batch_size 或处理完所有文件时进行批量推理
        if len(batch_items) >= batch_size or idx == len(all_paths) - 1:
            batch_features = []
            batch_labels = []
            batch_ids = []
            batch_paths = []
            
            # 预处理 batch 中的所有文件
            for path_item, label_item in batch_items:
                try:
                    report = load_speakeasy_report(path_item)
                    if report is None:
                        continue
                    features = nebula_model.preprocess(report)
                    # 确保 features 是一维数组
                    if isinstance(features, np.ndarray):
                        features = features.flatten()
                    batch_features.append(features)
                    batch_labels.append(label_item)
                    batch_ids.append(path_item.stem)
                    batch_paths.append(path_item)
                except Exception as exc:  # pragma: no cover - 推理失败时记录
                    LOGGER.warning("处理失败 %s: %s", path_item.name, exc)
                    continue
            
            # 批量推理
            if batch_features:
                batch_probs = predict_proba_batch(nebula_model, batch_features, device)
                scores.extend(batch_probs)
                labels.extend(batch_labels)
                ids.extend(batch_ids)
            
            # 更新进度条（在清空 batch 之前）
            if progress:
                processed_count = len(batch_items)
                pbar.update(processed_count)
            
            # 清空 batch
            batch_items = []
    
    if progress:
        pbar.close()
    
    # 处理剩余的 items（理论上不应该有，因为已经在循环中处理了）
    if batch_items:
        batch_features = []
        batch_labels = []
        batch_ids = []
        
        for path_item, label_item in batch_items:
            try:
                report = load_speakeasy_report(path_item)
                if report is None:
                    continue
                features = nebula_model.preprocess(report)
                batch_features.append(features)
                batch_labels.append(label_item)
                batch_ids.append(path_item.stem)
            except Exception as exc:  # pragma: no cover - 推理失败时记录
                LOGGER.warning("处理失败 %s: %s", path_item.name, exc)
                continue
        
        if batch_features:
            batch_probs = predict_proba_batch(nebula_model, batch_features, device)
            scores.extend(batch_probs)
            labels.extend(batch_labels)
            ids.extend(batch_ids)

    if not scores:
        raise RuntimeError("未成功生成任何 Nebula 评分，请检查 JSON 格式或依赖。")

    scores_arr = np.array(scores, dtype=float)
    labels_arr = np.array(labels, dtype=int)
    if cache_key[2] is not None:
        LOGGER.info(
            "已加载每类 %d / %d 个样本，最终 %d 个样本进入评估。",
            limit_per_class,
            max(len(mal_paths), len(ben_paths)),
            len(scores_arr),
        )

    if use_cache:
        _SCORE_CACHE[cache_key] = (scores_arr, labels_arr, ids)
    return scores_arr, labels_arr, ids


# --------------------------------------------------------------------------- #
# 阈值与评估工具
# --------------------------------------------------------------------------- #


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

    mu_neg, std_neg = _safe_stats(neg_scores)
    mu_all, std_all = _safe_stats(y_score)
    z_neg = (tau - mu_neg) / std_neg
    z_all = (tau - mu_all) / std_all

    return {"tau": tau, "fpr": fpr, "tpr": tpr, "z_neg": float(z_neg), "z_all": float(z_all)}


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


# --------------------------------------------------------------------------- #
# 评估流程
# --------------------------------------------------------------------------- #


def evaluate_holdout_with_predefined_split(
    scores: np.ndarray,
    labels: np.ndarray,
    train_paths: List[Path],
    val_paths: List[Path],
    test_paths: List[Path],
    cfg: TrainConfig,
) -> Dict:
    """Evaluate with predefined train/val/test split (dir mode)."""
    n_train = len(train_paths)
    n_val = len(val_paths)
    n_test = len(test_paths)
    
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)
    
    train_scores = scores[train_idx]
    val_scores = scores[val_idx]
    test_scores = scores[test_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]
    
    tau_info = quantile_threshold(val_labels, val_scores, cfg.target_fpr)
    val_metrics = evaluate_at_threshold(val_labels, val_scores, tau_info["tau"])
    test_metrics = evaluate_at_threshold(test_labels, test_scores, tau_info["tau"])

    summary = {
        "mode": "holdout",
        "split_mode": "dir",
        "target_fpr": cfg.target_fpr,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "best": {
            "threshold": tau_info["tau"],
            "tpr_id": tau_info["tpr"],
            "fpr_id": tau_info["fpr"],
            "z_id": tau_info["z_all"],
            "macro_f1": test_metrics["macro_f1"],
            "auroc": test_metrics["auroc"],
            "aupr": test_metrics["aupr"],
            "accuracy": test_metrics["accuracy"],
            "tpr_ood": test_metrics["tpr"],
            "fpr_ood": test_metrics["fpr"],
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    return summary


def evaluate_holdout(scores: np.ndarray, labels: np.ndarray, cfg: TrainConfig) -> Dict:
    train_idx, val_idx, test_idx = stratified_holdout_indices(labels, cfg.val_ratio, cfg.test_ratio, cfg.seed)
    tau_info = quantile_threshold(labels[val_idx], scores[val_idx], cfg.target_fpr)
    val_metrics = evaluate_at_threshold(labels[val_idx], scores[val_idx], tau_info["tau"])
    test_metrics = evaluate_at_threshold(labels[test_idx], scores[test_idx], tau_info["tau"])

    summary = {
        "mode": "holdout",
        "target_fpr": cfg.target_fpr,
        "val_ratio": cfg.val_ratio,
        "test_ratio": cfg.test_ratio,
        "best": {
            "threshold": tau_info["tau"],
            "tpr_id": tau_info["tpr"],
            "fpr_id": tau_info["fpr"],
            "z_id": tau_info["z_all"],
            "macro_f1": test_metrics["macro_f1"],
            "auroc": test_metrics["auroc"],
            "aupr": test_metrics["aupr"],
            "accuracy": test_metrics["accuracy"],
            "tpr_ood": test_metrics["tpr"],
            "fpr_ood": test_metrics["fpr"],
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "indices": {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
        },
    }
    return summary


def evaluate_cv(scores: np.ndarray, labels: np.ndarray, cfg: TrainConfig) -> Dict:
    splits = max(2, min(cfg.n_splits, int(np.bincount(labels).min())))
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=cfg.seed)
    fold_summaries: List[Dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(scores, labels)):
        tau_info = quantile_threshold(labels[train_idx], scores[train_idx], cfg.target_fpr)
        val_metrics = evaluate_at_threshold(labels[val_idx], scores[val_idx], tau_info["tau"])
        fold_entry = {
            "fold": fold_idx,
            "threshold": tau_info["tau"],
            "tpr_id": tau_info["tpr"],
            "fpr_id": tau_info["fpr"],
            "z_id": tau_info["z_all"],
            "macro_f1": val_metrics["macro_f1"],
            "auroc": val_metrics["auroc"],
            "aupr": val_metrics["aupr"],
            "accuracy": val_metrics["accuracy"],
            "tpr_val": val_metrics["tpr"],
            "fpr_val": val_metrics["fpr"],
        }
        fold_summaries.append(fold_entry)

    metric_candidates = []
    for fold in fold_summaries:
        metric_candidates.extend(
            k for k, v in fold.items()
            if k != "fold" and isinstance(v, (int, float))
        )
    metrics_keys = set(metric_candidates)
    best_mean: Dict[str, float] = {}
    best_std: Dict[str, float] = {}
    for key in sorted(metrics_keys):
        values = [float(fold[key]) for fold in fold_summaries if isinstance(fold.get(key), (int, float))]
        if not values:
            continue
        best_mean[key] = float(statistics.mean(values))
        best_std[key] = float(statistics.stdev(values)) if len(values) > 1 else 0.0

    summary = {
        "mode": "cv",
        "target_fpr": cfg.target_fpr,
        "cv_splits": splits,
        "folds": fold_summaries,
        "best": best_mean,
        "best_std": best_std,
    }
    return summary


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
        run_name = f"nebula-{cfg.split_mode}"
        if cfg.run_tag:
            run_name += f"-{cfg.run_tag}"

        # 结束之前的 run（如果存在），避免并行执行时的 run 名称混乱
        if wandb.run is not None:
            wandb.finish()
        
        # 初始化 W&B
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=run_name,
            config=asdict(cfg),
            tags=[cfg.split_mode, "nebula_baseline"],
            reinit=True,  # 允许重新初始化，确保每个基线都有独立的 run
        )

        LOGGER.info(f"W&B 初始化成功，项目: {cfg.wandb_project}, 运行: {run_name}")
        return wandb.run.id

    except Exception as exc:
        LOGGER.warning(f"W&B 初始化失败: {exc}")
        return None


def log_metrics_to_wandb(metrics: Dict, prefix: str = "") -> None:
    """将指标记录到 W&B"""
    if not WANDB_AVAILABLE or not wandb.run:
        return

    # 添加前缀
    wandb_metrics = {}
    for key, value in metrics.items():
        wandb_key = f"{prefix}/{key}" if prefix else key
        wandb_metrics[wandb_key] = value

    try:
        wandb.log(wandb_metrics)
    except Exception:
        # W&B run 可能已结束，静默忽略
        pass


def run_experiment(cfg: TrainConfig) -> Dict:
    # 初始化 W&B
    run_id = init_wandb(cfg)
    nebula_kwargs = build_nebula_kwargs(cfg)

    if cfg.split_mode == "dir":
        # dir 模式：从预定义的 train/val/test 目录加载数据
        from src.baselines._nebula_common import load_cape_reports_from_dir
        
        train_paths, train_labels, train_ids = load_cape_reports_from_dir(
            cfg.train_dir, progress=cfg.progress
        )
        val_paths, val_labels, val_ids = load_cape_reports_from_dir(
            cfg.val_dir, progress=cfg.progress
        )
        test_paths, test_labels, test_ids = load_cape_reports_from_dir(
            cfg.test_dir, progress=cfg.progress
        )
        
        # 计算所有数据的 scores
        all_paths = train_paths + val_paths + test_paths
        all_labels = train_labels + val_labels + test_labels
        all_ids = train_ids + val_ids + test_ids
        
        # 计算 scores
        scores_list = []
        nebula_model = Nebula(
            vocab_size=50000,
            seq_len=cfg.seq_len,
            tokenizer=cfg.tokenizer,
            **nebula_kwargs,
        )
        nebula_model.model.eval()
        nebula_model.model.to(cfg.device)
        LOGGER.info(f"使用设备: {cfg.device} 进行 Nebula 推理，批处理大小: {cfg.batch_size}")

        # 批量处理以提高速度
        batch_size = cfg.batch_size
        pbar = None
        if cfg.progress:
            pbar = tqdm(all_paths, desc="Nebula Inference (batch)", total=len(all_paths))
        iterator = pbar if pbar is not None else all_paths

        batch_paths = []
        batch_indices = []
        
        for idx, path in enumerate(iterator):
            batch_paths.append(path)
            batch_indices.append(idx)
            
            # 当达到 batch_size 或处理完所有文件时进行批量推理
            if len(batch_paths) >= batch_size or idx == len(all_paths) - 1:
                batch_features = []
                batch_valid_indices = []
                
                # 预处理 batch 中的所有文件
                for batch_path in batch_paths:
                    try:
                        report = load_speakeasy_report(batch_path)
                        if report is None:
                            batch_features.append(None)
                            continue
                        features = nebula_model.preprocess(report)
                        # 确保 features 是一维数组
                        if isinstance(features, np.ndarray):
                            features = features.flatten()
                        batch_features.append(features)
                        batch_valid_indices.append(len(batch_features) - 1)
                    except Exception as exc:
                        LOGGER.warning("处理失败 %s: %s", batch_path.name, exc)
                        batch_features.append(None)
                
                # 批量推理
                if batch_features and any(f is not None for f in batch_features):
                    valid_features = [f for f in batch_features if f is not None]
                    if valid_features:
                        batch_probs = predict_proba_batch(nebula_model, valid_features, cfg.device)
                        
                        # 将结果填充回 scores_list
                        prob_idx = 0
                        for i, features in enumerate(batch_features):
                            if features is not None:
                                scores_list.append(batch_probs[prob_idx])
                                prob_idx += 1
                            else:
                                scores_list.append(0.5)  # 失败的文件使用默认分数
                    else:
                        # 所有文件都失败了
                        scores_list.extend([0.5] * len(batch_paths))
                else:
                    # 没有有效文件
                    scores_list.extend([0.5] * len(batch_paths))
                
                # 更新进度条（在清空 batch 之前）
                if cfg.progress:
                    processed_count = len(batch_paths)
                    pbar.update(processed_count)
                
                # 清空 batch
                batch_paths = []
                batch_indices = []
        
        if cfg.progress:
            pbar.close()
        
        scores = np.array(scores_list)
        labels = np.array(all_labels)
        ids = all_ids
        
        # 使用 holdout 评估，但使用预定义的划分
        summary = evaluate_holdout_with_predefined_split(
            scores, labels, train_paths, val_paths, test_paths, cfg
        )
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
        
        # 合并所有数据
        all_paths = train_paths + val_paths + test_paths
        all_labels = train_labels_list + val_labels_list + test_labels_list
        all_ids = train_ids_list + val_ids_list + test_ids_list
        
        # 计算 scores (nebula 需要 JSON 文件，但这里路径是 .md，需要转换)
        # 注意：time_ood 模式假设使用 markdown 文件，但 nebula 需要 JSON
        # 这里需要根据实际情况调整
        LOGGER.warning("time_ood mode with nebula: 需要确保路径指向 JSON 文件")
        
        scores_list = []
        nebula_model = Nebula(
            vocab_size=50000,
            seq_len=cfg.seq_len,
            tokenizer=cfg.tokenizer,
            **nebula_kwargs,
        )
        nebula_model.model.eval()
        nebula_model.model.to(cfg.device)
        LOGGER.info(f"使用设备: {cfg.device} 进行 Nebula 推理，批处理大小: {cfg.batch_size}")

        # 批量处理以提高速度
        batch_size = cfg.batch_size
        pbar = None
        if cfg.progress:
            pbar = tqdm(all_paths, desc="Nebula Inference (batch)", total=len(all_paths))
        iterator = pbar if pbar is not None else all_paths

        batch_paths = []
        batch_json_paths = []
        
        for idx, path in enumerate(iterator):
            # 尝试加载 JSON（如果路径是 .md，需要找到对应的 JSON）
            json_path = path.with_suffix('.json') if path.suffix == '.md' else path
            batch_paths.append(path)
            batch_json_paths.append(json_path)
            
            # 当达到 batch_size 或处理完所有文件时进行批量推理
            if len(batch_paths) >= batch_size or idx == len(all_paths) - 1:
                batch_features = []
                batch_valid_indices = []
                
                # 预处理 batch 中的所有文件
                for json_path_item, path_item in zip(batch_json_paths, batch_paths):
                    try:
                        if not json_path_item.exists():
                            LOGGER.warning(f"JSON file not found for {path_item}, using default score")
                            batch_features.append(None)
                            continue
                        report = load_speakeasy_report(json_path_item)
                        if report is None:
                            batch_features.append(None)
                            continue
                        features = nebula_model.preprocess(report)
                        # 确保 features 是一维数组
                        if isinstance(features, np.ndarray):
                            features = features.flatten()
                        batch_features.append(features)
                    except Exception as exc:
                        LOGGER.warning("处理失败 %s: %s", path_item.name, exc)
                        batch_features.append(None)
                
                # 批量推理
                if batch_features and any(f is not None for f in batch_features):
                    valid_features = [f for f in batch_features if f is not None]
                    if valid_features:
                        batch_probs = predict_proba_batch(nebula_model, valid_features, cfg.device)
                        
                        # 将结果填充回 scores_list
                        prob_idx = 0
                        for features in batch_features:
                            if features is not None:
                                scores_list.append(batch_probs[prob_idx])
                                prob_idx += 1
                            else:
                                scores_list.append(0.5)  # 失败的文件使用默认分数
                    else:
                        # 所有文件都失败了
                        scores_list.extend([0.5] * len(batch_paths))
                else:
                    # 没有有效文件
                    scores_list.extend([0.5] * len(batch_paths))
                
                # 清空 batch
                batch_paths = []
                batch_json_paths = []
                
                # 更新进度条（在清空 batch 之前）
                if cfg.progress:
                    processed_count = len(batch_paths)
                    pbar.update(processed_count)
                
                # 清空 batch
                batch_paths = []
                batch_json_paths = []
        
        if cfg.progress:
            pbar.close()
        
        scores = np.array(scores_list)
        labels = np.array(all_labels)
        ids = all_ids
        
        # 使用 holdout 评估，但使用预定义的划分
        summary = evaluate_holdout_with_predefined_split(
            scores, labels, train_paths, val_paths, test_paths, cfg
        )
    else:
        # 原有的 holdout/cv 模式
        scores, labels, ids = compute_nebula_scores(
            cfg.mal_dir,
            cfg.benign_dir,
            limit_per_class=cfg.limit_per_class,
            tokenizer=cfg.tokenizer,
            seq_len=cfg.seq_len,
            device=cfg.device,
            progress=cfg.progress,
            use_cache=cfg.cache_scores,
            batch_size=cfg.batch_size,
            nebula_kwargs=nebula_kwargs,
        )

        if cfg.split_mode == "holdout":
            summary = evaluate_holdout(scores, labels, cfg)
        else:
            summary = evaluate_cv(scores, labels, cfg)

    # 记录指标到 W&B
    if "best" in summary:
        log_metrics_to_wandb(summary["best"], prefix=cfg.split_mode)
        # 记录摘要
        if WANDB_AVAILABLE and wandb.run:
            wandb.run.summary.update({
                "best_macro_f1": summary["best"].get("macro_f1", 0),
                "best_auroc": summary["best"].get("auroc", 0),
                "best_aupr": summary["best"].get("aupr", 0),
                "best_threshold": summary["best"].get("threshold", 0),
            })
            wandb.finish()
            LOGGER.info("实验结果已记录到 W&B")

    if cfg.save_raw:
        summary["raw_scores"] = {
            "ids": ids,
            "scores": scores.tolist(),
            "labels": labels.tolist(),
        }

    if cfg.split_mode in {"dir", "time_ood"}:
        if cfg.split_mode == "dir":
            summary["config_digest"] = {
                "train_dir": str(Path(cfg.train_dir).resolve()),
                "val_dir": str(Path(cfg.val_dir).resolve()),
                "test_dir": str(Path(cfg.test_dir).resolve()),
                "tokenizer": cfg.tokenizer,
                "seq_len": cfg.seq_len,
            }
        else:  # time_ood
            summary["config_digest"] = {
                "train_end_date": getattr(cfg, 'train_end_date', '2025-04-23 08:08:46'),
                "val_end_date": getattr(cfg, 'val_end_date', '2025-06-14 11:39:39'),
                "malicious_manifest_path": str(Path(cfg.malicious_manifest_path).resolve()),
                "benign_manifest_path": str(Path(cfg.benign_manifest_path).resolve()),
                "tokenizer": cfg.tokenizer,
                "seq_len": cfg.seq_len,
            }
    else:
        summary["config_digest"] = {
            "mal_dir": str(Path(cfg.mal_dir).resolve()),
            "benign_dir": str(Path(cfg.benign_dir).resolve()),
            "limit_per_class": cfg.limit_per_class,
            "tokenizer": cfg.tokenizer,
            "seq_len": cfg.seq_len,
        }
    return summary


# --------------------------------------------------------------------------- #
# 输出管理
# --------------------------------------------------------------------------- #


def save_summary(summary: Dict, path: Optional[str]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    LOGGER.info("结果已保存至 %s", out_path)


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


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
        "config": {k: v for k, v in asdict(cfg).items() if k not in {"config_path", "seed"}},
    }
    return aggregated


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Nebula baseline experiments")
    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument("--mal_dir", type=str, help="Malicious JSON directory")
    parser.add_argument("--benign_dir", type=str, help="Benign JSON directory")
    parser.add_argument("--split_mode", type=str, default="holdout", choices=["holdout", "cv", "dir", "time_ood"])
    parser.add_argument("--limit_per_class", type=int, help="Limit samples per class")
    parser.add_argument("--out", type=str, help="Output JSON file")
    parser.add_argument("--progress", action="store_true", help="Show progress bars")
    # Time-OOD specific arguments
    parser.add_argument("--train_end_date", type=str, help="Train end date for time-OOD split")
    parser.add_argument("--val_end_date", type=str, help="Validation end date for time-OOD split")
    parser.add_argument("--malicious_manifest_path", type=str, help="Path to malicious manifest CSV")
    parser.add_argument("--benign_manifest_path", type=str, help="Path to benign manifest CSV")
    # Dir mode specific arguments
    parser.add_argument("--train_dir", type=str, help="Training directory for dir split mode")
    parser.add_argument("--val_dir", type=str, help="Validation directory for dir split mode")
    parser.add_argument("--test_dir", type=str, help="Test directory for dir split mode")
    # Other common arguments
    parser.add_argument("--tokenizer", type=str, default="bpe", choices=["bpe", "whitespace"], help="Tokenizer type")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--target_fpr", type=float, default=0.01, help="Target FPR")
    parser.add_argument("--seeds", type=str, help="Comma-separated seeds")
    parser.add_argument("--use_wandb", action="store_true", help="Use WandB")
    parser.add_argument("--wandb_project", type=str, help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, help="WandB entity")
    parser.add_argument("--wandb_api_key", type=str, help="WandB API key")
    parser.add_argument("--run_tag", type=str, help="WandB run tag")
    parser.add_argument("--cache_scores", action="store_true", help="Cache scores")
    
    args = parser.parse_args()
    
    # Load config
    config_dict = load_config_file(args.config) if args.config else {}
    
    # Merge args and config_dict, with args taking precedence
    merged_config = {**config_dict}
    if args.mal_dir:
        merged_config["mal_dir"] = args.mal_dir
    if args.benign_dir:
        merged_config["benign_dir"] = args.benign_dir
    if args.split_mode:
        merged_config["split_mode"] = args.split_mode
    if args.limit_per_class is not None:
        merged_config["limit_per_class"] = args.limit_per_class
    if args.out:
        merged_config["out"] = args.out
    if args.progress:
        merged_config["progress"] = True
    if args.train_end_date:
        merged_config["train_end_date"] = args.train_end_date
    if args.val_end_date:
        merged_config["val_end_date"] = args.val_end_date
    if args.malicious_manifest_path:
        merged_config["malicious_manifest_path"] = args.malicious_manifest_path
    if args.benign_manifest_path:
        merged_config["benign_manifest_path"] = args.benign_manifest_path
    if args.train_dir:
        merged_config["train_dir"] = args.train_dir
    if args.val_dir:
        merged_config["val_dir"] = args.val_dir
    if args.test_dir:
        merged_config["test_dir"] = args.test_dir
    if args.tokenizer:
        merged_config["tokenizer"] = args.tokenizer
    if args.seq_len:
        merged_config["seq_len"] = args.seq_len
    if args.device:
        merged_config["device"] = args.device
    if args.batch_size:
        merged_config["batch_size"] = args.batch_size
    if args.target_fpr:
        merged_config["target_fpr"] = args.target_fpr
    if args.seeds:
        merged_config["seeds"] = [int(s.strip()) for s in args.seeds.split(",")]
    if args.use_wandb:
        merged_config["use_wandb"] = True
    if args.wandb_project:
        merged_config["wandb_project"] = args.wandb_project
    if args.wandb_entity:
        merged_config["wandb_entity"] = args.wandb_entity
    if args.wandb_api_key:
        merged_config["wandb_api_key"] = args.wandb_api_key
    if args.run_tag:
        merged_config["run_tag"] = args.run_tag
    if args.cache_scores:
        merged_config["cache_scores"] = True
    
    # Create config
    cfg = TrainConfig(**merged_config)
    
    cfg.validate()
    
    # Run experiment
    summary = run_experiment(cfg)
    
    # Save results
    save_summary(summary, cfg.out)
    
    # Print summary
    if "best" in summary:
        best = summary["best"]
        LOGGER.info("=" * 70)
        LOGGER.info("实验结果")
        LOGGER.info("=" * 70)
        LOGGER.info("Baseline: Nebula")
        LOGGER.info("Mode: %s", cfg.split_mode)
        LOGGER.info("AUC: %.4f", best.get("auroc", 0))
        LOGGER.info("F1: %.4f", best.get("macro_f1", 0))
        LOGGER.info("=" * 70)


if __name__ == "__main__":
    setup_logging()
    main()
