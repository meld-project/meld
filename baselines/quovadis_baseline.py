"""QuoVadis baseline model for CAPE JSON reports.

QuoVadis extracts API sequences and uses 1D CNN architecture.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

# Nebula imports
ROOT_DIR = Path(__file__).resolve().parents[2]
NEBULA_REPO = ROOT_DIR / "third_party" / "nebula"
if str(NEBULA_REPO) not in sys.path:
    sys.path.insert(0, str(NEBULA_REPO))

try:
    from nebula.models.quovadis import QuoVadisModel
    from nebula.models.quovadis.preprocessor import flatten
except ImportError as e:
    raise RuntimeError(f"无法导入 nebula 模块: {e}") from e

from src.baselines._nebula_common import (
    aggregate_seed_summaries,
    evaluate_cv,
    evaluate_holdout,
    init_wandb,
    load_cape_reports,
    load_config_file,
    load_json,
    log_metrics_to_wandb,
    save_summary,
    setup_logging,
    ModelTrainer,
    BCEWithLogitsLoss,
    AdamW,
    WANDB_AVAILABLE,
    wandb,
)
from src.baselines.cape_adapter import extract_api_sequence_from_cape

LOGGER = logging.getLogger("quovadis_baseline")


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    """Configuration for QuoVadis baseline training."""
    mal_dir: Optional[str] = None
    benign_dir: Optional[str] = None
    split_mode: str = "holdout"  # holdout / cv / dir
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
    
    # QuoVadis specific
    seq_len: int = 150
    top_api: int = 600
    
    # Training
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 2.5e-4
    weight_decay: float = 1e-2
    device: str = "cuda"
    
    # Output
    out: Optional[str] = None
    save_raw: bool = False
    progress: bool = False
    
    # W&B 配置
    use_wandb: bool = False
    wandb_project: str = "nebula-baselines"
    wandb_entity: Optional[str] = None
    wandb_api_key: Optional[str] = None
    run_tag: str = ""
    
    config_path: Optional[str] = field(default=None, repr=False, compare=False)
    
    def validate(self) -> None:
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


# --------------------------------------------------------------------------- #
# 预处理函数
# --------------------------------------------------------------------------- #


def preprocess_quovadis(
    paths: List[Path],
    model: QuoVadisModel,
    progress: bool = False,
) -> np.ndarray:
    """Preprocess CAPE JSON reports for QuoVadis."""
    api_sequences = []
    iterator = paths
    if progress:
        iterator = tqdm(iterator, desc="Preprocessing QuoVadis", total=len(paths))
    
    for path in iterator:
        try:
            report = load_json(path)
            api_seq = extract_api_sequence_from_cape(report)
            api_sequences.append(api_seq)
        except Exception as e:
            LOGGER.warning(f"Failed to preprocess {path.name}: {e}")
            api_sequences.append([])  # Empty sequence as fallback
    
    # Convert to array
    return model.apisequences_to_arr(api_sequences)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def run_experiment(cfg: TrainConfig) -> Dict:
    """Run QuoVadis baseline experiment."""
    baseline_name = "quovadis"
    
    # Initialize W&B
    run_id = init_wandb(cfg, baseline_name)
    
    # Load data
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
        
        # 合并所有数据
        paths = train_paths + val_paths + test_paths
        labels = train_labels + val_labels + test_labels
        ids = train_ids + val_ids + test_ids
        
        # 保存划分信息用于后续评估
        n_train = len(train_paths)
        n_val = len(val_paths)
        n_test = len(test_paths)
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
        paths = train_paths + val_paths + test_paths
        labels = train_labels_list + val_labels_list + test_labels_list
        ids = train_ids_list + val_ids_list + test_ids_list
        
        # 保存划分信息用于后续评估
        n_train = len(train_paths)
        n_val = len(val_paths)
        n_test = len(test_paths)
    else:
        # 原有的 holdout/cv 模式
        LOGGER.info("Loading CAPE JSON reports...")
        paths, labels, ids = load_cape_reports(
            cfg.mal_dir,
            cfg.benign_dir,
            limit_per_class=cfg.limit_per_class,
            progress=cfg.progress,
        )
        n_train = n_val = n_test = None
    
    if not paths:
        raise RuntimeError("未读取到任何样本，请检查 JSON 目录。")
    
    labels_arr = np.array(labels, dtype=int)
    
    # Build vocabulary from API sequences
    LOGGER.info("Building vocabulary for QuoVadis...")
    if cfg.split_mode in {"dir", "time_ood"}:
        # dir 模式：只使用训练集构建词汇表
        train_paths_for_vocab = paths[:n_train]
    else:
        # holdout/cv 模式：使用前一半数据构建词汇表
        train_paths_for_vocab = paths[:len(paths) // 2]
    train_api_sequences = []
    for path in train_paths_for_vocab:
        try:
            report = load_json(path)
            api_seq = extract_api_sequence_from_cape(report)
            train_api_sequences.append(api_seq)
        except Exception:
            continue
    
    model = QuoVadisModel(seq_len=cfg.seq_len)
    # Build vocab from sequences directly
    api_counter = Counter(flatten(train_api_sequences))
    api_calls_preserved = [x[0] for x in api_counter.most_common(cfg.top_api - 2)]
    model.vocab = dict(zip(['<pad>', '<other>'] + api_calls_preserved, range(len(api_calls_preserved) + 2)))
    model.reverse_vocab = {v: k for k, v in model.vocab.items()}
    
    # Preprocess all data
    LOGGER.info("Preprocessing all data...")
    X = preprocess_quovadis(paths, model, progress=cfg.progress)
    
    LOGGER.info(f"Preprocessed data shape: {X.shape}")
    
    # Model config
    model_config = {
        "vocab": model.vocab,
        "seq_len": cfg.seq_len,
    }
    model_class = QuoVadisModel
    
    # Create model trainer
    model_instance = model_class(**model_config)
    fp_rates = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
    
    model_trainer = ModelTrainer(
        model=model_instance,
        device=cfg.device,
        loss_function=BCEWithLogitsLoss(),
        optimizer_class=AdamW,
        optimizer_config={"lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
        batchSize=cfg.batch_size,
        falsePositiveRates=fp_rates,
    )
    
    # Evaluate
    if cfg.split_mode in {"dir", "time_ood"}:
        # dir/time_ood 模式：使用预定义的划分
        from src.baselines._nebula_common import evaluate_holdout_with_predefined_indices
        summary = evaluate_holdout_with_predefined_indices(
            X, labels_arr, n_train, n_val, n_test, cfg, model_trainer, baseline_name
        )
    elif cfg.split_mode == "holdout":
        summary = evaluate_holdout(X, labels_arr, cfg, model_trainer, baseline_name)
    else:
        summary = evaluate_cv(X, labels_arr, cfg, model_class, model_config, baseline_name, use_dmds_trainer=False)
    
    # Log to W&B
    if "best" in summary:
        from src.baselines._nebula_common import log_comprehensive_wandb_summary
        test_idx = summary.get("_test_idx")
        log_comprehensive_wandb_summary(
            summary, labels_arr, cfg, baseline_name,
            model_trainer=model_trainer if cfg.split_mode == "holdout" else None,
            X=X if cfg.split_mode == "holdout" else None,
            test_idx=test_idx,
        )
        if WANDB_AVAILABLE and wandb and wandb.run:
            wandb.finish()
            LOGGER.info("实验结果已记录到 W&B")
    
    if cfg.save_raw:
        summary["raw_data"] = {
            "ids": ids,
            "labels": labels_arr.tolist(),
        }
    
    if cfg.split_mode in {"dir", "time_ood"}:
        if cfg.split_mode == "dir":
            summary["config_digest"] = {
                "train_dir": str(Path(cfg.train_dir).resolve()),
                "val_dir": str(Path(cfg.val_dir).resolve()),
                "test_dir": str(Path(cfg.test_dir).resolve()),
                "baseline": baseline_name,
            }
        else:  # time_ood
            summary["config_digest"] = {
                "train_end_date": getattr(cfg, 'train_end_date', '2025-04-23 08:08:46'),
                "val_end_date": getattr(cfg, 'val_end_date', '2025-06-14 11:39:39'),
                "malicious_manifest_path": str(Path(cfg.malicious_manifest_path).resolve()),
                "benign_manifest_path": str(Path(cfg.benign_manifest_path).resolve()),
                "baseline": baseline_name,
            }
    else:
        summary["config_digest"] = {
            "mal_dir": str(Path(cfg.mal_dir).resolve()),
            "benign_dir": str(Path(cfg.benign_dir).resolve()),
            "baseline": baseline_name,
            "limit_per_class": cfg.limit_per_class,
        }
    
    return summary


# --------------------------------------------------------------------------- #
# 命令行接口
# --------------------------------------------------------------------------- #


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="QuoVadis baseline experiments")
    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument("--mal_dir", type=str, required=True, help="Malicious JSON directory")
    parser.add_argument("--benign_dir", type=str, required=True, help="Benign JSON directory")
    parser.add_argument("--split_mode", type=str, default="holdout", choices=["holdout", "cv"])
    parser.add_argument("--limit_per_class", type=int, help="Limit samples per class")
    parser.add_argument("--out", type=str, help="Output JSON file")
    parser.add_argument("--progress", action="store_true", help="Show progress bars")
    
    args = parser.parse_args()
    
    # Load config
    config_dict = load_config_file(args.config) if args.config else {}
    
    # Create config
    cfg = TrainConfig(
        mal_dir=args.mal_dir or config_dict.get("mal_dir", ""),
        benign_dir=args.benign_dir or config_dict.get("benign_dir", ""),
        split_mode=args.split_mode or config_dict.get("split_mode", "holdout"),
        limit_per_class=args.limit_per_class or config_dict.get("limit_per_class"),
        out=args.out or config_dict.get("out"),
        progress=args.progress or config_dict.get("progress", False),
        **{k: v for k, v in config_dict.items() if k not in {
            "mal_dir", "benign_dir", "split_mode", "limit_per_class", "out", "progress"
        }},
    )
    
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
        LOGGER.info("Baseline: QuoVadis")
        LOGGER.info("Mode: %s", cfg.split_mode)
        LOGGER.info("AUC: %.4f", best.get("auroc", 0))
        LOGGER.info("F1: %.4f", best.get("macro_f1", 0))
        LOGGER.info("=" * 70)


if __name__ == "__main__":
    setup_logging()
    main()

