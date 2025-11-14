"""Shared utilities for Nebula baseline models.

This module contains common functions used by neurlux, quovadis, and dmds baselines.
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm

# W&B integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Nebula imports
ROOT_DIR = Path(__file__).resolve().parents[2]
NEBULA_REPO = ROOT_DIR / "third_party" / "nebula"
if not NEBULA_REPO.exists():
    raise RuntimeError(
        f"未找到 {NEBULA_REPO}。请先克隆 upstream 仓库："
        " git clone https://github.com/dtrizna/nebula third_party/nebula"
    )

if str(NEBULA_REPO) not in sys.path:
    sys.path.insert(0, str(NEBULA_REPO))

try:
    from nebula import ModelTrainer
    from torch.nn import BCEWithLogitsLoss
    from torch.optim import AdamW
except ImportError as e:
    raise RuntimeError(f"无法导入 nebula 模块: {e}") from e

from src.baselines.cape_adapter import cape_json_to_text

LOGGER = logging.getLogger("nebula_common")


# --------------------------------------------------------------------------- #
# DMDS ModelTrainer 包装器
# --------------------------------------------------------------------------- #


class DMDSModelTrainer(ModelTrainer):
    """ModelTrainer wrapper for DMDS that handles float32 input."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override forwardPass to handle float32
        original_forward = self.model.forward
        def dmds_forward_pass(x):
            if x.dtype == torch.long:
                x = x.float()
            return original_forward(x)
        self.model.forwardPass = dmds_forward_pass
    
    def fit(self, X, y, epochs=10, **kwargs):
        """Override fit to use float32 for DMDS."""
        from torch.utils.data import DataLoader, TensorDataset
        import torch
        import time
        from sklearn.metrics import roc_auc_score, f1_score
        from tqdm import tqdm
        
        # Convert to float32 for DMDS
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        
        trainLoader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        self.epochs = epochs
        self.trainset_size = len(trainLoader)
        self.model.train()
        self.train_tprs = np.empty(shape=(self.epochs, len(self.fp_rates)))
        self.train_f1s = np.empty(shape=(self.epochs, len(self.fp_rates)))
        self.train_losses = []
        self.train_auc = []
        
        # Log training metrics to W&B if available
        log_to_wandb = WANDB_AVAILABLE and wandb and wandb.run
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            epoch_losses, epoch_tprs, epoch_f1s, epoch_auc = self._train_epoch(trainLoader, y)
            epoch_time = time.time() - epoch_start_time
            
            self.train_losses.extend(epoch_losses)
            self.train_tprs[epoch] = epoch_tprs
            self.train_f1s[epoch] = epoch_f1s
            self.train_auc.append(epoch_auc)
            
            # Log epoch metrics to W&B
            if log_to_wandb:
                try:
                    avg_loss = np.mean(epoch_losses)
                    target_fpr_idx = min(range(len(self.fp_rates)), key=lambda i: abs(self.fp_rates[i] - 0.01))
                    wandb.log({
                        "train/epoch": epoch + 1,
                        "train/loss": avg_loss,
                        "train/auc": epoch_auc,
                        "train/f1": epoch_f1s[target_fpr_idx],
                        "train/tpr": epoch_tprs[target_fpr_idx],
                        "train/epoch_time": epoch_time,
                    })
                except Exception:
                    # W&B run 可能已结束，静默忽略
                    pass
        
        return self.train_losses, self.train_tprs, self.train_f1s, self.train_auc
    
    def _train_epoch(self, trainLoader, y_true):
        """Train one epoch."""
        from torch.nn.utils import clip_grad_norm_
        import time
        
        self.tick = time.time()
        epoch_losses = []
        epochProbs = []
        targets = []
        
        for batch_idx, (data, target) in enumerate(trainLoader):
            targets.extend(target.cpu().numpy())
            data, target = data.to(self.device), target.to(self.device).float()
            
            self.optimizer.zero_grad()
            logits = self.model.forwardPass(data).float()
            
            if isinstance(self.loss_function, BCEWithLogitsLoss) and target.dim() == 1:
                target = target.reshape(-1, 1)
            
            loss = self.loss_function(logits, target)
            loss.backward()
            epoch_losses.append(loss.item())
            
            if self.clip_grad_norm is not None:
                clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            
            self.optimizer.step()
            
            predProbs = torch.sigmoid(logits).clone().detach().cpu().numpy()
            epochProbs.extend(predProbs)
        
        # Calculate metrics
        from sklearn.metrics import roc_auc_score
        epoch_auc = roc_auc_score(targets, epochProbs)
        epochMetrics = self.getMetrics(np.array(targets), np.array(epochProbs))
        epoch_tprs, epoch_f1s = epochMetrics[:, 0], epochMetrics[:, 1]
        
        return epoch_losses, epoch_tprs, epoch_f1s, epoch_auc
    
    def evaluate(self, X, y, metrics="array"):
        """Override evaluate to use float32 for DMDS."""
        from torch.utils.data import DataLoader, TensorDataset
        import torch
        from tqdm import tqdm
        
        # Convert to float32
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        
        testLoader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        self.testLoss = []
        self.predProbs = []
        self.trueLabels = []
        
        for data, target in tqdm(testLoader):
            self.trueLabels.extend(target.cpu().numpy())
            data, target = data.to(self.device), target.to(self.device)
            
            with torch.no_grad():
                logits = self.model.forwardPass(data)
            
            loss = self.loss_function(logits, target.float().reshape(-1, 1))
            self.testLoss.append(loss.item())
            
            predProbs = torch.sigmoid(logits).clone().detach().cpu().numpy()
            self.predProbs.extend(predProbs)
        
        from sklearn.metrics import roc_auc_score
        from nebula.misc import get_tpr_at_fpr
        
        metricsValues = self.getMetrics(np.array(self.trueLabels), np.array(self.predProbs))
        self.test_tprs, self.test_f1s = metricsValues[:, 0], metricsValues[:, 1]
        self.test_auc = roc_auc_score(self.trueLabels, self.predProbs)
        
        if metrics == "array":
            return self.testLoss, self.test_tprs, self.test_f1s, self.test_auc
        elif metrics == "json":
            return self.metricsToJSON([self.testLoss, self.test_tprs, self.test_f1s, self.test_auc])
        else:
            raise ValueError("evaluate(): Inappropriate metrics value")
    
    def predict_proba(self, arr):
        """Override predict_proba to use float32 for DMDS."""
        from torch.utils.data import DataLoader, TensorDataset
        import torch
        
        out = torch.empty((0, self.n_output_classes)).to(self.device)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(arr).float()),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        for batch_idx, data in enumerate(loader):
            with torch.no_grad():
                logits = self.model.forwardPass(data[0].to(self.device))
                out = torch.vstack([out, logits])
        
        if self.n_output_classes == 1:
            return torch.sigmoid(out).clone().detach().cpu().numpy().flatten()
        else:
            return torch.softmax(out, axis=1).clone().detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #


def list_json_files(directory: str) -> List[Path]:
    """List all JSON files in directory."""
    dir_path = Path(directory)
    return sorted(dir_path.glob("*.json"))


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def load_cape_reports(
    mal_dir: str,
    benign_dir: str,
    limit_per_class: Optional[int] = None,
    progress: bool = False,
    n_jobs: int = 1,
) -> Tuple[List[Path], List[int], List[str]]:
    """Load CAPE JSON reports with labels.
    
    Args:
        mal_dir: Malicious JSON directory
        benign_dir: Benign JSON directory
        limit_per_class: Limit samples per class
        progress: Show progress bar
        n_jobs: Number of parallel jobs for file listing (-1 for all CPUs)
    
    Returns:
        Tuple of (paths, labels, ids)
    """
    import time
    start_time = time.time()
    
    # Parallel file listing if n_jobs > 1
    if n_jobs > 1:
        try:
            from src.utils.performance import parallel_map
            mal_paths = parallel_map(
                lambda _: list_json_files(mal_dir),
                [None],
                n_jobs=1,  # File listing is fast, no need to parallelize
                progress=False,
            )[0]
            ben_paths = parallel_map(
                lambda _: list_json_files(benign_dir),
                [None],
                n_jobs=1,
                progress=False,
            )[0]
        except ImportError:
            mal_paths = list_json_files(mal_dir)
            ben_paths = list_json_files(benign_dir)
    else:
        mal_paths = list_json_files(mal_dir)
        ben_paths = list_json_files(benign_dir)
    
    if limit_per_class:
        if len(mal_paths) > limit_per_class:
            mal_paths = mal_paths[:limit_per_class]
        if len(ben_paths) > limit_per_class:
            ben_paths = ben_paths[:limit_per_class]
    
    all_paths = [(p, 1) for p in mal_paths] + [(p, 0) for p in ben_paths]
    
    paths = []
    labels = []
    ids = []
    
    iterator = all_paths
    if progress:
        iterator = tqdm(iterator, desc="Loading reports", total=len(all_paths))
    
    for path, label in iterator:
        paths.append(path)
        labels.append(label)
        ids.append(path.stem)
    
    elapsed = time.time() - start_time
    LOGGER.debug(f"加载了 {len(paths)} 个报告，耗时 {elapsed:.2f}s")
    
    return paths, labels, ids


def load_cape_reports_from_dir(
    data_dir: str,
    progress: bool = False,
) -> Tuple[List[Path], List[int], List[str]]:
    """Load CAPE JSON reports from a directory with black/white subdirectories.
    
    Args:
        data_dir: Directory containing black/ and white/ subdirectories
        progress: Show progress bar
    
    Returns:
        Tuple of (paths, labels, ids)
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"目录不存在: {data_dir}")
    
    # Try black/white structure
    mal_dir = data_path / "black"
    ben_dir = data_path / "white"
    
    # Try alternative names
    if not mal_dir.exists():
        mal_dir = data_path / "cape_reports_malicious"
    if not ben_dir.exists():
        ben_dir = data_path / "cape_reports_benign"
    
    mal_paths = list_json_files(str(mal_dir)) if mal_dir.exists() else []
    ben_paths = list_json_files(str(ben_dir)) if ben_dir.exists() else []
    
    all_paths = [(p, 1) for p in mal_paths] + [(p, 0) for p in ben_paths]
    
    paths = []
    labels = []
    ids = []
    
    iterator = all_paths
    if progress:
        iterator = tqdm(iterator, desc=f"Loading from {data_dir}", total=len(all_paths))
    
    for path, label in iterator:
        paths.append(path)
        labels.append(label)
        ids.append(path.stem)
    
    LOGGER.info(f"从 {data_dir} 加载了 {len(paths)} 个报告 (恶意: {len(mal_paths)}, 良性: {len(ben_paths)})")
    
    return paths, labels, ids


# --------------------------------------------------------------------------- #
# 评估函数
# --------------------------------------------------------------------------- #


def quantile_threshold(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> Dict[str, float]:
    """Calculate threshold at target FPR."""
    mask_neg = y_true == 0
    mask_pos = y_true == 1
    if mask_neg.sum() == 0 or mask_pos.sum() == 0:
        raise ValueError("正负样本数量不足，无法计算阈值。")
    neg_scores = y_score[mask_neg]
    tau = float(np.quantile(neg_scores, 1 - target_fpr))
    preds = (y_score >= tau).astype(int)
    fpr = float(((preds == 1) & mask_neg).sum() / mask_neg.sum())
    tpr = float(((preds == 1) & mask_pos).sum() / mask_pos.sum())
    mu_neg, std_neg = float(neg_scores.mean()), float(neg_scores.std(ddof=1) + 1e-8)
    mu_all, std_all = float(y_score.mean()), float(y_score.std(ddof=1) + 1e-8)
    z_neg = (tau - mu_neg) / std_neg
    z_all = (tau - mu_all) / std_all
    return {"tau": tau, "fpr": fpr, "tpr": tpr, "z_neg": float(z_neg), "z_all": float(z_all)}


def stratified_holdout_indices(
    y: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train/val/test."""
    idx = np.arange(len(y))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(idx, y))
    val_rel = val_ratio / (1 - test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed + 1)
    train_idx_rel, val_idx_rel = next(sss2.split(train_val_idx, y[train_val_idx]))
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]
    return train_idx, val_idx, test_idx


def evaluate_holdout_with_predefined_indices(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    cfg,
    model_trainer: ModelTrainer,
    baseline_name: str,
) -> Dict:
    """Evaluate with predefined train/val/test split (dir mode)."""
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)
    
    return _evaluate_with_indices(X, y, train_idx, val_idx, test_idx, cfg, model_trainer, baseline_name)


def _evaluate_with_indices(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg,
    model_trainer: ModelTrainer,
    baseline_name: str,
) -> Dict:
    """Internal function to evaluate with given indices."""
    # Train
    model_trainer.fit(
        X[train_idx],
        y[train_idx],
        epochs=cfg.epochs,
    )
    
    # Evaluate on validation set
    val_loss, val_tprs, val_f1s, val_auc = model_trainer.evaluate(X[val_idx], y[val_idx])
    
    # Find threshold at target FPR
    val_probs = model_trainer.predict_proba(X[val_idx])
    # Ensure val_probs is 1D (force flatten if needed)
    if val_probs.ndim > 1:
        val_probs = val_probs.squeeze()
    if val_probs.ndim > 1:
        val_probs = val_probs.flatten()
    val_probs = np.asarray(val_probs).ravel()
    id_stats = quantile_threshold(y[val_idx], val_probs, cfg.target_fpr)
    
    # Evaluate on test set
    test_loss, test_tprs, test_f1s, test_auc = model_trainer.evaluate(X[test_idx], y[test_idx])
    test_probs = model_trainer.predict_proba(X[test_idx])
    # Ensure test_probs is 1D (force flatten if needed)
    if test_probs.ndim > 1:
        test_probs = test_probs.squeeze()
    if test_probs.ndim > 1:
        test_probs = test_probs.flatten()
    test_probs = np.asarray(test_probs).ravel()
    
    # Apply threshold
    preds = (test_probs >= id_stats["tau"]).astype(int)
    macro_f1 = float(f1_score(y[test_idx], preds, average="macro"))
    auroc = float(roc_auc_score(y[test_idx], test_probs))
    aupr = float(average_precision_score(y[test_idx], test_probs))
    accuracy = float(accuracy_score(y[test_idx], preds))
    
    # Find target FPR index
    fp_rates = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
    target_idx = min(range(len(fp_rates)), key=lambda i: abs(fp_rates[i] - cfg.target_fpr))
    
    return {
        "mode": "holdout",
        "baseline": baseline_name,
        "target_fpr": cfg.target_fpr,
        "best": {
            "threshold": id_stats["tau"],
            "tpr_id": float(val_tprs[target_idx]),
            "fpr_id": id_stats["fpr"],
            "f1_id": float(val_f1s[target_idx]),
            "auc_id": float(val_auc),
            "tpr_ood": float(test_tprs[target_idx]),
            "fpr_ood": id_stats["fpr"],
            "f1_ood": float(test_f1s[target_idx]),
            "auc_ood": float(test_auc),
            "macro_f1": macro_f1,
            "auroc": auroc,
            "aupr": aupr,
            "accuracy": accuracy,
        },
        "_test_idx": test_idx,  # Store for visualization
    }


def evaluate_cv(
    X: np.ndarray,
    y: np.ndarray,
    cfg,
    model_class,
    model_config: Dict,
    baseline_name: str,
    use_dmds_trainer: bool = False,
) -> Dict:
    """Evaluate with cross-validation."""
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    
    all_probs = np.zeros_like(y, dtype=float)
    fp_rates = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
    target_idx = min(range(len(fp_rates)), key=lambda i: abs(fp_rates[i] - cfg.target_fpr))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Create model trainer for this fold
        model = model_class(**model_config)
        
        # DMDS needs special handling for float32 input
        if use_dmds_trainer:
            model_trainer = DMDSModelTrainer(
                model=model,
                device=cfg.device,
                loss_function=BCEWithLogitsLoss(),
                optimizer_class=AdamW,
                optimizer_config={"lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
                batchSize=cfg.batch_size,
                falsePositiveRates=fp_rates,
            )
        else:
            model_trainer = ModelTrainer(
                model=model,
                device=cfg.device,
                loss_function=BCEWithLogitsLoss(),
                optimizer_class=AdamW,
                optimizer_config={"lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
                batchSize=cfg.batch_size,
                falsePositiveRates=fp_rates,
            )
        
        # Train with W&B logging for each fold
        # Log fold info to W&B if available
        log_to_wandb = WANDB_AVAILABLE and wandb and wandb.run
        if log_to_wandb:
            try:
                wandb.log({"cv/fold": fold + 1, "cv/total_folds": cfg.n_splits})
            except Exception:
                # W&B run 可能已结束，静默忽略
                pass
        
        model_trainer.fit(X[train_idx], y[train_idx], epochs=cfg.epochs)
        
        # Log fold training metrics if ModelTrainer doesn't have W&B logging
        # (for non-DMDS trainers)
        if log_to_wandb and not use_dmds_trainer:
            # Try to extract training metrics from ModelTrainer
            if hasattr(model_trainer, 'train_losses') and hasattr(model_trainer, 'auc'):
                if model_trainer.train_losses and model_trainer.auc:
                    # Log last epoch metrics
                    last_epoch_loss = np.mean(model_trainer.train_losses[-len(X[train_idx])//cfg.batch_size:]) if model_trainer.train_losses else 0
                    last_auc = model_trainer.auc[-1] if model_trainer.auc else 0
                    try:
                        wandb.log({
                            f"cv/fold_{fold+1}/train/loss": last_epoch_loss,
                            f"cv/fold_{fold+1}/train/auc": last_auc,
                        })
                    except Exception:
                        # W&B run 可能已结束，静默忽略
                        pass
        
        # Predict
        val_probs = model_trainer.predict_proba(X[val_idx])
        # Ensure val_probs is 1D (force flatten if needed)
        if val_probs.ndim > 1:
            val_probs = val_probs.squeeze()
        # Additional safety: ensure it's truly 1D
        if val_probs.ndim > 1:
            val_probs = val_probs.flatten()
        # Convert to numpy array if needed and ensure 1D
        val_probs = np.asarray(val_probs).ravel()
        all_probs[val_idx] = val_probs
        
        # Log fold validation metrics
        if log_to_wandb:
            val_labels = y[val_idx]
            val_auc = float(roc_auc_score(val_labels, val_probs))
            val_ap = float(average_precision_score(val_labels, val_probs))
            try:
                wandb.log({
                    f"cv/fold_{fold+1}/val/auc": val_auc,
                    f"cv/fold_{fold+1}/val/ap": val_ap,
                })
            except Exception:
                # W&B run 可能已结束，静默忽略
                pass
    
    # Calculate metrics
    id_stats = quantile_threshold(y, all_probs, cfg.target_fpr)
    preds = (all_probs >= id_stats["tau"]).astype(int)
    macro_f1 = float(f1_score(y, preds, average="macro"))
    auroc = float(roc_auc_score(y, all_probs))
    aupr = float(average_precision_score(y, all_probs))
    accuracy = float(accuracy_score(y, preds))
    
    return {
        "mode": "cv",
        "baseline": baseline_name,
        "target_fpr": cfg.target_fpr,
        "best": {
            "threshold": id_stats["tau"],
            "tpr": id_stats["tpr"],
            "fpr": id_stats["fpr"],
            "macro_f1": macro_f1,
            "auroc": auroc,
            "aupr": aupr,
            "accuracy": accuracy,
        },
    }


# --------------------------------------------------------------------------- #
# W&B 集成
# --------------------------------------------------------------------------- #


def init_wandb(cfg, baseline_name: str) -> Optional[str]:
    """Initialize W&B run."""
    if not cfg.use_wandb or not WANDB_AVAILABLE:
        return None
    
    if cfg.wandb_api_key:
        wandb.login(key=cfg.wandb_api_key)
    
    # 结束之前的 run（如果存在），避免并行执行时的 run 名称混乱
    if wandb.run is not None:
        wandb.finish()
    
    run_name = f"{baseline_name}_{cfg.split_mode}"
    if cfg.run_tag:
        run_name += f"_{cfg.run_tag}"
    
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=run_name,
        reinit=True,  # 允许重新初始化，确保每个基线都有独立的 run
        config={
            "baseline": baseline_name,
            "split_mode": cfg.split_mode,
            "limit_per_class": cfg.limit_per_class,
            "target_fpr": cfg.target_fpr,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
        },
    )
    return wandb.run.id if wandb.run else None


def log_metrics_to_wandb(metrics: Dict, prefix: str = "") -> None:
    """Log metrics to W&B with consistent formatting."""
    if not WANDB_AVAILABLE:
        return
    
    # 检查 W&B run 是否活跃
    if not wandb.run:
        return
    
    try:
        # 检查 run 是否已结束
        if hasattr(wandb.run, 'settings') and wandb.run.settings._run_id is None:
            return
        
        wandb_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # Use consistent format: prefix/key for hierarchical structure
                if prefix:
                    wandb_key = f"{prefix}/{key}" if "/" not in prefix else f"{prefix}_{key}"
                else:
                    wandb_key = key
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


def log_comprehensive_wandb_summary(
    summary: Dict,
    labels_arr: np.ndarray,
    cfg,
    baseline_name: str,
    model_trainer=None,
    X=None,
    test_idx=None,
) -> None:
    """Comprehensive W&B logging with metrics, dataset stats, and visualizations."""
    if not WANDB_AVAILABLE or not wandb or not wandb.run:
        return
    
    if "best" not in summary:
        return
    
    # Log all metrics
    log_metrics_to_wandb(summary["best"], prefix=cfg.split_mode)
    
    # Dataset statistics
    n_samples = len(labels_arr)
    n_malicious = int(labels_arr.sum())
    n_benign = n_samples - n_malicious
    try:
        wandb.log({
            "dataset/n_samples": n_samples,
            "dataset/n_malicious": n_malicious,
            "dataset/n_benign": n_benign,
            "dataset/malicious_ratio": n_malicious / n_samples if n_samples > 0 else 0,
        })
    except Exception:
        # W&B run 可能已结束，静默忽略
        pass
    
    # Update summary with all metrics
    summary_metrics = {}
    for key, value in summary["best"].items():
        if isinstance(value, (int, float)):
            summary_metrics[f"best_{key}"] = value
    wandb.run.summary.update(summary_metrics)
    
    # Log visualizations if model_trainer and X are available
    if model_trainer is not None and X is not None:
        try:
            from src.utils.wandb_viz import (
                log_roc_curve,
                log_pr_curve,
                log_confusion_matrix,
                log_score_distribution,
            )
            
            # Get predictions for visualization
            if cfg.split_mode == "holdout" and test_idx is not None:
                test_probs = model_trainer.predict_proba(X[test_idx])
                test_labels = labels_arr[test_idx]
            else:
                # For CV, use all data
                test_probs = model_trainer.predict_proba(X)
                test_labels = labels_arr
            
            # Ensure test_probs is 1D (force flatten if needed)
            if test_probs.ndim > 1:
                test_probs = test_probs.squeeze()
            if test_probs.ndim > 1:
                test_probs = test_probs.flatten()
            test_probs = np.asarray(test_probs).ravel()
            
            threshold = summary["best"].get("threshold", 0.5)
            y_pred = (test_probs >= threshold).astype(int)
            
            log_roc_curve(test_labels, test_probs, title=f"{baseline_name.upper()} - ROC Curve")
            log_pr_curve(test_labels, test_probs, title=f"{baseline_name.upper()} - PR Curve")
            log_confusion_matrix(test_labels, y_pred, title=f"{baseline_name.upper()} - Confusion Matrix")
            log_score_distribution(test_labels, test_probs, threshold, title=f"{baseline_name.upper()} - Score Distribution")
        except Exception as exc:
            LOGGER.warning(f"可视化记录失败: {exc}")


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #


def load_config_file(path: Optional[str]) -> Dict:
    """Load configuration from JSON file."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 JSON 对象，当前: {type(data)}")
    return data


def aggregate_seed_summaries(cfg, summaries: List[Dict], baseline_name: str) -> Dict:
    """Aggregate results across multiple seeds."""
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
        "baseline": baseline_name,
        "seeds": seeds,
        "per_seed": summaries,
        "best_mean": best_mean,
        "best_std": best_std,
        "config": {k: v for k, v in asdict(cfg).items() if k not in {"config_path", "seed"}},
    }
    return aggregated


def save_summary(summary: Dict, path: Optional[str]) -> None:
    """Save summary to JSON file."""
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    LOGGER.info("结果已保存到: %s", out_path)


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

