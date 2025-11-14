"""Common W&B logging utilities for all baselines."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    from src.utils.wandb_viz import (
        log_roc_curve,
        log_pr_curve,
        log_confusion_matrix,
        log_score_distribution,
    )
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False


def log_comprehensive_wandb_summary_simple(
    summary: Dict,
    labels_arr: np.ndarray,
    cfg,
    baseline_name: str,
    y_true: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
) -> None:
    """Comprehensive W&B logging for baselines without model_trainer."""
    if not WANDB_AVAILABLE or not wandb or not wandb.run:
        return
    
    if "best" not in summary:
        return
    
    # Log all metrics
    from src.baselines._nebula_common import log_metrics_to_wandb
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
    
    # Log visualizations if predictions are available
    if y_true is not None and y_prob is not None and VIZ_AVAILABLE:
        try:
            threshold = summary["best"].get("threshold", 0.5)
            if y_pred is None:
                y_pred = (y_prob >= threshold).astype(int)
            
            log_roc_curve(y_true, y_prob, title=f"{baseline_name.upper()} - ROC Curve")
            log_pr_curve(y_true, y_prob, title=f"{baseline_name.upper()} - PR Curve")
            log_confusion_matrix(y_true, y_pred, title=f"{baseline_name.upper()} - Confusion Matrix")
            log_score_distribution(y_true, y_prob, threshold, title=f"{baseline_name.upper()} - Score Distribution")
        except Exception as exc:
            import logging
            LOGGER = logging.getLogger(__name__)
            LOGGER.warning(f"可视化记录失败: {exc}")

