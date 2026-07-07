#!/usr/bin/env python3
"""
逐层评估脚本 — MELD 实验阶段三 RQ1

对每一层独立训练 LR 分类器，在验证集校准阈值，在测试集报告性能。

用法:
    python evaluate_all_layers.py \
        --embedding_dir /path/to/meld_alllayer_cache/qwen3 \
        --split_file /path/to/time_ood_split.json \
        --output experiments/results/qwen3_layer_profile.json

输出 JSON 结构:
{
  "model": "qwen3",
  "num_layers": 28,
  "layers": [
    {
      "layer": 1,
      "time_ood": {
        "macro_f1": 0.95,
        "tpr_at_fpr_0.001": 0.85,
        "tpr_at_fpr_0.005": 0.90,
        "tpr_at_fpr_0.01": 0.92,
        "auc": 0.98
      }
    },
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("evaluate_all_layers")


def load_embeddings(embedding_dir: str, layer: int) -> np.ndarray:
    """加载指定层的嵌入，转为 float32 并处理 inf/nan。"""
    path = Path(embedding_dir) / f"layer_{layer:02d}.npy"
    X = np.load(path).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def load_labels(embedding_dir: str) -> np.ndarray:
    return np.load(Path(embedding_dir) / "labels.npy")


def load_split(split_file: str) -> Dict[str, np.ndarray]:
    """加载划分文件，返回 train/val/test 的索引数组。"""
    with open(split_file) as f:
        split = json.load(f)
    return {
        "train": np.array(split["train_indices"]),
        "val": np.array(split["val_indices"]),
        "test": np.array(split["test_indices"]),
    }


def tpr_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float) -> float:
    """计算指定 FPR 下的 TPR。"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    # 找到不超过 target_fpr 的最大 TPR
    valid = fpr <= target_fpr
    if valid.any():
        return float(tpr[valid][-1])
    return 0.0


def find_threshold_at_fpr(
    y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float
) -> float:
    """找到使 FPR <= target_fpr 的阈值。"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid = fpr <= target_fpr
    if valid.any():
        idx = np.where(valid)[0][-1]
        return float(thresholds[idx])
    return 1.0


def evaluate_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C_values: Tuple[float, ...] = (0.01, 0.1, 1.0),
) -> Dict:
    """训练LR，验证集选C和校准阈值，测试集报告。"""

    # 在验证集上选最优 C
    best_c = C_values[0]
    best_val_f1 = -1
    for c in C_values:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=c, max_iter=2000, solver="lbfgs")),
        ])
        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_c = c

    # 用最优 C 重新训练（train + val 合并可选，这里严格只用 train）
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=best_c, max_iter=2000, solver="lbfgs")),
    ])
    pipe.fit(X_train, y_train)

    # 验证集校准阈值
    val_prob = pipe.predict_proba(X_val)[:, 1]

    # 测试集评估
    test_prob = pipe.predict_proba(X_test)[:, 1]
    test_pred = pipe.predict(X_test)

    results = {
        "best_C": best_c,
        "val_macro_f1": float(best_val_f1),
        "macro_f1": float(f1_score(y_test, test_pred, average="macro")),
        "auc": float(roc_auc_score(y_test, test_prob)),
        "tpr_at_fpr_0.001": tpr_at_fpr(y_test, test_prob, 0.001),
        "tpr_at_fpr_0.005": tpr_at_fpr(y_test, test_prob, 0.005),
        "tpr_at_fpr_0.01": tpr_at_fpr(y_test, test_prob, 0.01),
        "threshold_at_fpr_0.001": find_threshold_at_fpr(y_val, val_prob, 0.001),
        "threshold_at_fpr_0.005": find_threshold_at_fpr(y_val, val_prob, 0.005),
        "threshold_at_fpr_0.01": find_threshold_at_fpr(y_val, val_prob, 0.01),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="逐层性能评估")
    parser.add_argument("--embedding_dir", required=True, help="全层嵌入目录")
    parser.add_argument("--split_file", required=True, help="划分文件 JSON")
    parser.add_argument("--output", required=True, help="输出结果 JSON")
    parser.add_argument("--model_name", default="model")
    args = parser.parse_args()

    # 加载元数据
    meta_path = Path(args.embedding_dir) / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    num_layers = meta["num_layers"]

    # 加载标签和划分
    labels = load_labels(args.embedding_dir)
    split = load_split(args.split_file)

    y_train = labels[split["train"]]
    y_val = labels[split["val"]]
    y_test = labels[split["test"]]

    LOGGER.info(f"模型: {args.model_name}, 层数: {num_layers}")
    LOGGER.info(f"训练: {len(y_train)}, 验证: {len(y_val)}, 测试: {len(y_test)}")
    LOGGER.info(f"恶意比例 — 训练: {y_train.mean():.3f}, 验证: {y_val.mean():.3f}, 测试: {y_test.mean():.3f}")

    results = {
        "model": args.model_name,
        "num_layers": num_layers,
        "split_file": args.split_file,
        "layers": [],
    }

    for layer in range(1, num_layers + 1):
        LOGGER.info(f"评估层 {layer}/{num_layers}...")

        X = load_embeddings(args.embedding_dir, layer)
        X_train = X[split["train"]]
        X_val = X[split["val"]]
        X_test = X[split["test"]]

        layer_result = evaluate_layer(X_train, y_train, X_val, y_val, X_test, y_test)
        layer_result["layer"] = layer
        results["layers"].append(layer_result)

        LOGGER.info(
            f"  层{layer}: F1={layer_result['macro_f1']:.4f}, "
            f"TPR@0.1%={layer_result['tpr_at_fpr_0.001']:.4f}, "
            f"AUC={layer_result['auc']:.4f}"
        )

    # 找最优层
    best = max(results["layers"], key=lambda x: x["tpr_at_fpr_0.001"])
    results["best_layer_by_tpr001"] = best["layer"]
    results["best_tpr_at_fpr_0.001"] = best["tpr_at_fpr_0.001"]
    results["best_macro_f1"] = best["macro_f1"]

    best_f1 = max(results["layers"], key=lambda x: x["macro_f1"])
    results["best_layer_by_f1"] = best_f1["layer"]

    LOGGER.info(f"最优层(TPR@0.1%FPR): 层{best['layer']}, TPR={best['tpr_at_fpr_0.001']:.4f}")
    LOGGER.info(f"最优层(Macro-F1): 层{best_f1['layer']}, F1={best_f1['macro_f1']:.4f}")

    # 保存
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
