#!/usr/bin/env python3
"""
Family-OOD 逐层评估脚本 — MELD 实验阶段三 RQ3

对每一层 × 每一折（8家族留一），训练 LR → 校准 → 评估。

用法:
    python evaluate_family_ood_layers.py \
        --embedding_dir /data/rkodata/meld_alllayer_cache/qwen3 \
        --family_split_file /path/to/family_ood_spec.json \
        --output experiments/results/qwen3_family_ood_layer_profile.json

family_split_file 格式:
{
  "families": ["lummastealer", "agenttesla", ...],
  "folds": {
    "lummastealer": {
      "train_indices": [...],
      "val_indices": [...],
      "test_indices": [...]
    },
    ...
  }
}

输出:
{
  "model": "qwen3",
  "num_layers": 28,
  "families": ["lummastealer", ...],
  "results": {
    "lummastealer": {
      "layers": [
        {"layer": 1, "ood_macro_f1": ..., "ood_tpr_at_fpr_0.001": ..., ...},
        ...
      ],
      "best_layer_by_tpr001": 15,
    },
    ...
  },
  "stability": {
    "best_layers": [15, 16, 14, 15, 16, 15, 14, 15],
    "mean": 15.0,
    "std": 0.71,
    "min": 14,
    "max": 16
  }
}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("family_ood_layers")


def load_embeddings(embedding_dir: str, layer: int) -> np.ndarray:
    X = np.load(Path(embedding_dir) / f"layer_{layer:02d}.npy").astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def load_labels(embedding_dir: str) -> np.ndarray:
    return np.load(Path(embedding_dir) / "labels.npy")


def tpr_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid = fpr <= target_fpr
    if valid.any():
        return float(tpr[valid][-1])
    return 0.0


def evaluate_layer_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """单层单折评估。"""
    # 选 C
    best_c, best_val_f1 = 1.0, -1
    for c in (0.01, 0.1, 1.0):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=c, max_iter=2000, solver="lbfgs")),
        ])
        pipe.fit(X_train, y_train)
        val_f1 = f1_score(y_val, pipe.predict(X_val), average="macro")
        if val_f1 > best_val_f1:
            best_val_f1, best_c = val_f1, c

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=best_c, max_iter=2000, solver="lbfgs")),
    ])
    pipe.fit(X_train, y_train)

    # 验证集（域内）指标
    val_prob = pipe.predict_proba(X_val)[:, 1]
    val_pred = pipe.predict(X_val)
    id_f1 = float(f1_score(y_val, val_pred, average="macro"))

    # 测试集（域外）指标
    test_prob = pipe.predict_proba(X_test)[:, 1]
    test_pred = pipe.predict(X_test)

    ood_f1 = float(f1_score(y_test, test_pred, average="macro"))

    try:
        ood_auc = float(roc_auc_score(y_test, test_prob))
    except ValueError:
        ood_auc = 0.0

    return {
        "best_C": best_c,
        "id_macro_f1": id_f1,
        "ood_macro_f1": ood_f1,
        "f1_drop": round(id_f1 - ood_f1, 6),
        "ood_auc": ood_auc,
        "ood_tpr_at_fpr_0.001": tpr_at_fpr(y_test, test_prob, 0.001),
        "ood_tpr_at_fpr_0.005": tpr_at_fpr(y_test, test_prob, 0.005),
        "ood_tpr_at_fpr_0.01": tpr_at_fpr(y_test, test_prob, 0.01),
    }


def main():
    parser = argparse.ArgumentParser(description="Family-OOD 逐层评估")
    parser.add_argument("--embedding_dir", required=True)
    parser.add_argument("--family_split_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name", default="model")
    parser.add_argument("--layers", type=str, default=None,
                        help="指定层号（逗号分隔），默认全部层")
    args = parser.parse_args()

    meta_path = Path(args.embedding_dir) / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    num_layers = meta["num_layers"]

    labels = load_labels(args.embedding_dir)

    with open(args.family_split_file) as f:
        family_spec = json.load(f)

    # 适配两种格式：folds 为 dict 或 list
    raw_folds = family_spec["folds"]
    if isinstance(raw_folds, list):
        # 列表格式：每折有 held_out_family, train_ids, val_ids, test_ids
        families = [f["held_out_family"] for f in raw_folds]
        folds_map = {}
        for f in raw_folds:
            folds_map[f["held_out_family"]] = {
                "train_indices": f.get("train_ids", f.get("train_indices", [])),
                "val_indices": f.get("val_ids", f.get("val_indices", [])),
                "test_indices": f.get("test_ids", f.get("test_indices", [])),
            }
    else:
        # 字典格式
        families = family_spec.get("families", list(raw_folds.keys()))
        folds_map = raw_folds

    if args.layers:
        layer_list = [int(x) for x in args.layers.split(",")]
    else:
        layer_list = list(range(1, num_layers + 1))

    LOGGER.info(f"模型: {args.model_name}, 层数: {len(layer_list)}, 家族数: {len(families)}")

    results = {
        "model": args.model_name,
        "num_layers": num_layers,
        "evaluated_layers": layer_list,
        "families": families,
        "results": {},
    }

    for family in families:
        LOGGER.info(f"=== 家族: {family} ===")
        fold = folds_map[family]
        train_idx = np.array(fold["train_indices"])
        val_idx = np.array(fold["val_indices"])
        test_idx = np.array(fold["test_indices"])

        y_train = labels[train_idx]
        y_val = labels[val_idx]
        y_test = labels[test_idx]

        LOGGER.info(f"  训练: {len(y_train)} (恶意{y_train.sum()}), "
                     f"验证: {len(y_val)} (恶意{y_val.sum()}), "
                     f"测试: {len(y_test)} (恶意{y_test.sum()})")

        family_results = {"layers": []}

        for layer in layer_list:
            X = load_embeddings(args.embedding_dir, layer)
            X_train = X[train_idx]
            X_val = X[val_idx]
            X_test = X[test_idx]

            layer_result = evaluate_layer_fold(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            layer_result["layer"] = layer
            family_results["layers"].append(layer_result)

        # 该折最优层
        best = max(family_results["layers"], key=lambda x: x["ood_tpr_at_fpr_0.001"])
        family_results["best_layer_by_tpr001"] = best["layer"]
        family_results["best_ood_tpr001"] = best["ood_tpr_at_fpr_0.001"]

        best_f1 = max(family_results["layers"], key=lambda x: x["ood_macro_f1"])
        family_results["best_layer_by_f1"] = best_f1["layer"]

        LOGGER.info(f"  最优层(TPR@0.1%): 层{best['layer']}, TPR={best['ood_tpr_at_fpr_0.001']:.4f}")

        results["results"][family] = family_results

    # 层选择稳定性分析
    best_layers = [results["results"][f]["best_layer_by_tpr001"] for f in families]
    results["stability"] = {
        "best_layers": best_layers,
        "mean": round(float(np.mean(best_layers)), 2),
        "std": round(float(np.std(best_layers)), 2),
        "min": int(np.min(best_layers)),
        "max": int(np.max(best_layers)),
    }
    LOGGER.info(f"层选择稳定性: {best_layers}, 均值={results['stability']['mean']}, "
                f"标准差={results['stability']['std']}")

    # 保存
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
