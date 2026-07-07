#!/usr/bin/env python3
"""
多分类器对比 — 在指定层的嵌入上训练 LR/RF/SVM/XGBoost 并评估。

用法:
    python evaluate_classifiers.py \
        --embedding_dir /data/rkodata/meld_alllayer_cache/qwen3 \
        --split_file /path/to/time_ood_split.json \
        --layer 7 \
        --output results_v2/supplementary/p2_classifiers/classifier_comparison.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("classifiers")


def tpr_at_fpr(y_true, y_prob, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid = fpr <= target_fpr
    return float(tpr[valid][-1]) if valid.any() else 0.0


def evaluate_classifier(name, pipe, X_train, y_train, X_val, y_val, X_test, y_test):
    LOGGER.info(f"训练 {name}...")
    start = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    # 预测概率
    if hasattr(pipe, "predict_proba"):
        test_prob = pipe.predict_proba(X_test)[:, 1]
        val_prob = pipe.predict_proba(X_val)[:, 1]
    elif hasattr(pipe, "decision_function"):
        test_prob = pipe.decision_function(X_test)
        val_prob = pipe.decision_function(X_val)
    else:
        test_prob = pipe.predict(X_test).astype(float)
        val_prob = pipe.predict(X_val).astype(float)

    test_pred = (test_prob > 0.5).astype(int) if test_prob.max() <= 1 else pipe.predict(X_test)

    result = {
        "classifier": name,
        "train_time_sec": round(train_time, 3),
        "macro_f1": float(f1_score(y_test, test_pred, average="macro")),
        "auc": float(roc_auc_score(y_test, test_prob)),
        "tpr_at_fpr_0.001": tpr_at_fpr(y_test, test_prob, 0.001),
        "tpr_at_fpr_0.005": tpr_at_fpr(y_test, test_prob, 0.005),
        "tpr_at_fpr_0.01": tpr_at_fpr(y_test, test_prob, 0.01),
    }
    LOGGER.info(f"  {name}: F1={result['macro_f1']:.4f}, TPR@0.1%={result['tpr_at_fpr_0.001']:.4f}, time={train_time:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # 加载数据
    X = np.load(Path(args.embedding_dir) / f"layer_{args.layer:02d}.npy").astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.load(Path(args.embedding_dir) / "labels.npy")

    with open(args.split_file) as f:
        split = json.load(f)

    train_idx = np.array(split["train_indices"])
    val_idx = np.array(split["val_indices"])
    test_idx = np.array(split["test_indices"])

    X_train, y_train = X[train_idx], labels[train_idx]
    X_val, y_val = X[val_idx], labels[val_idx]
    X_test, y_test = X[test_idx], labels[test_idx]

    LOGGER.info(f"层 {args.layer}, 特征维度: {X_train.shape[1]}")
    LOGGER.info(f"训练: {len(y_train)}, 验证: {len(y_val)}, 测试: {len(y_test)}")

    # 定义分类器
    classifiers = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)),
        ]),
        "GBDT": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)),
        ]),
        "LinearSVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5000, random_state=42), cv=3)),
        ]),
    }

    # XGBoost（如果可用）
    try:
        import xgboost as xgb
        classifiers["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                       random_state=42, use_label_encoder=False, eval_metric="logloss")),
        ])
    except ImportError:
        LOGGER.warning("xgboost 未安装，跳过")

    results = {
        "layer": args.layer,
        "embedding_dir": args.embedding_dir,
        "classifiers": [],
    }

    for name, pipe in classifiers.items():
        result = evaluate_classifier(name, pipe, X_train, y_train, X_val, y_val, X_test, y_test)
        results["classifiers"].append(result)

    # 保存
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
