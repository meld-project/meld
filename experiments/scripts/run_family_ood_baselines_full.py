#!/usr/bin/env python3
"""
Family-OOD 传统方法补全 — TF-IDF word+LR, TF-IDF char+LR, TF-IDF word+GBDT

对 8 个家族的留一折实验，补全 Time-OOD 中已有但 Family-OOD 中缺失的传统基线。

用法:
    python run_family_ood_baselines_full.py \
        --mal_md_dir /data/meld-data/cape_reports_malicious_md \
        --ben_md_dir /data/meld-data/cape_reports_benign_md \
        --family_ood_split /path/to/family_ood_spec.json \
        --output results_v2/supplementary/p1_family_baselines/family_ood_baselines_full.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("family_baselines")


def tpr_at_fpr(y_true, y_prob, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid = fpr <= target_fpr
    return float(tpr[valid][-1]) if valid.any() else 0.0


def evaluate_method(name, clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)

    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X_test)[:, 1]
    else:
        prob = clf.decision_function(X_test)

    pred = clf.predict(X_test)

    return {
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "auc": float(roc_auc_score(y_test, prob)),
        "tpr_at_fpr_0.001": tpr_at_fpr(y_test, prob, 0.001),
        "tpr_at_fpr_0.005": tpr_at_fpr(y_test, prob, 0.005),
        "tpr_at_fpr_0.01": tpr_at_fpr(y_test, prob, 0.01),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mal_md_dir", required=True)
    parser.add_argument("--ben_md_dir", required=True)
    parser.add_argument("--family_ood_split", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # 加载文本
    mal_paths = sorted(Path(args.mal_md_dir).glob("*.md"))
    ben_paths = sorted(Path(args.ben_md_dir).glob("*.md"))
    LOGGER.info(f"加载文本: 恶意 {len(mal_paths)}, 良性 {len(ben_paths)}")

    all_texts = [p.read_text(encoding="utf-8", errors="replace") for p in mal_paths + ben_paths]
    labels = np.array([1] * len(mal_paths) + [0] * len(ben_paths))

    # 加载 Family-OOD 划分
    with open(args.family_ood_split) as f:
        spec = json.load(f)

    raw_folds = spec["folds"]
    if isinstance(raw_folds, list):
        folds = {f["held_out_family"]: f for f in raw_folds}
    else:
        folds = raw_folds

    results = {}
    methods = {
        "tfidf_word_lr": {
            "vectorizer": lambda: TfidfVectorizer(analyzer="word", max_features=50000, sublinear_tf=True),
            "classifier": lambda: LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"),
        },
        "tfidf_char_lr": {
            "vectorizer": lambda: TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=50000, sublinear_tf=True),
            "classifier": lambda: LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"),
        },
        "tfidf_word_gbdt": {
            "vectorizer": lambda: TfidfVectorizer(analyzer="word", max_features=50000, sublinear_tf=True),
            "classifier": lambda: GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
        },
    }

    for method_name, method_cfg in methods.items():
        LOGGER.info(f"\n=== {method_name} ===")
        method_results = {}

        for family, fold in folds.items():
            LOGGER.info(f"  家族: {family}")
            train_idx = np.array(fold.get("train_ids", fold.get("train_indices", [])))
            test_idx = np.array(fold.get("test_ids", fold.get("test_indices", [])))

            train_texts = [all_texts[i] for i in train_idx]
            test_texts = [all_texts[i] for i in test_idx]
            y_train = labels[train_idx]
            y_test = labels[test_idx]

            vec = method_cfg["vectorizer"]()
            X_train = vec.fit_transform(train_texts)
            X_test = vec.transform(test_texts)

            clf = method_cfg["classifier"]()
            fold_result = evaluate_method(method_name, clf, X_train, y_train, X_test, y_test)
            method_results[family] = fold_result
            LOGGER.info(f"    TPR@0.1%={fold_result['tpr_at_fpr_0.001']:.4f}, F1={fold_result['macro_f1']:.4f}")

        # 计算均值
        mean_tpr = np.mean([v["tpr_at_fpr_0.001"] for v in method_results.values()])
        mean_f1 = np.mean([v["macro_f1"] for v in method_results.values()])
        method_results["_mean"] = {"tpr_at_fpr_0.001": float(mean_tpr), "macro_f1": float(mean_f1)}
        LOGGER.info(f"  均值: TPR@0.1%={mean_tpr:.4f}, F1={mean_f1:.4f}")

        results[method_name] = method_results

    # 保存
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
