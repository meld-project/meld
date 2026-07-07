#!/usr/bin/env python3
"""
基线评估脚本 — TF-IDF 和嵌入变体

基线列表（不含 Nebula）：
  - TF-IDF word + LR
  - TF-IDF char + LR
  - TF-IDF word + GBDT
  - MELD-Mean（全层均值）
  - MELD-Last4（后4层拼接）
  - MELD-MLP（最优层 + 2层MLP）

在 Time-OOD 和 Family-OOD 上评估。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("baselines")


def load_layer(embedding_dir, layer_idx):
    """安全加载层嵌入，处理 inf/nan。"""
    X = np.load(Path(embedding_dir) / f"layer_{layer_idx:02d}.npy").astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def tpr_at_fpr(y_true, y_prob, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid = fpr <= target_fpr
    return float(tpr[valid][-1]) if valid.any() else 0.0


def evaluate_classifier(clf, X_train, y_train, X_val, y_val, X_test, y_test):
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    id_f1 = float(f1_score(y_val, val_pred, average="macro"))

    test_pred = clf.predict(X_test)
    ood_f1 = float(f1_score(y_test, test_pred, average="macro"))

    if hasattr(clf, "predict_proba"):
        test_prob = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        test_prob = clf.decision_function(X_test)
    else:
        test_prob = test_pred.astype(float)

    try:
        auc = float(roc_auc_score(y_test, test_prob))
    except ValueError:
        auc = 0.0

    return {
        "id_macro_f1": id_f1,
        "ood_macro_f1": ood_f1,
        "f1_drop": round(id_f1 - ood_f1, 6),
        "auc": auc,
        "tpr_at_fpr_0.001": tpr_at_fpr(y_test, test_prob, 0.001),
        "tpr_at_fpr_0.005": tpr_at_fpr(y_test, test_prob, 0.005),
        "tpr_at_fpr_0.01": tpr_at_fpr(y_test, test_prob, 0.01),
    }


def load_texts(mal_md_dir, ben_md_dir):
    mal_paths = sorted(Path(mal_md_dir).glob("*.md"))
    ben_paths = sorted(Path(ben_md_dir).glob("*.md"))
    texts, labels = [], []
    for p in mal_paths:
        texts.append(p.read_text(encoding="utf-8", errors="replace"))
        labels.append(1)
    for p in ben_paths:
        texts.append(p.read_text(encoding="utf-8", errors="replace"))
        labels.append(0)
    return texts, np.array(labels)


def run_tfidf_baselines(texts, labels, split, output_dir):
    """运行 TF-IDF 基线。"""
    results = {}

    configs = [
        ("tfidf_word_lr", "word", "lr"),
        ("tfidf_char_lr", "char", "lr"),
        ("tfidf_word_gbdt", "word", "gbdt"),
    ]

    for name, analyzer, clf_type in configs:
        LOGGER.info(f"运行基线: {name}")

        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=(1, 3) if analyzer == "word" else (2, 5),
            max_features=50000,
            sublinear_tf=True,
        )

        train_texts = [texts[i] for i in split["train"]]
        val_texts = [texts[i] for i in split["val"]]
        test_texts = [texts[i] for i in split["test"]]

        X_train = vectorizer.fit_transform(train_texts)
        X_val = vectorizer.transform(val_texts)
        X_test = vectorizer.transform(test_texts)

        y_train = labels[split["train"]]
        y_val = labels[split["val"]]
        y_test = labels[split["test"]]

        if clf_type == "lr":
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        else:
            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
            )

        result = evaluate_classifier(clf, X_train, y_train, X_val, y_val, X_test, y_test)
        results[name] = result
        LOGGER.info(f"  {name}: F1={result['ood_macro_f1']:.4f}, TPR@0.1%={result['tpr_at_fpr_0.001']:.4f}")

    save_json(results, Path(output_dir) / "tfidf_baselines.json")
    return results


def run_embedding_variants(embedding_dir, labels, split, output_dir):
    """运行嵌入变体基线：Mean, Last4, MLP。"""
    meta_path = Path(embedding_dir) / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    num_layers = meta["num_layers"]

    results = {}

    y_train = labels[split["train"]]
    y_val = labels[split["val"]]
    y_test = labels[split["test"]]

    # --- MELD-Mean: 全层均值 ---
    LOGGER.info("运行基线: MELD-Mean")
    X_mean = np.zeros_like(load_layer(embedding_dir, 1))
    for i in range(1, num_layers + 1):
        X_mean += load_layer(embedding_dir, i)
    X_mean /= num_layers

    clf_mean = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000)),
    ])
    results["meld_mean"] = evaluate_classifier(
        clf_mean, X_mean[split["train"]], y_train,
        X_mean[split["val"]], y_val, X_mean[split["test"]], y_test
    )
    LOGGER.info(f"  MELD-Mean: F1={results['meld_mean']['ood_macro_f1']:.4f}")

    # --- MELD-Last4: 后4层拼接 ---
    LOGGER.info("运行基线: MELD-Last4")
    last4 = []
    for i in range(num_layers - 3, num_layers + 1):
        last4.append(load_layer(embedding_dir, i))
    X_last4 = np.concatenate(last4, axis=1)

    clf_last4 = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000)),
    ])
    results["meld_last4"] = evaluate_classifier(
        clf_last4, X_last4[split["train"]], y_train,
        X_last4[split["val"]], y_val, X_last4[split["test"]], y_test
    )
    LOGGER.info(f"  MELD-Last4: F1={results['meld_last4']['ood_macro_f1']:.4f}")

    # --- MELD-MLP: 最优层 + MLP 分类头 ---
    LOGGER.info("运行基线: MELD-MLP")
    # 先找最优层（用验证集F1）
    best_layer, best_f1 = 1, -1
    for i in range(1, num_layers + 1):
        X_i = load_layer(embedding_dir, i)
        clf_tmp = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000)),
        ])
        clf_tmp.fit(X_i[split["train"]], y_train)
        f1 = f1_score(y_val, clf_tmp.predict(X_i[split["val"]]), average="macro")
        if f1 > best_f1:
            best_f1, best_layer = f1, i

    X_best = load_layer(embedding_dir, best_layer)
    clf_mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 64),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )),
    ])
    results["meld_mlp"] = evaluate_classifier(
        clf_mlp, X_best[split["train"]], y_train,
        X_best[split["val"]], y_val, X_best[split["test"]], y_test
    )
    results["meld_mlp"]["best_layer"] = best_layer
    LOGGER.info(f"  MELD-MLP(层{best_layer}): F1={results['meld_mlp']['ood_macro_f1']:.4f}")

    save_json(results, Path(output_dir) / "embedding_variants.json")
    return results


def run_family_ood_baselines(texts, labels, embedding_dir, family_split_file, output_dir):
    """在 Family-OOD 上运行所有基线。"""
    with open(family_split_file) as f:
        family_spec = json.load(f)

    raw_folds = family_spec["folds"]
    if isinstance(raw_folds, list):
        families = [f["held_out_family"] for f in raw_folds]
        folds_map = {}
        for f in raw_folds:
            folds_map[f["held_out_family"]] = {
                "train_indices": f.get("train_ids", f.get("train_indices", [])),
                "val_indices": f.get("val_ids", f.get("val_indices", [])),
                "test_indices": f.get("test_ids", f.get("test_indices", [])),
            }
    else:
        families = family_spec.get("families", list(raw_folds.keys()))
        folds_map = raw_folds

    meta_path = Path(embedding_dir) / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    num_layers = meta["num_layers"]

    results = {}

    for family in families:
        LOGGER.info(f"=== Family-OOD: {family} ===")
        fold = folds_map[family]
        train_idx = np.array(fold["train_indices"])
        val_idx = np.array(fold["val_indices"])
        test_idx = np.array(fold["test_indices"])

        y_train = labels[train_idx]
        y_val = labels[val_idx]
        y_test = labels[test_idx]

        family_results = {}

        # TF-IDF word + LR
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), max_features=50000, sublinear_tf=True)
        X_tr = vec.fit_transform([texts[i] for i in train_idx])
        X_va = vec.transform([texts[i] for i in val_idx])
        X_te = vec.transform([texts[i] for i in test_idx])
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        family_results["tfidf_word_lr"] = evaluate_classifier(clf, X_tr, y_train, X_va, y_val, X_te, y_test)

        # MELD-Mean
        X_mean = np.zeros_like(load_layer(embedding_dir, 1))
        for i in range(1, num_layers + 1):
            X_mean += load_layer(embedding_dir, i)
        X_mean /= num_layers
        clf_m = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=2000))])
        family_results["meld_mean"] = evaluate_classifier(
            clf_m, X_mean[train_idx], y_train, X_mean[val_idx], y_val, X_mean[test_idx], y_test
        )

        results[family] = family_results
        LOGGER.info(f"  TF-IDF: F1={family_results['tfidf_word_lr']['ood_macro_f1']:.4f}, "
                     f"Mean: F1={family_results['meld_mean']['ood_macro_f1']:.4f}")

    save_json(results, Path(output_dir) / "family_ood_baselines.json")
    return results


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="基线评估")
    parser.add_argument("--embedding_dir", required=True)
    parser.add_argument("--mal_md_dir", required=True)
    parser.add_argument("--ben_md_dir", required=True)
    parser.add_argument("--time_ood_split", required=True)
    parser.add_argument("--family_ood_split", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    texts, labels = load_texts(args.mal_md_dir, args.ben_md_dir)

    with open(args.time_ood_split) as f:
        time_split_raw = json.load(f)
    time_split = {k: np.array(v) for k, v in time_split_raw.items()
                  if k in ("train_indices", "val_indices", "test_indices")}
    time_split = {"train": time_split["train_indices"],
                  "val": time_split["val_indices"],
                  "test": time_split["test_indices"]}

    LOGGER.info("=== Time-OOD TF-IDF 基线 ===")
    run_tfidf_baselines(texts, labels, time_split, args.output_dir)

    LOGGER.info("=== Time-OOD 嵌入变体基线 ===")
    run_embedding_variants(args.embedding_dir, labels, time_split, args.output_dir)

    LOGGER.info("=== Family-OOD 基线 ===")
    run_family_ood_baselines(texts, labels, args.embedding_dir, args.family_ood_split, args.output_dir)

    LOGGER.info("全部基线完成！")


if __name__ == "__main__":
    main()
