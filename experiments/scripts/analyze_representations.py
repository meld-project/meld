#!/usr/bin/env python3
"""
表示分析脚本 — RQ4

1. t-SNE 可视化（浅层 / 最优中间层 / 最终层）
2. Fisher 判别比逐层计算
3. 线性可分性 gap 分析
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("analyze")


def fisher_discriminant_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """类间方差 / 类内方差。"""
    classes = np.unique(y)
    if len(classes) != 2:
        return 0.0

    X0 = X[y == classes[0]]
    X1 = X[y == classes[1]]

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    between = np.sum((mu1 - mu0) ** 2)

    var0 = np.mean(np.var(X0, axis=0))
    var1 = np.mean(np.var(X1, axis=0))
    within = var0 + var1

    if within < 1e-10:
        return float("inf")
    return float(between / within)


def load_embeddings_safe(embedding_dir: str, layer: int) -> np.ndarray:
    """加载嵌入并处理 inf/nan。"""
    X = np.load(Path(embedding_dir) / f"layer_{layer:02d}.npy").astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def compute_layer_fisher(embedding_dir: str, labels: np.ndarray, indices: np.ndarray, num_layers: int):
    """逐层计算 Fisher 判别比。"""
    results = []
    y = labels[indices]

    for layer in range(1, num_layers + 1):
        X = load_embeddings_safe(embedding_dir, layer)
        X_subset = X[indices]

        fdr = fisher_discriminant_ratio(X_subset, y)
        results.append({"layer": layer, "fisher_ratio": round(fdr, 6)})
        LOGGER.info(f"  层{layer}: Fisher={fdr:.4f}")

    return results


def compute_tsne(embedding_dir: str, labels: np.ndarray, indices: np.ndarray,
                 layers: list, output_dir: str, sample_ids=None):
    """对指定层做 t-SNE 并保存坐标。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    y = labels[indices]
    # 限制样本数以加速
    max_samples = 5000
    if len(indices) > max_samples:
        rng = np.random.RandomState(42)
        sel = rng.choice(len(indices), max_samples, replace=False)
        indices = indices[sel]
        y = labels[indices]

    for layer in layers:
        LOGGER.info(f"t-SNE 层{layer}...")
        X = load_embeddings_safe(embedding_dir, layer)
        X_subset = X[indices]

        # 标准化
        X_scaled = StandardScaler().fit_transform(X_subset)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        coords = tsne.fit_transform(X_scaled)

        np.savez(
            out / f"tsne_layer_{layer:02d}.npz",
            coords=coords,
            labels=y,
            indices=indices,
        )
        LOGGER.info(f"  已保存 tsne_layer_{layer:02d}.npz, shape={coords.shape}")


def main():
    parser = argparse.ArgumentParser(description="表示分析")
    parser.add_argument("--embedding_dir", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--family_split_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="model")
    args = parser.parse_args()

    meta_path = Path(args.embedding_dir) / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    num_layers = meta["num_layers"]

    labels = np.load(Path(args.embedding_dir) / "labels.npy")

    with open(args.split_file) as f:
        split = json.load(f)
    test_indices = np.array(split["test_indices"])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Fisher 判别比
    LOGGER.info("=== Fisher 判别比 ===")
    fisher_results = compute_layer_fisher(args.embedding_dir, labels, test_indices, num_layers)
    with open(out / "fisher_ratios.json", "w") as f:
        json.dump({"model": args.model_name, "layers": fisher_results}, f, indent=2)

    # 2. t-SNE（浅层、中间层、最终层）
    LOGGER.info("=== t-SNE 可视化 ===")
    # 选择层：浅层=3, 中间=num_layers*2//3, 最终=num_layers
    shallow = 3
    middle = num_layers * 2 // 3
    final = num_layers
    tsne_layers = [shallow, middle, final]
    LOGGER.info(f"t-SNE 层: {tsne_layers}")

    compute_tsne(args.embedding_dir, labels, test_indices, tsne_layers, str(out / "tsne"))

    LOGGER.info("分析完成！")


if __name__ == "__main__":
    main()
