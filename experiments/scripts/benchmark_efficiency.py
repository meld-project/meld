#!/usr/bin/env python3
"""
效率对比 Benchmark — 测量各方案的训练时间、推理延迟、显存和参数量。

用法:
    python benchmark_efficiency.py \
        --model_dir /path/to/Qwen3-0.6B \
        --ft_encoder_dir /path/to/ft_qwen3_encoder \
        --bert_model_name bert-base-uncased \
        --embedding_dir /path/to/meld_alllayer_cache/qwen3 \
        --split_file /path/to/time_ood_split.json \
        --mal_md_dir /path/to/cape_reports_malicious_md \
        --ben_md_dir /path/to/cape_reports_benign_md \
        --output experiments/results/supplementary/p0_efficiency/efficiency_benchmark.json \
        --gpu 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lec.feature_extractor import LayerwiseFeatureExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("benchmark")

N_WARMUP = 5
N_MEASURE = 50


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def benchmark_inference(extractor, texts, max_tokens=1024, stride=256, until_layer=None):
    """测量推理延迟和峰值显存。"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Warmup
    for i in range(min(N_WARMUP, len(texts))):
        extractor.encode_document_layers(texts[i], max_tokens=max_tokens, stride=stride, until_layer=until_layer)
    torch.cuda.synchronize()

    # Measure
    n = min(N_MEASURE, len(texts))
    start = time.perf_counter()
    for i in range(n):
        extractor.encode_document_layers(texts[i], max_tokens=max_tokens, stride=stride, until_layer=until_layer)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

    return {
        "latency_per_sample_ms": round(elapsed / n * 1000, 2),
        "peak_memory_mb": round(peak_mem, 1),
        "num_samples": n,
    }


def benchmark_lr_training(embedding_dir, split_file, layer):
    """测量 LR 训练时间。"""
    X = np.load(Path(embedding_dir) / f"layer_{layer:02d}.npy").astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.load(Path(embedding_dir) / "labels.npy")

    with open(split_file) as f:
        split = json.load(f)
    train_idx = np.array(split["train_indices"])

    X_train = X[train_idx]
    y_train = labels[train_idx]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")),
    ])

    start = time.perf_counter()
    pipe.fit(X_train, y_train)
    elapsed = time.perf_counter() - start

    return {
        "training_time_sec": round(elapsed, 3),
        "num_train_samples": len(X_train),
        "feature_dim": X_train.shape[1],
        "trainable_params": X_train.shape[1] + 1,  # weights + bias
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="预训练 Qwen3 模型路径")
    parser.add_argument("--ft_encoder_dir", default=None, help="微调后 encoder 路径（可选）")
    parser.add_argument("--embedding_dir", required=True, help="全层嵌入缓存目录")
    parser.add_argument("--split_file", required=True, help="Time-OOD 划分文件")
    parser.add_argument("--mal_md_dir", required=True)
    parser.add_argument("--ben_md_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    results = {}

    # 采样文本
    mal_paths = sorted(Path(args.mal_md_dir).glob("*.md"))[:100]
    ben_paths = sorted(Path(args.ben_md_dir).glob("*.md"))[:50]
    sample_texts = [p.read_text(encoding="utf-8", errors="replace") for p in mal_paths + ben_paths]
    LOGGER.info(f"采样 {len(sample_texts)} 条文本用于 benchmark")

    # === 1. 预训练模型推理（全层 vs 部分层）===
    LOGGER.info("=== 预训练 Qwen3 推理 benchmark ===")
    extractor = LayerwiseFeatureExtractor(
        model_dir=args.model_dir, device=device, dtype="float16",
        trust_remote_code=args.trust_remote_code,
    )
    total_params, _ = count_parameters(extractor.model)

    # 全 28 层
    LOGGER.info("全层 (L28) 推理...")
    inf_full = benchmark_inference(extractor, sample_texts, until_layer=None)
    results["meld_full_28layers"] = {
        **inf_full,
        "total_params": total_params,
        "description": "冻结 Qwen3 全 28 层前向传播",
    }

    # 前 7 层
    LOGGER.info("部分层 (L7) 推理...")
    torch.cuda.empty_cache()
    inf_7 = benchmark_inference(extractor, sample_texts, until_layer=7)
    results["meld_partial_7layers"] = {
        **inf_7,
        "total_params": total_params,
        "active_layers": 7,
        "description": "冻结 Qwen3 前 7 层前向传播",
    }

    # 前 15 层
    LOGGER.info("部分层 (L15) 推理...")
    torch.cuda.empty_cache()
    inf_15 = benchmark_inference(extractor, sample_texts, until_layer=15)
    results["meld_partial_15layers"] = {
        **inf_15,
        "total_params": total_params,
        "active_layers": 15,
        "description": "冻结 Qwen3 前 15 层前向传播",
    }

    del extractor
    torch.cuda.empty_cache()

    # === 2. 微调模型推理（如有 checkpoint）===
    if args.ft_encoder_dir and Path(args.ft_encoder_dir).exists():
        LOGGER.info("=== 微调 Qwen3 推理 benchmark ===")
        ft_extractor = LayerwiseFeatureExtractor(
            model_dir=args.ft_encoder_dir, device=device, dtype="float16",
            trust_remote_code=args.trust_remote_code,
        )
        ft_params, _ = count_parameters(ft_extractor.model)
        inf_ft = benchmark_inference(ft_extractor, sample_texts, until_layer=None)
        results["ft_qwen3_inference"] = {
            **inf_ft,
            "total_params": ft_params,
            "description": "微调 Qwen3 全 28 层推理",
        }
        del ft_extractor
        torch.cuda.empty_cache()

    # === 3. LR 训练时间 ===
    LOGGER.info("=== LR 训练 benchmark ===")
    lr_result = benchmark_lr_training(args.embedding_dir, args.split_file, layer=7)
    results["lr_training_layer7"] = {
        **lr_result,
        "description": "逻辑回归训练（L7 嵌入，CPU）",
    }

    lr_result_28 = benchmark_lr_training(args.embedding_dir, args.split_file, layer=28)
    results["lr_training_layer28"] = {
        **lr_result_28,
        "description": "逻辑回归训练（L28 嵌入，CPU）",
    }

    # === 4. 模型存储大小 ===
    model_size_mb = sum(
        f.stat().st_size for f in Path(args.model_dir).rglob("*") if f.is_file()
    ) / (1024 ** 2)
    results["model_storage"] = {
        "pretrained_qwen3_mb": round(model_size_mb, 1),
        "lr_model_kb": 4.0 + 0.004,  # ~4KB for LR weights on 1024-dim
    }

    # === 汇总 ===
    LOGGER.info("=== 结果汇总 ===")
    for k, v in results.items():
        LOGGER.info(f"  {k}: {v}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
