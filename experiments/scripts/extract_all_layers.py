#!/usr/bin/env python3
"""
全层嵌入提取脚本 — MELD 实验阶段二

对指定模型提取全部样本在每一层的 mean-pooled 表示，按层分片存储。
支持断点续传（跳过已存在的层文件）和 float16 存储。

用法:
    python extract_all_layers.py \
        --model_dir /path/to/Qwen3-0.6B \
        --model_name qwen3 \
        --mal_md_dir /path/to/cape_reports_malicious_md \
        --ben_md_dir /path/to/cape_reports_benign_md \
        --output_dir /path/to/meld_alllayer_cache/qwen3 \
        --max_tokens 1024 \
        --stride 256 \
        --dtype float16 \
        --batch_size 64 \
        --gpu 0

输出:
    {output_dir}/layer_{i:02d}.npy   — shape [N, hidden_size], 每层一个文件
    {output_dir}/labels.npy          — shape [N], 1=恶意 0=良性
    {output_dir}/sample_ids.npy      — shape [N], 文件名（不含后缀）
    {output_dir}/meta.json           — 模型信息、样本数、层数等
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lec.feature_extractor import LayerwiseFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("extract_all_layers")


def collect_samples(
    mal_md_dir: str,
    ben_md_dir: str,
    limit: Optional[int] = None,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """收集所有样本路径、标签和ID。"""
    mal_paths = sorted(Path(mal_md_dir).glob("*.md"))
    ben_paths = sorted(Path(ben_md_dir).glob("*.md"))

    if limit:
        mal_paths = mal_paths[:limit]
        ben_paths = ben_paths[:limit]

    all_paths = list(mal_paths) + list(ben_paths)
    labels = np.array([1] * len(mal_paths) + [0] * len(ben_paths), dtype=np.int8)
    sample_ids = [p.stem for p in all_paths]

    LOGGER.info(f"恶意样本: {len(mal_paths)}, 良性样本: {len(ben_paths)}, 总计: {len(all_paths)}")
    return [str(p) for p in all_paths], labels, sample_ids


def extract_and_save(
    extractor: LayerwiseFeatureExtractor,
    paths: List[str],
    labels: np.ndarray,
    sample_ids: List[str],
    output_dir: str,
    max_tokens: int = 1024,
    stride: int = 256,
    save_dtype: str = "float16",
) -> None:
    """逐样本提取全层嵌入，按层累积并存储。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    num_layers = extractor.num_model_layers
    hidden_size = extractor.hidden_size
    N = len(paths)

    np_dtype = np.float16 if save_dtype == "float16" else np.float32

    # 检查哪些层已提取完成
    existing_layers = set()
    for i in range(1, num_layers + 1):
        layer_file = out / f"layer_{i:02d}.npy"
        if layer_file.exists():
            arr = np.load(layer_file)
            if arr.shape == (N, hidden_size):
                existing_layers.add(i)
                LOGGER.info(f"层 {i} 已存在且形状正确，跳过")

    if len(existing_layers) == num_layers:
        LOGGER.info("所有层已提取完毕，无需重新提取")
        _save_metadata(out, extractor, N, num_layers, hidden_size, save_dtype, max_tokens, stride)
        return

    # 初始化层缓冲区（仅未完成的层）
    # 为节省内存，先用 mmap 模式创建文件
    layer_buffers = {}
    for i in range(1, num_layers + 1):
        if i not in existing_layers:
            layer_file = out / f"layer_{i:02d}.npy"
            # 预分配文件
            buf = np.zeros((N, hidden_size), dtype=np_dtype)
            layer_buffers[i] = buf

    # 逐样本提取
    LOGGER.info(f"开始提取 {N} 个样本 × {num_layers - len(existing_layers)} 层...")
    failed_count = 0

    for idx in tqdm(range(N), desc="提取嵌入", disable=not sys.stdout.isatty()):
        try:
            text = Path(paths[idx]).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            LOGGER.warning(f"读取失败 {paths[idx]}: {e}")
            failed_count += 1
            continue

        try:
            # shape: [num_layers, hidden_size]
            features = extractor.encode_document_layers(
                text, max_tokens=max_tokens, stride=stride
            )
            features_np = features.numpy()

            for layer_idx, buf in layer_buffers.items():
                buf[idx] = features_np[layer_idx - 1].astype(np_dtype)

        except Exception as e:
            LOGGER.warning(f"提取失败 [{idx}] {paths[idx]}: {e}")
            failed_count += 1
            continue

        # 每 5000 个样本保存一次中间结果
        if (idx + 1) % 5000 == 0:
            LOGGER.info(f"已处理 {idx + 1}/{N}，保存中间结果...")
            for layer_idx, buf in layer_buffers.items():
                np.save(out / f"layer_{layer_idx:02d}.npy", buf)

    # 最终保存
    LOGGER.info(f"提取完成。失败: {failed_count}/{N}")
    for layer_idx, buf in layer_buffers.items():
        np.save(out / f"layer_{layer_idx:02d}.npy", buf)
        LOGGER.info(f"已保存 layer_{layer_idx:02d}.npy, shape={buf.shape}, dtype={buf.dtype}")

    # 保存标签和样本ID
    np.save(out / "labels.npy", labels)
    np.save(out / "sample_ids.npy", np.array(sample_ids, dtype=object))

    _save_metadata(out, extractor, N, num_layers, hidden_size, save_dtype, max_tokens, stride)


def _save_metadata(
    out: Path,
    extractor: LayerwiseFeatureExtractor,
    N: int,
    num_layers: int,
    hidden_size: int,
    save_dtype: str,
    max_tokens: int,
    stride: int,
) -> None:
    meta = {
        "num_samples": N,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "save_dtype": save_dtype,
        "max_tokens": max_tokens,
        "stride": stride,
        "model_config": {
            "num_hidden_layers": extractor.num_model_layers,
            "hidden_size": extractor.hidden_size,
        },
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"元数据已保存至 {out / 'meta.json'}")


def main():
    parser = argparse.ArgumentParser(description="全层嵌入提取")
    parser.add_argument("--model_dir", required=True, help="HuggingFace 模型路径")
    parser.add_argument("--model_name", default="model", help="模型名称（用于日志）")
    parser.add_argument("--mal_md_dir", required=True, help="恶意样本 MD 目录")
    parser.add_argument("--ben_md_dir", required=True, help="良性样本 MD 目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="限制样本数（调试用）")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"模型: {args.model_name}, 设备: {device}")

    # 收集样本
    paths, labels, sample_ids = collect_samples(
        args.mal_md_dir, args.ben_md_dir, limit=args.limit
    )

    # 初始化提取器
    LOGGER.info(f"加载模型 {args.model_dir}...")
    extractor = LayerwiseFeatureExtractor(
        model_dir=args.model_dir,
        device=device,
        dtype="float16",  # 推理用 float16 节省显存
        trust_remote_code=args.trust_remote_code,
    )
    LOGGER.info(f"模型层数: {extractor.num_model_layers}, hidden_size: {extractor.hidden_size}")

    # 提取并保存
    extract_and_save(
        extractor=extractor,
        paths=paths,
        labels=labels,
        sample_ids=sample_ids,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        stride=args.stride,
        save_dtype=args.dtype,
    )

    LOGGER.info("全部完成！")


if __name__ == "__main__":
    main()
