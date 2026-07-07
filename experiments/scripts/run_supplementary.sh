#!/bin/bash
# ============================================================
# MELD 补充实验调度脚本 — H200
# 按 Phase 顺序执行，利用多 GPU 并行
# ============================================================
set -euo pipefail

# === 路径配置 ===
MELD_ROOT="${MELD_ROOT:-$(pwd)}"
SCRIPTS="${MELD_ROOT}/experiments/scripts"
DATA="/data/meld-data"
MAL_MD="${DATA}/cape_reports_malicious_md"
BEN_MD="${DATA}/cape_reports_benign_md"
MODEL_DIR="${MELD_ROOT}/models/Qwen3-0.6B"
SPLITS="${MELD_ROOT}/experiments/splits"
TIME_SPLIT="${SPLITS}/time_ood_split.json"
FAMILY_SPLIT="${SPLITS}/family_ood_spec.json"
CACHE_DIR="/data/rkodata/meld_alllayer_cache/qwen3"
RESULTS="${MELD_ROOT}/experiments/results_v2/supplementary"

mkdir -p "${RESULTS}/p0_ft_layer_profile"
mkdir -p "${RESULTS}/p0_efficiency"
mkdir -p "${RESULTS}/p1_seed_variance"
mkdir -p "${RESULTS}/p1_family_baselines"
mkdir -p "${RESULTS}/p2_classifiers"

FT_ENCODER="${RESULTS}/p0_ft_layer_profile/ft_qwen3_encoder"
FT_CACHE="/data/rkodata/meld_alllayer_cache/ft_qwen3"

cd "${MELD_ROOT}"

echo "============================================================"
echo "Phase 1: 微调+保存checkpoint / CPU实验（并行）"
echo "============================================================"

# --- P0-1a: 微调 Qwen3 并保存 checkpoint (GPU 0) ---
echo "[P0-1a] 微调 Qwen3 + 保存 encoder (GPU 0)..."
python "${SCRIPTS}/finetune_classifier.py" \
    --model_name "${MODEL_DIR}" \
    --mal_md_dir "${MAL_MD}" \
    --ben_md_dir "${BEN_MD}" \
    --time_ood_split "${TIME_SPLIT}" \
    --family_ood_split "${FAMILY_SPLIT}" \
    --output_dir "${RESULTS}/p0_ft_layer_profile/ft_results" \
    --save_encoder "${FT_ENCODER}" \
    --max_length 512 \
    --epochs 3 \
    --lr 2e-5 \
    --batch_size 16 \
    --gpu 0 \
    --seed 42 \
    --trust_remote_code \
    2>&1 | tee "${RESULTS}/p0_ft_layer_profile/finetune.log" &
PID_FT=$!

# --- P1-1: 多种子微调 (GPU 1-3) ---
for seed in 40 41 42; do
    gpu_id=$((seed - 39))  # 40->1, 41->2, 42->3
    echo "[P1-1] 种子 ${seed} 微调 (GPU ${gpu_id})..."
    python "${SCRIPTS}/finetune_classifier.py" \
        --model_name "${MODEL_DIR}" \
        --mal_md_dir "${MAL_MD}" \
        --ben_md_dir "${BEN_MD}" \
        --time_ood_split "${TIME_SPLIT}" \
        --family_ood_split "${FAMILY_SPLIT}" \
        --output_dir "${RESULTS}/p1_seed_variance/seed_${seed}" \
        --max_length 512 \
        --epochs 3 \
        --lr 2e-5 \
        --batch_size 16 \
        --gpu ${gpu_id} \
        --seed ${seed} \
        --trust_remote_code \
        2>&1 | tee "${RESULTS}/p1_seed_variance/seed_${seed}.log" &
done

# --- P1-2: Family-OOD 传统方法补全 (CPU) ---
echo "[P1-2] Family-OOD 传统方法补全 (CPU)..."
python "${SCRIPTS}/run_family_ood_baselines_full.py" \
    --mal_md_dir "${MAL_MD}" \
    --ben_md_dir "${BEN_MD}" \
    --family_ood_split "${FAMILY_SPLIT}" \
    --output "${RESULTS}/p1_family_baselines/family_ood_baselines_full.json" \
    2>&1 | tee "${RESULTS}/p1_family_baselines/baselines.log" &
PID_BASELINE=$!

# --- P2: 多分类器对比 (CPU) ---
echo "[P2] 多分类器对比 (CPU)..."
python "${SCRIPTS}/evaluate_classifiers.py" \
    --embedding_dir "${CACHE_DIR}" \
    --split_file "${TIME_SPLIT}" \
    --layer 7 \
    --output "${RESULTS}/p2_classifiers/classifier_comparison.json" \
    2>&1 | tee "${RESULTS}/p2_classifiers/classifiers.log" &
PID_CLF=$!

# 等待 CPU 任务完成
wait ${PID_BASELINE} && echo "[P1-2] Family-OOD 基线完成" || echo "[P1-2] 失败!"
wait ${PID_CLF} && echo "[P2] 分类器对比完成" || echo "[P2] 失败!"

# 等待所有 Phase 1 GPU 任务完成
wait ${PID_FT} && echo "[P0-1a] 微调完成" || echo "[P0-1a] 失败!"
wait  # 等待所有种子微调

echo "============================================================"
echo "Phase 2: 提取微调模型嵌入 + 种子 43/44 + benchmark"
echo "============================================================"

# --- P0-1b: 提取微调后模型全层嵌入 (GPU 0) ---
echo "[P0-1b] 提取微调后模型嵌入 (GPU 0)..."
python "${SCRIPTS}/extract_all_layers.py" \
    --model_dir "${FT_ENCODER}" \
    --model_name ft_qwen3 \
    --mal_md_dir "${MAL_MD}" \
    --ben_md_dir "${BEN_MD}" \
    --output_dir "${FT_CACHE}" \
    --max_tokens 1024 \
    --stride 256 \
    --dtype float16 \
    --gpu 0 \
    --trust_remote_code \
    2>&1 | tee "${RESULTS}/p0_ft_layer_profile/extract.log" &
PID_EXTRACT=$!

# --- P1-1 续: 种子 43, 44 (GPU 1-2) ---
for seed in 43 44; do
    gpu_id=$((seed - 42))  # 43->1, 44->2
    echo "[P1-1] 种子 ${seed} 微调 (GPU ${gpu_id})..."
    python "${SCRIPTS}/finetune_classifier.py" \
        --model_name "${MODEL_DIR}" \
        --mal_md_dir "${MAL_MD}" \
        --ben_md_dir "${BEN_MD}" \
        --time_ood_split "${TIME_SPLIT}" \
        --family_ood_split "${FAMILY_SPLIT}" \
        --output_dir "${RESULTS}/p1_seed_variance/seed_${seed}" \
        --max_length 512 \
        --epochs 3 \
        --lr 2e-5 \
        --batch_size 16 \
        --gpu ${gpu_id} \
        --seed ${seed} \
        --trust_remote_code \
        2>&1 | tee "${RESULTS}/p1_seed_variance/seed_${seed}.log" &
done

# --- P0-2: 效率 benchmark (GPU 3) ---
echo "[P0-2] 效率 benchmark (GPU 3)..."
python "${SCRIPTS}/benchmark_efficiency.py" \
    --model_dir "${MODEL_DIR}" \
    --ft_encoder_dir "${FT_ENCODER}" \
    --embedding_dir "${CACHE_DIR}" \
    --split_file "${TIME_SPLIT}" \
    --mal_md_dir "${MAL_MD}" \
    --ben_md_dir "${BEN_MD}" \
    --output "${RESULTS}/p0_efficiency/efficiency_benchmark.json" \
    --gpu 3 \
    --trust_remote_code \
    2>&1 | tee "${RESULTS}/p0_efficiency/benchmark.log" &

# 等待 Phase 2 完成
wait ${PID_EXTRACT} && echo "[P0-1b] 嵌入提取完成" || echo "[P0-1b] 失败!"
wait  # 等待所有

echo "============================================================"
echo "Phase 3: 逐层评估 + 方差汇总"
echo "============================================================"

# --- P0-1c: 微调模型逐层 LR 评估 ---
echo "[P0-1c] 微调模型逐层 LR 评估..."
python "${SCRIPTS}/evaluate_all_layers.py" \
    --embedding_dir "${FT_CACHE}" \
    --split_file "${TIME_SPLIT}" \
    --output "${RESULTS}/p0_ft_layer_profile/ft_qwen3_layer_profile.json" \
    --model_name ft_qwen3 \
    2>&1 | tee "${RESULTS}/p0_ft_layer_profile/evaluate.log"

# --- P1-1f: 汇总多种子方差 ---
echo "[P1-1f] 汇总多种子方差..."
python3 -c "
import json, glob, numpy as np
from pathlib import Path

results_dir = Path('${RESULTS}/p1_seed_variance')
seeds = {}
for seed_dir in sorted(results_dir.glob('seed_*')):
    seed = seed_dir.name.split('_')[1]
    time_file = seed_dir / 'time_ood_results.json'
    family_file = seed_dir / 'family_ood_results.json'
    if time_file.exists():
        with open(time_file) as f:
            seeds[seed] = {'time_ood': json.load(f)}
    if family_file.exists():
        with open(family_file) as f:
            seeds[seed]['family_ood'] = json.load(f)

# 汇总
if seeds:
    time_metrics = {}
    for metric in ['macro_f1', 'tpr_at_fpr_0.001', 'tpr_at_fpr_0.005', 'auc']:
        vals = [s['time_ood'][metric] for s in seeds.values() if 'time_ood' in s]
        time_metrics[metric] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'values': vals,
        }

    summary = {
        'num_seeds': len(seeds),
        'seeds': list(seeds.keys()),
        'time_ood_summary': time_metrics,
        'per_seed': seeds,
    }

    out = results_dir / 'variance_summary.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'方差汇总已保存至 {out}')
    print(f'TPR@0.1%FPR: mean={time_metrics[\"tpr_at_fpr_0.001\"][\"mean\"]:.4f} +/- {time_metrics[\"tpr_at_fpr_0.001\"][\"std\"]:.4f}')
else:
    print('未找到种子结果文件')
"

echo "============================================================"
echo "全部补充实验完成！"
echo "结果目录: ${RESULTS}"
echo "============================================================"
ls -la "${RESULTS}"/*/
