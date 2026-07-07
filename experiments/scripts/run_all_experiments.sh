#!/bin/bash
# MELD 完整实验流水线 — 一键运行
# 在 h200 上执行: bash run_all_experiments.sh
set -e

# ============================================================
# 路径配置
# ============================================================
MELD_ROOT="${MELD_ROOT:-$(pwd)}"
SCRIPTS="${MELD_ROOT}/experiments/scripts"
DATA_DIR="${MELD_DATA_DIR:-${MELD_ROOT}/data}"
CACHE_BASE="${MELD_CACHE_BASE:-${MELD_ROOT}/cache/meld_alllayer_cache}"
RESULTS_BASE="${MELD_ROOT}/experiments/results_v2"
SPLITS_DIR="${MELD_ROOT}/experiments/splits"

MAL_MD="${DATA_DIR}/cape_reports_malicious_md"
BEN_MD="${DATA_DIR}/cape_reports_benign_md"

QWEN_MODEL="${MELD_ROOT}/models/Qwen3-0.6B"
BERT_MODEL="bert-base-uncased"
LLAMA_MODEL="${LLAMA_MODEL_DIR:-/path/to/llama-2-7b}"

mkdir -p "${CACHE_BASE}" "${RESULTS_BASE}" "${SPLITS_DIR}"

echo "============================================"
echo "MELD 完整实验流水线"
echo "开始时间: $(date)"
echo "============================================"

# ============================================================
# 阶段二：全层嵌入提取
# ============================================================
echo ""
echo ">>> 阶段二：全层嵌入提取"
echo "============================================"

# Qwen3-0.6B（~2h）
echo "[1/3] Qwen3-0.6B 全层提取..."
python3 "${SCRIPTS}/extract_all_layers.py" \
    --model_dir "${QWEN_MODEL}" \
    --model_name qwen3 \
    --mal_md_dir "${MAL_MD}" \
    --ben_md_dir "${BEN_MD}" \
    --output_dir "${CACHE_BASE}/qwen3" \
    --max_tokens 1024 --stride 256 --dtype float16 --gpu 0 \
    --trust_remote_code \
    2>&1 | tee "${RESULTS_BASE}/log_extract_qwen3.txt"

# BERT-base（~0.5h）
echo "[2/3] BERT-base 全层提取..."
python3 "${SCRIPTS}/extract_all_layers.py" \
    --model_dir "${BERT_MODEL}" \
    --model_name bert \
    --mal_md_dir "${MAL_MD}" \
    --ben_md_dir "${BEN_MD}" \
    --output_dir "${CACHE_BASE}/bert" \
    --max_tokens 512 --stride 128 --dtype float16 --gpu 0 \
    2>&1 | tee "${RESULTS_BASE}/log_extract_bert.txt"

# LLaMA-2-7B（~8h，后台运行）
echo "[3/3] LLaMA-2-7B 全层提取..."
python3 "${SCRIPTS}/extract_all_layers.py" \
    --model_dir "${LLAMA_MODEL}" \
    --model_name llama2 \
    --mal_md_dir "${MAL_MD}" \
    --ben_md_dir "${BEN_MD}" \
    --output_dir "${CACHE_BASE}/llama2" \
    --max_tokens 1024 --stride 256 --dtype float16 --gpu 1 \
    2>&1 | tee "${RESULTS_BASE}/log_extract_llama2.txt"

echo ">>> 嵌入提取完成: $(date)"

# ============================================================
# 阶段三：逐层性能评估 (RQ1)
# ============================================================
echo ""
echo ">>> 阶段三：逐层性能剖面 (RQ1)"
echo "============================================"

for model in qwen3 bert llama2; do
    echo "评估 ${model} 逐层性能..."
    python3 "${SCRIPTS}/evaluate_all_layers.py" \
        --embedding_dir "${CACHE_BASE}/${model}" \
        --split_file "${SPLITS_DIR}/time_ood_split.json" \
        --output "${RESULTS_BASE}/${model}_layer_profile.json" \
        --model_name "${model}" \
        2>&1 | tee "${RESULTS_BASE}/log_eval_layers_${model}.txt"
done

echo ">>> 逐层评估完成: $(date)"

# ============================================================
# 阶段三：Family-OOD 逐层评估 (RQ3)
# ============================================================
echo ""
echo ">>> 阶段三：Family-OOD 逐层评估 (RQ3)"
echo "============================================"

# 主模型 Qwen3 全层评估
echo "Qwen3 Family-OOD 全层..."
python3 "${SCRIPTS}/evaluate_family_ood_layers.py" \
    --embedding_dir "${CACHE_BASE}/qwen3" \
    --family_split_file "${SPLITS_DIR}/family_ood_spec.json" \
    --output "${RESULTS_BASE}/qwen3_family_ood_layer_profile.json" \
    --model_name qwen3 \
    2>&1 | tee "${RESULTS_BASE}/log_family_ood_qwen3.txt"

# BERT 和 LLaMA 只评估关键层（加速）
for model in bert llama2; do
    echo "${model} Family-OOD 全层..."
    python3 "${SCRIPTS}/evaluate_family_ood_layers.py" \
        --embedding_dir "${CACHE_BASE}/${model}" \
        --family_split_file "${SPLITS_DIR}/family_ood_spec.json" \
        --output "${RESULTS_BASE}/${model}_family_ood_layer_profile.json" \
        --model_name "${model}" \
        2>&1 | tee "${RESULTS_BASE}/log_family_ood_${model}.txt"
done

echo ">>> Family-OOD 评估完成: $(date)"

# ============================================================
# 阶段三：基线评估 (RQ3)
# ============================================================
echo ""
echo ">>> 阶段三：基线评估 (RQ3)"
echo "============================================"

echo "运行基线（TF-IDF, 嵌入变体, 微调）..."
python3 "${SCRIPTS}/run_baselines.py" \
    --embedding_dir "${CACHE_BASE}/qwen3" \
    --mal_md_dir "${MAL_MD}" \
    --ben_md_dir "${BEN_MD}" \
    --time_ood_split "${SPLITS_DIR}/time_ood_split.json" \
    --family_ood_split "${SPLITS_DIR}/family_ood_spec.json" \
    --output_dir "${RESULTS_BASE}/baselines" \
    2>&1 | tee "${RESULTS_BASE}/log_baselines.txt"

echo ">>> 基线评估完成: $(date)"

# ============================================================
# 阶段三：微调基线 (RQ3)
# ============================================================
echo ""
echo ">>> 阶段三：微调基线"
echo "============================================"

# FT-BERT
echo "微调 BERT..."
python3 "${SCRIPTS}/finetune_classifier.py" \
    --model_name bert-base-uncased \
    --mal_md_dir "${MAL_MD}" \
    --ben_md_dir "${BEN_MD}" \
    --time_ood_split "${SPLITS_DIR}/time_ood_split.json" \
    --family_ood_split "${SPLITS_DIR}/family_ood_spec.json" \
    --output_dir "${RESULTS_BASE}/ft_bert" \
    --max_length 512 --epochs 3 --lr 2e-5 --batch_size 16 --gpu 0 \
    2>&1 | tee "${RESULTS_BASE}/log_ft_bert.txt"

# FT-Qwen3
echo "微调 Qwen3..."
python3 "${SCRIPTS}/finetune_classifier.py" \
    --model_name "${QWEN_MODEL}" \
    --mal_md_dir "${MAL_MD}" \
    --ben_md_dir "${BEN_MD}" \
    --time_ood_split "${SPLITS_DIR}/time_ood_split.json" \
    --family_ood_split "${SPLITS_DIR}/family_ood_spec.json" \
    --output_dir "${RESULTS_BASE}/ft_qwen3" \
    --max_length 1024 --epochs 3 --lr 1e-5 --batch_size 4 \
    --gradient_accumulation 4 --gpu 0 \
    --trust_remote_code \
    2>&1 | tee "${RESULTS_BASE}/log_ft_qwen3.txt"

echo ">>> 微调基线完成: $(date)"

# ============================================================
# 阶段四：深度分析 (RQ4)
# ============================================================
echo ""
echo ">>> 阶段四：表示分析 (RQ4)"
echo "============================================"

python3 "${SCRIPTS}/analyze_representations.py" \
    --embedding_dir "${CACHE_BASE}/qwen3" \
    --split_file "${SPLITS_DIR}/time_ood_split.json" \
    --family_split_file "${SPLITS_DIR}/family_ood_spec.json" \
    --output_dir "${RESULTS_BASE}/analysis" \
    --model_name qwen3 \
    2>&1 | tee "${RESULTS_BASE}/log_analysis.txt"

echo ">>> 分析完成: $(date)"

# ============================================================
# 阶段四：Hash 消融
# ============================================================
echo ""
echo ">>> 阶段四：Hash 消融"
echo "============================================"

python3 "${SCRIPTS}/hash_ablation.py" \
    --model_dir "${QWEN_MODEL}" \
    --model_name qwen3 \
    --mal_md_dir "${MAL_MD}" \
    --ben_md_dir "${BEN_MD}" \
    --split_file "${SPLITS_DIR}/time_ood_split.json" \
    --output_dir "${RESULTS_BASE}/hash_ablation" \
    --baseline_embedding_dir "${CACHE_BASE}/qwen3" \
    --condition_cache_dir "${RESULTS_BASE}/hash_ablation/embedding_cache/qwen3" \
    --layers 7,28 \
    --conditions with_hash,without_hash \
    --c_values 0.01,0.1,1.0 \
    --selection_metric val_macro_f1 \
    --max_tokens 1024 \
    --stride 256 \
    --gpu 0 \
    --trust_remote_code \
    2>&1 | tee "${RESULTS_BASE}/log_hash_ablation.txt"

echo ">>> Hash 消融完成: $(date)"

# ============================================================
echo ""
echo "============================================"
echo "全部实验完成！"
echo "结束时间: $(date)"
echo "结果目录: ${RESULTS_BASE}"
echo "============================================"
