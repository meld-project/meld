#!/bin/bash
# Strengthened hash ablation for MELD.
#
# Runs the same Time-OOD split and LR C-selection protocol as the main
# layer-profile experiment. The main paper table should use with_hash vs
# without_hash; without_strong_identifiers can be enabled as a sanity check by
# setting MELD_HASH_CONDITIONS.

set -euo pipefail

MELD_ROOT="${MELD_ROOT:-$(pwd)}"
SCRIPTS="${MELD_ROOT}/experiments/scripts"
DATA_DIR="${DATA_DIR:-/data/meld-data}"
RESULTS_BASE="${RESULTS_BASE:-${MELD_ROOT}/experiments/results_v2}"
SPLITS_DIR="${SPLITS_DIR:-${MELD_ROOT}/experiments/splits}"
EMBEDDING_CACHE_BASE="${EMBEDDING_CACHE_BASE:-/data/rkodata/meld_alllayer_cache}"

MAL_MD="${MAL_MD:-${DATA_DIR}/cape_reports_malicious_md}"
BEN_MD="${BEN_MD:-${DATA_DIR}/cape_reports_benign_md}"
TIME_SPLIT="${TIME_SPLIT:-${SPLITS_DIR}/time_ood_split.json}"

QWEN_MODEL="${QWEN_MODEL:-${MELD_ROOT}/models/Qwen3-0.6B}"
if [ -z "${BERT_MODEL:-}" ]; then
    BERT_CACHE_DIR="${BERT_CACHE_DIR:-${HOME}/.cache/huggingface/hub/models--bert-base-uncased/snapshots}"
    if [ -d "${BERT_CACHE_DIR}" ]; then
        BERT_MODEL="$(find "${BERT_CACHE_DIR}" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)"
    else
        BERT_MODEL="bert-base-uncased"
    fi
fi

OUT_DIR="${OUT_DIR:-${RESULTS_BASE}/hash_ablation_strengthened}"
CONDITION_CACHE_DIR="${CONDITION_CACHE_DIR:-${OUT_DIR}/embedding_cache}"
CONDITIONS="${MELD_HASH_CONDITIONS:-with_hash,without_hash}"
C_VALUES="${MELD_C_VALUES:-0.01,0.1,1.0}"
SELECTION_METRIC="${MELD_SELECTION_METRIC:-val_macro_f1}"
PARALLEL="${MELD_HASH_PARALLEL:-1}"

mkdir -p "${OUT_DIR}"

echo "============================================"
echo "MELD strengthened hash ablation"
echo "split=${TIME_SPLIT}"
echo "conditions=${CONDITIONS}"
echo "C=${C_VALUES}; selection=${SELECTION_METRIC}"
echo "output=${OUT_DIR}"
echo "main embedding cache=${EMBEDDING_CACHE_BASE}"
echo "ablation embedding cache=${CONDITION_CACHE_DIR}"
echo "parallel=${PARALLEL}"
echo "============================================"

run_qwen3() {
python3 "${SCRIPTS}/hash_ablation.py" \
    --model_dir "${QWEN_MODEL}" \
    --model_name qwen3 \
    --mal_md_dir "${MAL_MD}" \
    --ben_md_dir "${BEN_MD}" \
    --split_file "${TIME_SPLIT}" \
    --output_dir "${OUT_DIR}/qwen3" \
    --baseline_embedding_dir "${EMBEDDING_CACHE_BASE}/qwen3" \
    --condition_cache_dir "${CONDITION_CACHE_DIR}/qwen3" \
    --layers 7,28 \
    --conditions "${CONDITIONS}" \
    --c_values "${C_VALUES}" \
    --selection_metric "${SELECTION_METRIC}" \
    --max_tokens 1024 \
    --stride 256 \
    --gpu "${QWEN_GPU:-0}" \
    --trust_remote_code \
    2>&1 | tee "${OUT_DIR}/log_qwen3.txt"
}

run_bert() {
python3 "${SCRIPTS}/hash_ablation.py" \
    --model_dir "${BERT_MODEL}" \
    --model_name bert \
    --mal_md_dir "${MAL_MD}" \
    --ben_md_dir "${BEN_MD}" \
    --split_file "${TIME_SPLIT}" \
    --output_dir "${OUT_DIR}/bert" \
    --baseline_embedding_dir "${EMBEDDING_CACHE_BASE}/bert" \
    --condition_cache_dir "${CONDITION_CACHE_DIR}/bert" \
    --layers 11,12 \
    --conditions "${CONDITIONS}" \
    --c_values "${C_VALUES}" \
    --selection_metric "${SELECTION_METRIC}" \
    --max_tokens 512 \
    --stride 128 \
    --gpu "${BERT_GPU:-1}" \
    2>&1 | tee "${OUT_DIR}/log_bert.txt"
}

if [ "${PARALLEL}" = "1" ]; then
    run_qwen3 &
    QWEN_PID=$!
    run_bert &
    BERT_PID=$!
    echo "${QWEN_PID}" > "${OUT_DIR}/pid_qwen3"
    echo "${BERT_PID}" > "${OUT_DIR}/pid_bert"
    QWEN_STATUS=0
    BERT_STATUS=0
    wait "${QWEN_PID}" || QWEN_STATUS=$?
    wait "${BERT_PID}" || BERT_STATUS=$?
    if [ "${QWEN_STATUS}" -ne 0 ] || [ "${BERT_STATUS}" -ne 0 ]; then
        echo "hash ablation failed: qwen3=${QWEN_STATUS}, bert=${BERT_STATUS}" >&2
        exit 1
    fi
else
    run_qwen3
    run_bert
fi

python3 "${SCRIPTS}/summarize_hash_ablation.py" --out_dir "${OUT_DIR}"

echo "Summary written to ${OUT_DIR}/summary.md"
