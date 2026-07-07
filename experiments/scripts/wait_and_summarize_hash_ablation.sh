#!/bin/bash
# Wait for manually launched hash-ablation workers, then write summary.md.

set -euo pipefail

MELD_ROOT="${MELD_ROOT:-$(pwd)}"
SCRIPTS="${MELD_ROOT}/experiments/scripts"
OUT_DIR="${OUT_DIR:-${MELD_ROOT}/experiments/results_v2/hash_ablation_strengthened}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"

QWEN_PID="$(cat "${OUT_DIR}/pid_qwen3")"
BERT_PID="$(cat "${OUT_DIR}/pid_bert")"

echo "Waiting for hash ablation workers: qwen3=${QWEN_PID}, bert=${BERT_PID}"
while kill -0 "${QWEN_PID}" 2>/dev/null || kill -0 "${BERT_PID}" 2>/dev/null; do
    date
    sleep "${SLEEP_SECONDS}"
done

test -f "${OUT_DIR}/qwen3/hash_ablation_results.json"
test -f "${OUT_DIR}/bert/hash_ablation_results.json"

python3 "${SCRIPTS}/summarize_hash_ablation.py" --out_dir "${OUT_DIR}"
