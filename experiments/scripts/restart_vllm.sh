#!/bin/bash
# 恢复 VLLM 推理服务（实验跑完后执行）
# 原始进程信息记录于 2026-04-19

# Gemma-4-31B-it (tensor parallel 2, GPU 0+1)
nohup vllm serve \
    --model /data/models/gemma-4-31B-it \
    --served-model-name gemma-4-31B-it \
    --host 0.0.0.0 --port 32312 \
    --gpu-memory-utilization 0.77 \
    --max-model-len 262144 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 2 \
    --tensor-parallel-size 2 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    > /tmp/vllm_gemma4.log 2>&1 &

echo "Gemma-4-31B-it 已启动, PID=$!"
