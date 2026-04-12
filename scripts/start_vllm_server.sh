#!/usr/bin/env bash

set -euo pipefail

# Minimal launcher for a local OpenAI-compatible vLLM server.
# Example:
#   VLLM_MODEL="Qwen3-30B-A3B-Instruct" ./scripts/start_vllm_server.sh

MODEL="${VLLM_MODEL:-gpt-oss-120b}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
API_KEY="${VLLM_API_KEY:-dummy}"
DTYPE="${VLLM_DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2}"

echo "[vLLM] starting server"
echo "[vLLM] model=${MODEL}"
echo "[vLLM] endpoint=http://${HOST}:${PORT}/v1"

exec vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-num-seqs "${MAX_NUM_SEQS}"
