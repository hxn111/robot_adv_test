#!/usr/bin/env bash

set -euo pipefail

export USER="${USER:-chtc}"
export LOGNAME="${LOGNAME:-${USER}}"
export USERNAME="${USERNAME:-${USER}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-.torchinductor}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

bash scripts/start_vllm_server.sh &
sleep "${VLLM_START_WAIT_S:-60}"

exec python3 test_llm.py "$@"
