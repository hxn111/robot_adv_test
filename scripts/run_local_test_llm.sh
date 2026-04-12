#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${ROOT_DIR}/scripts/start_vllm_server.sh" &
sleep "${VLLM_START_WAIT_S:-20}"

cd "${ROOT_DIR}"
exec python3 test_llm.py "$@"
