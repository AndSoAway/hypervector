#!/usr/bin/env bash
# Start the HyperVector gRPC server from a source checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_PYTHON="${REPO_ROOT}/src/python"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
PORT="${PORT:-50051}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${WORKERS:-10}"
MAX_MESSAGE_MB="${HYPERVEC_GRPC_MAX_MESSAGE_MB:-256}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --max-message-mb) MAX_MESSAGE_MB="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "${DATA_ROOT}"
export PYTHONPATH="${SRC_PYTHON}:${PYTHONPATH:-}"

exec python "${SRC_PYTHON}/hypervec_grpc_server.py" \
    --data-root "${DATA_ROOT}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --max-message-mb "${MAX_MESSAGE_MB}"
