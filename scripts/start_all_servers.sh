#!/usr/bin/env bash
# Start HTTP and gRPC in one process with one shared HypervecServerEngine.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_PYTHON="${REPO_ROOT}/src/python"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
HTTP_HOST="${HTTP_HOST:-0.0.0.0}"
HTTP_PORT="${HTTP_PORT:-8080}"
GRPC_HOST="${GRPC_HOST:-0.0.0.0}"
GRPC_PORT="${GRPC_PORT:-50051}"
GRPC_WORKERS="${GRPC_WORKERS:-10}"
MAX_MESSAGE_MB="${HYPERVEC_GRPC_MAX_MESSAGE_MB:-256}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --http-host) HTTP_HOST="$2"; shift 2 ;;
        --http-port) HTTP_PORT="$2"; shift 2 ;;
        --grpc-host) GRPC_HOST="$2"; shift 2 ;;
        --grpc-port) GRPC_PORT="$2"; shift 2 ;;
        --grpc-workers) GRPC_WORKERS="$2"; shift 2 ;;
        --grpc-max-message-mb) MAX_MESSAGE_MB="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "${DATA_ROOT}"
export PYTHONPATH="${SRC_PYTHON}:${PYTHONPATH:-}"

exec python "${SRC_PYTHON}/hypervec_dual_server.py" \
    --data-root "${DATA_ROOT}" \
    --http-host "${HTTP_HOST}" \
    --http-port "${HTTP_PORT}" \
    --grpc-host "${GRPC_HOST}" \
    --grpc-port "${GRPC_PORT}" \
    --grpc-workers "${GRPC_WORKERS}" \
    --grpc-max-message-mb "${MAX_MESSAGE_MB}"
