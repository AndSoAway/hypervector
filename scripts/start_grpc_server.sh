#!/usr/bin/env bash
# 启动 HyperVec gRPC server
#
# 用法：
#   ./scripts/start_grpc_server.sh [--data-root DIR] [--port PORT] [--host HOST] [--workers N]
#
# 默认值：
#   --data-root  ./data
#   --port       50051
#   --host       0.0.0.0
#   --workers    10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_PYTHON="${REPO_ROOT}/src/python"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
PORT="${PORT:-50051}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${WORKERS:-10}"

# 允许通过命令行参数覆盖
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --port)      PORT="$2";      shift 2 ;;
        --host)      HOST="$2";      shift 2 ;;
        --workers)   WORKERS="$2";   shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

mkdir -p "${DATA_ROOT}"

echo "启动 gRPC server"
echo "  data-root : ${DATA_ROOT}"
echo "  host      : ${HOST}"
echo "  port      : ${PORT}"
echo "  workers   : ${WORKERS}"
echo ""

exec python "${SRC_PYTHON}/hypervec_grpc_server.py" \
    --data-root "${DATA_ROOT}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}"
