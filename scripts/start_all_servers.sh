#!/usr/bin/env bash
# 同时启动 HyperVec gRPC server 和 HTTP server，两者共享同一个 data-root
#
# 用法：
#   ./scripts/start_all_servers.sh [--data-root DIR] [--grpc-port PORT] [--http-port PORT]
#
# 默认值：
#   --data-root   ./data
#   --grpc-port   50051
#   --http-port   8080

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_PYTHON="${REPO_ROOT}/src/python"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
GRPC_PORT="${GRPC_PORT:-50051}"
HTTP_PORT="${HTTP_PORT:-8080}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)  DATA_ROOT="$2";  shift 2 ;;
        --grpc-port)  GRPC_PORT="$2";  shift 2 ;;
        --http-port)  HTTP_PORT="$2";  shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

mkdir -p "${DATA_ROOT}"

echo "启动 gRPC server  → 0.0.0.0:${GRPC_PORT}"
echo "启动 HTTP server  → 0.0.0.0:${HTTP_PORT}"
echo "共享 data-root    → ${DATA_ROOT}"
echo ""

# 启动 gRPC server（后台）
PYTHONPATH="${SRC_PYTHON}:${PYTHONPATH:-}" \
python "${SRC_PYTHON}/hypervec_grpc_server.py" \
    --data-root "${DATA_ROOT}" \
    --port "${GRPC_PORT}" &
GRPC_PID=$!

# 启动 HTTP server（前台，方便日志输出）
PYTHONPATH="${SRC_PYTHON}:${PYTHONPATH:-}" \
uvicorn hypervec_http_server:app \
    --app-dir "${SRC_PYTHON}" \
    --host 0.0.0.0 \
    --port "${HTTP_PORT}" &
HTTP_PID=$!

# 收到 SIGINT / SIGTERM 时一起关闭
trap "echo '正在关闭...'; kill ${GRPC_PID} ${HTTP_PID} 2>/dev/null; exit 0" INT TERM

echo "两个 server 已启动，按 Ctrl+C 退出"
wait
