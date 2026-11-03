#!/usr/bin/env bash
# Start the HyperVec HTTP server with the manually-built BM25 SWIG module
# on THIS machine (CMake too old for the official build; _swighypervec.so was
# built by hand into /data/pybuild_t2). For the normal CMake build, use
# scripts/start_all_servers.sh instead.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-/data/pybuild_t2}"          # holds _swighypervec.so + swighypervec.py
DATA_ROOT="${DATA_ROOT:-/data/pybuild_t2/server_data}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8099}"

mkdir -p "${DATA_ROOT}" "${BUILD_DIR}/tmp"

export PYTHONPATH="${BUILD_DIR}:${REPO_ROOT}/src/python"
export HYPERVEC_DICT_DIR="${HYPERVEC_DICT_DIR:-${REPO_ROOT}/third_party/cppjieba/dict}"
# This box's /usr/local python3.10 lacks a new enough sqlite; preload conda's.
export LD_PRELOAD="${LD_PRELOAD:-/root/miniconda3/lib/libsqlite3.so.0}"
export TMPDIR="${BUILD_DIR}/tmp"
export DATA_ROOT HOST PORT

echo "HyperVec HTTP server -> http://${HOST}:${PORT}  (data=${DATA_ROOT})"
exec python3 - <<'PY'
import os, swighypervec as hv, uvicorn
from hypervec_server_engine import HypervecServerEngine
from hypervec_http_server import create_app
engine = HypervecServerEngine(os.environ["DATA_ROOT"], hypervec_module=hv)
app = create_app(data_root=os.environ["DATA_ROOT"], engine=engine)
uvicorn.run(app, host=os.environ["HOST"], port=int(os.environ["PORT"]), log_level="warning")
PY
