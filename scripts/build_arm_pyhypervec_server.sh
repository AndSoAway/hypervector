#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build-arm}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OPT_LEVEL="${HYPERVEC_OPT_LEVEL:-generic}"
GENERATOR="${CMAKE_GENERATOR:-Ninja}"
INSTALL_PYHYPERVEC="${INSTALL_PYHYPERVEC:-1}"
INSTALL_SERVER_DEPS="${INSTALL_SERVER_DEPS:-1}"
INSTALL_CMAKE="${INSTALL_CMAKE:-1}"
DATA_ROOT="${DATA_ROOT:-${HOME}/hypervec_data}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8080}"
SERVER_IMPL="${SERVER_IMPL:-hypercorn}"
START_SERVER="${START_SERVER:-0}"

clean_python_build_artifacts() {
  local package_dir="$1"

  if [[ ! -d "${package_dir}" ]]; then
    echo "[hypervec] Python package directory '${package_dir}' was not found." >&2
    exit 1
  fi

  echo "[hypervec] cleaning stale Python build artifacts in ${package_dir}..."
  rm -rf "${package_dir}/build" "${package_dir}"/*.egg-info
}

cd "${ROOT_DIR}"

echo "[hypervec] source root: ${ROOT_DIR}"
echo "[hypervec] build dir: ${BUILD_DIR}"
echo "[hypervec] venv dir: ${VENV_DIR}"
echo "[hypervec] python bin: ${PYTHON_BIN}"
echo "[hypervec] opt level: ${OPT_LEVEL}"
echo "[hypervec] data root: ${DATA_ROOT}"
echo "[hypervec] server: ${SERVER_HOST}:${SERVER_PORT}"
echo "[hypervec] server impl: ${SERVER_IMPL}"

if command -v uname >/dev/null 2>&1; then
  ARCH="$(uname -m)"
  echo "[hypervec] machine arch: ${ARCH}"
  case "${ARCH}" in
    aarch64|arm64) ;;
    *)
      echo "[hypervec] warning: expected ARM64/AArch64, got '${ARCH}'." >&2
      ;;
  esac
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[hypervec] Python executable '${PYTHON_BIN}' was not found." >&2
  exit 1
fi

PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
PYTHON_MAJOR="$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info[0])')"
PYTHON_MINOR="$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info[1])')"
echo "[hypervec] python version: ${PYTHON_VERSION}"
if (( PYTHON_MAJOR < 3 || (PYTHON_MAJOR == 3 && PYTHON_MINOR < 10) )); then
  echo "[hypervec] Python >= 3.10 is required; got ${PYTHON_VERSION}." >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[hypervec] creating virtualenv..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "[hypervec] upgrading Python build dependencies..."
python -m pip install --upgrade pip setuptools wheel packaging numpy build

if [[ "${INSTALL_CMAKE}" == "1" ]]; then
  echo "[hypervec] installing CMake into virtualenv..."
  python -m pip install "cmake>=3.24"
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "[hypervec] cmake is required." >&2
  exit 1
fi

CMAKE_VERSION="$(cmake --version | head -1 | awk '{print $3}')"
echo "[hypervec] cmake version: ${CMAKE_VERSION}"
python - "${CMAKE_VERSION}" <<'PY'
import sys
from packaging.version import Version
if Version(sys.argv[1]) < Version("3.24"):
    raise SystemExit(f"cmake >= 3.24 is required; got {sys.argv[1]}")
PY

if [[ "${INSTALL_SERVER_DEPS}" == "1" ]]; then
  echo "[hypervec] installing server dependencies..."
  python -m pip install fastapi uvicorn hypercorn h2
fi

if [[ "${GENERATOR}" == "Ninja" ]] && ! command -v ninja >/dev/null 2>&1; then
  echo "[hypervec] ninja is required when CMAKE_GENERATOR=Ninja." >&2
  exit 1
fi

if ! command -v swig >/dev/null 2>&1; then
  echo "[hypervec] swig is required." >&2
  exit 1
fi
echo "[hypervec] swig: $(command -v swig)"
swig -version | head -2

echo "[hypervec] configuring CMake..."
cmake -S . -B "${BUILD_DIR}" \
  -G "${GENERATOR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DHYPERVEC_ENABLE_PYTHON=ON \
  -DHYPERVEC_ENABLE_EXTRAS=OFF \
  -DBUILD_TESTING=OFF \
  -DHYPERVEC_OPT_LEVEL="${OPT_LEVEL}" \
  -DPython_EXECUTABLE="$(command -v python)"

echo "[hypervec] building hypervec core..."
cmake --build "${BUILD_DIR}" --target hypervec -j"$(nproc)"

echo "[hypervec] building Python binding..."
cmake --build "${BUILD_DIR}" --target swighypervec -j"$(nproc)"

if [[ "${OPT_LEVEL}" == "sve" ]]; then
  echo "[hypervec] building SVE Python binding..."
  cmake --build "${BUILD_DIR}" --target swighypervec_sve -j"$(nproc)"
fi

echo "[hypervec] building hypervec wheel..."
pushd "${BUILD_DIR}/src/python" >/dev/null
rm -rf dist
python -m build --wheel --no-isolation
python -m pip install --force-reinstall --no-deps dist/*.whl
popd >/dev/null

if [[ "${INSTALL_PYHYPERVEC}" == "1" ]]; then
  echo "[hypervec] installing pyhypervec client package..."
  clean_python_build_artifacts "./pyhypervec"
  python -m pip install ./pyhypervec
fi

mkdir -p "${DATA_ROOT}"

echo "[hypervec] verifying installation..."
python - <<'PY'
import tempfile

import hypervec
print("hypervec:", hypervec.__file__)
print("has IndexHNSWFlat:", hasattr(hypervec, "IndexHNSWFlat"))

import hypervec.hypervec_index_io as index_io
import hypervec.hypervec_meta_store as meta_store
import hypervec.hypervec_scalar_store as scalar_store
import hypervec.hypervec_http_server as server
import hypercorn
import h2
print("index io module:", index_io.__file__)
print("meta store module:", meta_store.__file__)
print("scalar store module:", scalar_store.__file__)
print("server module:", server.__file__)
print("hypercorn:", hypercorn.__file__)
print("h2:", h2.__file__)

app = server.create_app(data_root=tempfile.mkdtemp())
print("server app:", app.title)

from pyhypervec import HypervecClient, DataType
required_methods = [
    "get_version",
    "sync_check",
    "download_index",
    "upload_index",
]
missing = [name for name in required_methods if not hasattr(HypervecClient, name)]
if missing:
    raise RuntimeError(f"missing HypervecClient methods: {missing}")
print("pyhypervec:", HypervecClient, DataType.FLOAT_VECTOR)
HypervecClient("http://127.0.0.1:8080", http2=True)
HypervecClient("https://127.0.0.1:8443", http2=True)
print("pyhypervec sync methods: ok")
print("pyhypervec http2 mode: ok")
PY

if [[ "${START_SERVER}" == "1" ]]; then
  echo "[hypervec] starting HyperVec HTTP server..."
  exec python -m hypervec.hypervec_http_server \
    --data-root "${DATA_ROOT}" \
    --host "${SERVER_HOST}" \
    --port "${SERVER_PORT}" \
    --server "${SERVER_IMPL}"
fi

cat <<EOF

[hypervec] build complete.

Activate the environment:
  source ${VENV_DIR}/bin/activate

Start the server:
  python -m hypervec.hypervec_http_server --data-root ${DATA_ROOT} --host ${SERVER_HOST} --port ${SERVER_PORT} --server ${SERVER_IMPL}

Build and start in one command:
  START_SERVER=1 PYTHON_BIN=${PYTHON_BIN} bash scripts/build_arm_pyhypervec_server.sh

EOF
