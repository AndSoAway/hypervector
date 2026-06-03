#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build-arm}"
VENV_DIR="${VENV_DIR:-.venv}"
OPT_LEVEL="${HYPERVEC_OPT_LEVEL:-generic}"
GENERATOR="${CMAKE_GENERATOR:-Ninja}"
INSTALL_PYHYPERVEC="${INSTALL_PYHYPERVEC:-1}"
INSTALL_SERVER_DEPS="${INSTALL_SERVER_DEPS:-1}"

cd "${ROOT_DIR}"

echo "[hypervec] source root: ${ROOT_DIR}"
echo "[hypervec] build dir: ${BUILD_DIR}"
echo "[hypervec] venv dir: ${VENV_DIR}"
echo "[hypervec] opt level: ${OPT_LEVEL}"

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

if ! command -v python3 >/dev/null 2>&1; then
  echo "[hypervec] python3 is required." >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "[hypervec] cmake is required." >&2
  exit 1
fi

if [[ "${GENERATOR}" == "Ninja" ]] && ! command -v ninja >/dev/null 2>&1; then
  echo "[hypervec] ninja is required when CMAKE_GENERATOR=Ninja." >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[hypervec] creating virtualenv..."
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "[hypervec] upgrading Python build dependencies..."
python -m pip install --upgrade pip setuptools wheel packaging numpy

if [[ "${INSTALL_SERVER_DEPS}" == "1" ]]; then
  echo "[hypervec] installing server dependencies..."
  python -m pip install fastapi uvicorn
fi

echo "[hypervec] configuring CMake..."
cmake -S . -B "${BUILD_DIR}" \
  -G "${GENERATOR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DHYPERVEC_ENABLE_PYTHON=ON \
  -DHYPERVEC_ENABLE_EXTRAS=OFF \
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
python setup.py bdist_wheel
python -m pip install --force-reinstall dist/*.whl
popd >/dev/null

if [[ "${INSTALL_PYHYPERVEC}" == "1" ]]; then
  echo "[hypervec] installing pyhypervec client package..."
  python -m pip install ./pyhypervec
fi

echo "[hypervec] verifying installation..."
python - <<'PY'
import hypervec
print("hypervec:", hypervec.__file__)
print("has IndexHNSWFlat:", hasattr(hypervec, "IndexHNSWFlat"))
import hypervec.hypervec_http_server as server
print("server module:", server.__file__)
from pyhypervec import HypervecClient, DataType
print("pyhypervec:", HypervecClient, DataType.FLOAT_VECTOR)
PY

cat <<EOF

[hypervec] build complete.

Activate the environment:
  source ${VENV_DIR}/bin/activate

Start the server:
  python -m hypervec.hypervec_http_server --data-root /data/hypervec --host 0.0.0.0 --port 8080

EOF
