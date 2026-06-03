"""
Thin adapter over the raw swighypervec SWIG bindings.
Provides a FAISS-compatible interface (add, search, ntotal, reset,
read_index, write_index) so index_manager.py doesn't need to know
about SWIG pointer mechanics.
"""

import os, sys
import numpy as np

# Make MinGW DLLs findable on Windows
_MSYS2_BIN = r"C:\msys64\mingw64\bin"
if os.path.isdir(_MSYS2_BIN):
    try:
        os.add_dll_directory(_MSYS2_BIN)
    except AttributeError:
        os.environ["PATH"] = _MSYS2_BIN + ";" + os.environ.get("PATH", "")

# Point Python at the build output directory
_BUILD_PYTHON = os.path.join(
    os.path.dirname(__file__), "..", "build3", "src", "python"
)
if os.path.isdir(_BUILD_PYTHON):
    sys.path.insert(0, os.path.abspath(_BUILD_PYTHON))

import swighypervec as _hv  # noqa: E402


class _IndexWrapper:
    """Wraps a raw swighypervec Index to expose add/search/ntotal/reset."""

    def __init__(self, raw):
        self._raw = raw

    @property
    def d(self) -> int:
        return self._raw.d

    @property
    def ntotal(self) -> int:
        return int(self._raw.n_total)

    def add(self, x: np.ndarray) -> None:
        x = np.ascontiguousarray(x, dtype=np.float32)
        self._raw.Add(x.shape[0], x.ctypes.data)

    def reset(self) -> None:
        self._raw.Reset()

    def search(self, x: np.ndarray, k: int):
        x = np.ascontiguousarray(x, dtype=np.float32)
        n = x.shape[0]
        D = np.empty((n, k), dtype=np.float32)
        I = np.empty((n, k), dtype=np.int64)
        self._raw.Search(n, x.ctypes.data, k, D.ctypes.data, I.ctypes.data)
        return D, I

    @property
    def _raw_index(self):
        return self._raw


def IndexFlatL2(dim: int) -> _IndexWrapper:
    return _IndexWrapper(_hv.IndexFlatL2(dim))


def IndexHNSWFlat(dim: int, M: int = 32) -> _IndexWrapper:
    return _IndexWrapper(_hv.IndexHNSWFlat(dim, M))


def read_index(path: str) -> _IndexWrapper:
    return _IndexWrapper(_hv.ReadIndex(path))


def write_index(index: _IndexWrapper, path: str) -> None:
    _hv.WriteIndex(index._raw_index, path)
