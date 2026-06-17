"""
pytest fixtures for integration tests that spin up in-process servers.

These fixtures are module-scoped: one server pair per test module that
requests them.  Tests that don't import grpc_port or http_port are
unaffected.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import pytest

# Bypass system proxy for localhost (avoids 502 errors on some networks)
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

_REPO_ROOT = Path(__file__).parents[3]
_SRC_PYTHON = _REPO_ROOT / "src" / "python"
_PYHYPERVEC = _REPO_ROOT / "pyhypervec"

for _p in [str(_SRC_PYTHON), str(_PYHYPERVEC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# FakeHypervec — mirrors FakeHypervec in test_hypervec_server_engine.py.
# Lets integration tests run without a compiled C++ extension.
# ---------------------------------------------------------------------------

import numpy as _np


class _FakeIndex:
    def __init__(self, d: int) -> None:
        self.d = d
        self.is_trained = True
        self._vecs = _np.empty((0, d), dtype=_np.float32)

    def add(self, x) -> None:
        self._vecs = _np.vstack([self._vecs, _np.asarray(x, dtype=_np.float32)])

    def search(self, x, k: int):
        x = _np.asarray(x, dtype=_np.float32)
        dists = (((x[:, None, :] - self._vecs[None, :, :]) ** 2).sum(axis=2))
        labels = _np.argsort(dists, axis=1)[:, :k].astype(_np.int64)
        d_out = _np.take_along_axis(dists, labels, axis=1).astype(_np.float32)
        return d_out, labels


class _FakeHypervec:
    kMetricL2 = 1
    kMetricInnerProduct = 0

    def __init__(self) -> None:
        self._last_index = None

    def IndexFlatL2(self, d):
        return _FakeIndex(d)

    def IndexFlatIP(self, d):
        return _FakeIndex(d)

    def IndexHNSWFlat(self, d, m, metric=1):
        return _FakeIndex(d)

    def IndexHNSWLVQ(self, d, nlocal, nbits, m, metric=1):
        return _FakeIndex(d)

    def write_index(self, index, path: str) -> None:
        self._last_index = index
        Path(path).write_bytes(b"fake-index")

    def read_index(self, path: str):
        return self._last_index


# ---------------------------------------------------------------------------
# In-process gRPC server
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grpc_server_engine(tmp_path_factory):
    """HypervecServerEngine backed by FakeHypervec and a temp directory.

    Uses a _RetryMetaStore subclass to work around Windows file-rename
    locking that can occur when many tests share one collections.json.
    """
    import time as _time
    from hypervec_server_engine import HypervecServerEngine
    from hypervec_meta_store import MetaStore
    from hypervec_scalar_store import ScalarStore

    class _RetryMetaStore(MetaStore):
        def _save(self):
            for attempt in range(10):
                try:
                    super()._save()
                    return
                except (PermissionError, OSError):
                    _time.sleep(0.02 * (attempt + 1))
            super()._save()

    data_root = tmp_path_factory.mktemp("grpc_data")
    meta_store = _RetryMetaStore(data_root / "collections.json")
    scalar_store = ScalarStore(data_root / "scalar.db")
    return HypervecServerEngine(
        str(data_root),
        hypervec_module=_FakeHypervec(),
        meta_store=meta_store,
        scalar_store=scalar_store,
    )


@pytest.fixture(scope="module")
def grpc_port(grpc_server_engine) -> Iterator[int]:
    """Start an in-process gRPC server; yield its port."""
    try:
        import grpc
        from concurrent.futures import ThreadPoolExecutor
        import hypervec_pb2_grpc as pb2_grpc
        from hypervec_grpc_server import HyperVecServicer
    except ImportError as exc:
        pytest.skip(f"gRPC server not importable: {exc}")
        return

    port = _free_port()
    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_HyperVecServicer_to_server(HyperVecServicer(grpc_server_engine), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    yield port
    server.stop(grace=0).wait()


# ---------------------------------------------------------------------------
# In-process HTTP server (uvicorn)
# ---------------------------------------------------------------------------

class _UvicornThread(threading.Thread):
    def __init__(self, app, host: str, port: int) -> None:
        super().__init__(daemon=True)
        self._app = app
        self._host = host
        self._port = port
        self.server = None

    def run(self) -> None:
        import asyncio
        import uvicorn
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        config = uvicorn.Config(
            self._app, host=self._host, port=self._port,
            log_level="error", loop="asyncio",
        )
        self.server = uvicorn.Server(config)
        loop.run_until_complete(self.server.serve())
        loop.close()

    def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        self.join(timeout=5)


@pytest.fixture(scope="module")
def http_port(grpc_server_engine) -> Iterator[int]:
    """Start an in-process uvicorn HTTP server sharing grpc_server_engine."""
    try:
        import uvicorn  # noqa: F401
        import importlib.util as _ilu

        # hypervec_http_server uses relative imports, so it must be loaded
        # as part of a "hypervec" package — same trick as test_hypervec_http_server.py.
        _pkg = type(sys)("hypervec")
        _pkg.__path__ = [str(_SRC_PYTHON)]
        sys.modules.setdefault("hypervec", _pkg)

        spec = _ilu.spec_from_file_location(
            "hypervec.hypervec_http_server",
            _SRC_PYTHON / "hypervec_http_server.py",
            submodule_search_locations=[str(_SRC_PYTHON)],
        )
        http_mod = _ilu.module_from_spec(spec)
        sys.modules["hypervec.hypervec_http_server"] = http_mod
        assert spec.loader is not None
        spec.loader.exec_module(http_mod)
        create_app = http_mod.create_app
    except ImportError as exc:
        pytest.skip(f"HTTP server not importable: {exc}")
        return

    app = create_app(data_root="unused", engine=grpc_server_engine)
    port = _free_port()
    t = _UvicornThread(app, "127.0.0.1", port)
    t.start()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        t.stop()
        pytest.skip("HTTP server did not start in time")
        return

    yield port
    t.stop()
