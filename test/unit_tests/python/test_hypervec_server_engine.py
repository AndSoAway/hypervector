from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path

import numpy as np


class FakeIndexFlatL2:
    def __init__(self, d: int) -> None:
        self.d = d
        self.is_trained = True
        self.vectors = np.empty((0, d), dtype=np.float32)

    def add(self, x) -> None:
        self.vectors = np.vstack([self.vectors, np.asarray(x, dtype=np.float32)])

    def search(self, x, k: int):
        x = np.asarray(x, dtype=np.float32)
        distances = ((x[:, None, :] - self.vectors[None, :, :]) ** 2).sum(axis=2)
        labels = np.argsort(distances, axis=1)[:, :k].astype(np.int64)
        dists = np.take_along_axis(distances, labels, axis=1).astype(np.float32)
        return dists, labels


class FakeIVFIndex(FakeIndexFlatL2):
    def __init__(self, d: int) -> None:
        super().__init__(d)
        self.last_nprobe = None

    def search_with_nprobe(self, x, k: int, nprobe: int):
        self.last_nprobe = nprobe
        return self.search(x, k)


class FakeHypervec:
    kMetricL2 = 1
    kMetricInnerProduct = 0

    def __init__(self) -> None:
        self.saved_index = None

    def IndexFlatL2(self, d: int):
        return FakeIndexFlatL2(d)

    def IndexFlatIP(self, d: int):
        return FakeIndexFlatL2(d)

    def IndexIVFFlat(self, d: int, nlist: int, metric: int):
        return FakeIVFIndex(d)

    def IndexHNSWFlat(self, d: int, m: int, metric: int):
        return FakeIndexFlatL2(d)

    def IndexPQ(self, d: int, m: int, nbits: int, metric: int):
        return FakeIndexFlatL2(d)

    def IndexIVFPQ(self, d: int, nlist: int, m: int, nbits: int, metric: int):
        return FakeIndexFlatL2(d)

    def IndexHNSWPQ(self, d: int, m_pq: int, nbits: int, m_hnsw: int, metric: int):
        return FakeIndexFlatL2(d)

    def IndexLVQ(self, d: int, nlocal: int, nbits: int, metric: int):
        return FakeIndexFlatL2(d)

    def IndexIVFLVQ(self, d: int, nlist: int, nlocal: int, nbits: int, metric: int):
        return FakeIndexFlatL2(d)

    def IndexHNSWLVQ(self, d: int, nlocal: int, nbits: int, m_hnsw: int, metric: int):
        return FakeIndexFlatL2(d)

    def write_index(self, index, path: str) -> None:
        self.saved_index = index
        Path(path).write_text("fake", encoding="utf-8")

    def read_index(self, path: str):
        return self.saved_index


def load_engine_module():
    module_path = Path(__file__).parents[3] / "src" / "python" / "hypervec_server_engine.py"
    spec = importlib.util.spec_from_file_location("hypervec_server_engine_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hypervec_server_engine_create_insert_flush_load_search(tmp_path):
    module = load_engine_module()
    fake = FakeHypervec()
    engine = module.HypervecServerEngine(str(tmp_path), hypervec_module=fake)

    schema = {
        "auto_id": False,
        "enable_dynamic_field": True,
        "fields": [
            {"name": "id", "datatype": "VARCHAR", "is_primary": True},
            {"name": "vector", "datatype": "FLOAT_VECTOR", "dim": 2},
            {"name": "contents", "datatype": "VARCHAR"},
        ],
    }
    index_params = {
        "indexes": [
            {
                "field_name": "vector",
                "metric_type": "L2",
                "index_type": "Flat",
                "params": {},
            }
        ]
    }

    created = engine.create_collection("demo", schema=schema, index_params=index_params)
    assert created["collection_name"] == "demo"
    assert engine.has_collection("demo")

    inserted = engine.insert(
        "demo",
        [
            {"id": "a", "vector": [0, 0], "contents": "zero", "source": "manual"},
            {"id": "b", "vector": [1, 1], "contents": "one", "source": "manual"},
            {"id": "c", "vector": [10, 10], "contents": "ten", "source": "other"},
        ],
    )
    assert inserted["total"] == 3
    assert engine.get_version("demo")["version"] == 1
    flushed = engine.flush("demo")
    assert flushed["dim"] == 2
    assert flushed["version"] == 2
    assert engine.sync_check("demo", client_version=1)["needs_sync"]
    assert not engine.sync_check("demo", client_version=2)["needs_sync"]

    engine.close_collection("demo")
    loaded = engine.load_collection("demo")
    assert loaded["loaded"]
    assert loaded["version"] == 2

    results = engine.search(
        "demo",
        data=[[0.1, 0.1]],
        limit=2,
        output_fields=["id", "contents", "source"],
        filter="source == 'manual'",
    )
    assert results == [
        [
            {
                "id": "a",
                "distance": results[0][0]["distance"],
                "entity": {"id": "a", "contents": "zero", "source": "manual"},
            },
            {
                "id": "b",
                "distance": results[0][1]["distance"],
                "entity": {"id": "b", "contents": "one", "source": "manual"},
            },
        ]
    ]

    dropped = engine.drop_collection("demo")
    assert dropped["dropped"]
    assert not engine.has_collection("demo")


def test_hypervec_server_engine_uses_explicit_nprobe_after_reload(tmp_path):
    module = load_engine_module()
    fake = FakeHypervec()
    engine = module.HypervecServerEngine(str(tmp_path), hypervec_module=fake)
    schema = {
        "auto_id": False,
        "fields": [
            {"name": "id", "datatype": "VARCHAR", "is_primary": True},
            {"name": "vector", "datatype": "FLOAT_VECTOR", "dim": 2},
        ],
    }
    engine.create_collection(
        "ivf_demo",
        schema=schema,
        index_params={"indexes": [{"field_name": "vector", "metric_type": "L2", "index_type": "IVFFlat", "params": {"nlist": 4}}]},
    )
    engine.insert("ivf_demo", [{"id": "a", "vector": [0, 0]}, {"id": "b", "vector": [1, 1]}])
    engine.flush("ivf_demo")
    engine.close_collection("ivf_demo")
    engine.load_collection("ivf_demo")
    result = engine.search("ivf_demo", data=[[0.1, 0.1]], limit=1, output_fields=["id"], search_params={"nprobe": 4})
    assert result[0][0]["id"] == "a"
    assert fake.saved_index.last_nprobe == 4



def test_hypervec_server_engine_search_runs_concurrently_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERVEC_SEARCH_CONCURRENCY", raising=False)
    module = load_engine_module()

    class BlockingIndex(FakeIndexFlatL2):
        def __init__(self, d: int) -> None:
            super().__init__(d)
            self.entered = 0
            self.entered_cond = threading.Condition()
            self.release = threading.Event()

        def search(self, x, k: int):
            with self.entered_cond:
                self.entered += 1
                self.entered_cond.notify_all()
            assert self.release.wait(timeout=2.0)
            return super().search(x, k)

    class BlockingHypervec(FakeHypervec):
        def __init__(self) -> None:
            super().__init__()
            self.index = BlockingIndex(2)

        def IndexFlatL2(self, d: int):
            return self.index

    fake = BlockingHypervec()
    engine = module.HypervecServerEngine(str(tmp_path), hypervec_module=fake)
    schema = {
        "auto_id": False,
        "fields": [
            {"name": "id", "datatype": "VARCHAR", "is_primary": True},
            {"name": "vector", "datatype": "FLOAT_VECTOR", "dim": 2},
        ],
    }
    engine.create_collection(
        "demo",
        schema=schema,
        index_params={"indexes": [{"field_name": "vector", "metric_type": "L2", "index_type": "Flat"}]},
    )
    engine.insert("demo", [{"id": "a", "vector": [0, 0]}, {"id": "b", "vector": [1, 1]}])
    engine.flush("demo")

    errors = []

    def run_search():
        try:
            engine.search("demo", data=[[0.1, 0.1]], limit=1, output_fields=["id"])
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=run_search), threading.Thread(target=run_search)]
    for thread in threads:
        thread.start()

    deadline = time.time() + 2.0
    with fake.index.entered_cond:
        while fake.index.entered < 2 and time.time() < deadline:
            fake.index.entered_cond.wait(timeout=0.05)

    assert fake.index.entered == 2
    fake.index.release.set()
    for thread in threads:
        thread.join(timeout=2.0)
    assert not errors
