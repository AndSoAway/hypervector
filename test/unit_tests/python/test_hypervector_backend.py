from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


class FakeIndexFlatL2:
    def __init__(self, d: int) -> None:
        self.d = d
        self.is_trained = True
        self.vectors = np.empty((0, d), dtype=np.float32)

    def Add(self, x: np.ndarray) -> None:
        self.vectors = np.vstack([self.vectors, np.asarray(x, dtype=np.float32)])

    def Search(self, x: np.ndarray, k: int):
        x = np.asarray(x, dtype=np.float32)
        distances = ((x[:, None, :] - self.vectors[None, :, :]) ** 2).sum(axis=2)
        labels = np.argsort(distances, axis=1)[:, :k].astype(np.int64)
        dists = np.take_along_axis(distances, labels, axis=1).astype(np.float32)
        return dists, labels


def load_backend_module():
    fake_hypervec = types.ModuleType("hypervec")
    fake_hypervec.kMetricL2 = 1
    fake_hypervec.kMetricInnerProduct = 0
    fake_hypervec.IndexFlatL2 = FakeIndexFlatL2
    fake_hypervec.IndexFlatIP = FakeIndexFlatL2
    fake_hypervec.write_index = lambda index, path: Path(path).write_text("fake")
    fake_hypervec.read_index = lambda path: fake_hypervec._loaded_index
    sys.modules["hypervec"] = fake_hypervec

    module_path = Path(__file__).parents[3] / "src" / "python" / "hypervector_backend.py"
    spec = importlib.util.spec_from_file_location("hypervector_backend_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, fake_hypervec


def test_hypervector_backend_build_search_and_metadata(tmp_path):
    if np is None:
        return

    module, fake_hypervec = load_backend_module()
    backend = module.HyperVectorIndexBackend(
        contents=[],
        config={
            "uri": str(tmp_path),
            "collection_name": "demo",
            "metric_type": "L2",
            "index_params": {"index_type": "Flat"},
        },
        logger=logging.getLogger("test"),
    )

    embeddings = np.array([[0, 0], [1, 1], [10, 10]], dtype=np.float32)
    ids = np.array(["a", "b", "c"])
    contents = ["zero", "one", "ten"]
    metadatas = [
        {"source": "manual", "doc_id": "0"},
        {"source": "manual", "doc_id": "1"},
        {"source": "other", "doc_id": "2"},
    ]

    backend.build_index(
        embeddings=embeddings,
        ids=ids,
        contents=contents,
        metadatas=metadatas,
        overwrite=True,
    )
    fake_hypervec._loaded_index = backend.index

    docs = backend.search(np.array([[0.1, 0.1]], dtype=np.float32), 2)
    assert docs == [["zero", "one"]]

    rows = backend.search_with_meta(
        np.array([[9.9, 9.9]], dtype=np.float32),
        1,
        filter="source == 'other'",
    )
    assert rows[0][0]["content"] == "ten"
    assert rows[0][0]["chunk_id"] == "c"
    assert rows[0][0]["metadata"]["source"] == "other"

    reloaded = module.HyperVectorIndexBackend(
        contents=[],
        config={"uri": str(tmp_path), "collection_name": "demo"},
        logger=logging.getLogger("test"),
    )
    reloaded.load_index()
    assert len(reloaded.rows) == 3
