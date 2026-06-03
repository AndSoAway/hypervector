"""
Integration tests for server/ components.
Uses the real swighypervec.pyd via hv.py — requires compiled C++ bindings.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from scalar_store import ScalarStore
from store import MetaStore
from index_manager import IndexManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_stack(tmp: Path):
    store  = MetaStore(tmp / "collections.json")
    scalar = ScalarStore(tmp / "scalar.db")
    mgr    = IndexManager(store, scalar)
    return store, scalar, mgr


# ---------------------------------------------------------------------------
# 1. ScalarStore
# ---------------------------------------------------------------------------

class TestScalarStore(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.sc  = ScalarStore(self.tmp / "scalar.db")

    def test_create_and_retrieve(self):
        self.sc.ensure_table("col1")
        self.sc.insert_batch("col1", [
            (0, "doc_a", "hello world", {"source": "test.pdf", "title": "Doc A"}),
            (1, "doc_b", "foo bar",     {"source": "test.pdf", "title": "Doc B"}),
        ])
        rows = self.sc.get_by_row_ids("col1", [0, 1])
        self.assertEqual(rows[0]["doc_id"], "doc_a")
        self.assertEqual(rows[0]["text_content"], "hello world")
        self.assertEqual(rows[0]["metadata"]["title"], "Doc A")
        self.assertEqual(rows[1]["doc_id"], "doc_b")

    def test_missing_row_returns_none(self):
        self.sc.ensure_table("col1")
        rows = self.sc.get_by_row_ids("col1", [99])
        self.assertIsNone(rows[0])

    def test_drop_table(self):
        self.sc.ensure_table("col1")
        self.sc.insert_batch("col1", [(0, "x", "txt", {})])
        self.sc.drop_table("col1")
        self.assertEqual(self.sc.count("col1"), 0)

    def test_count(self):
        self.sc.ensure_table("c")
        self.sc.insert_batch("c", [(i, f"id{i}", f"text{i}", {}) for i in range(5)])
        self.assertEqual(self.sc.count("c"), 5)


# ---------------------------------------------------------------------------
# 2. MetaStore
# ---------------------------------------------------------------------------

class TestMetaStore(unittest.TestCase):
    def setUp(self):
        self.tmp   = Path(tempfile.mkdtemp())
        self.store = MetaStore(self.tmp / "collections.json")

    def test_create_and_get(self):
        meta = self.store.create("kb1", "/path/to/kb1.index")
        self.assertEqual(meta.name, "kb1")
        self.assertEqual(meta.version, 1)
        self.assertIsNotNone(self.store.get("kb1"))

    def test_bump_version(self):
        self.store.create("kb1", "/path/kb1.index")
        m = self.store.bump_version("kb1")
        self.assertEqual(m.version, 2)
        m = self.store.bump_version("kb1")
        self.assertEqual(m.version, 3)

    def test_delete(self):
        self.store.create("kb1", "/path/kb1.index")
        self.assertTrue(self.store.delete("kb1"))
        self.assertIsNone(self.store.get("kb1"))

    def test_persist_and_reload(self):
        self.store.create("kb1", "/path/kb1.index")
        self.store.bump_version("kb1")
        store2 = MetaStore(self.tmp / "collections.json")
        m = store2.get("kb1")
        self.assertEqual(m.version, 2)


# ---------------------------------------------------------------------------
# 3. IndexManager (real HyperVec index)
# ---------------------------------------------------------------------------

class TestIndexManager(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store, self.scalar, self.mgr = make_stack(self.tmp)

    def tearDown(self):
        # Clean up any index files written to server/indexes/
        import index_manager as im
        for p in im._INDEX_DIR.glob("*.index"):
            p.unlink(missing_ok=True)

    def test_create_and_add(self):
        self.mgr.create_collection("kb1", dim=4)
        vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        meta = self.mgr.add_vectors(
            "kb1", vecs,
            doc_ids=["d0", "d1"],
            texts=["text zero", "text one"],
            metadatas=[{"source": "a.pdf"}, {"source": "b.pdf"}],
        )
        self.assertEqual(meta.version, 2)
        self.assertEqual(self.scalar.count("kb1"), 2)

    def test_ntotal_matches_added(self):
        self.mgr.create_collection("kb1", dim=4)
        vecs = np.eye(4, dtype=np.float32)
        self.mgr.add_vectors("kb1", vecs, ["d0","d1","d2","d3"],
                             ["t0","t1","t2","t3"], [{},{},{},{}])
        idx = self.mgr._indexes["kb1"]
        self.assertEqual(idx.ntotal, 4)

    def test_search_returns_metadata(self):
        self.mgr.create_collection("kb1", dim=4)
        vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.mgr.add_vectors("kb1", vecs,
                             doc_ids=["d0", "d1"],
                             texts=["text zero", "text one"],
                             metadatas=[{"source": "a.pdf"}, {"source": "b.pdf"}])
        q = np.array([[1, 0, 0, 0]], dtype=np.float32)
        results = self.mgr.search_with_meta("kb1", q, top_k=2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0]["doc_id"], "d0")
        self.assertAlmostEqual(results[0][0]["score"], 0.0, places=5)
        self.assertEqual(results[0][0]["metadata"]["source"], "a.pdf")

    def test_search_nearest_is_exact_match(self):
        self.mgr.create_collection("kb1", dim=4)
        vecs = np.eye(4, dtype=np.float32)
        self.mgr.add_vectors("kb1", vecs, [f"d{i}" for i in range(4)],
                             [f"t{i}" for i in range(4)], [{} for _ in range(4)])
        for i in range(4):
            q = vecs[i:i+1]
            results = self.mgr.search_with_meta("kb1", q, top_k=1)
            self.assertEqual(results[0][0]["doc_id"], f"d{i}")
            self.assertAlmostEqual(results[0][0]["score"], 0.0, places=5)

    def test_rebuild_resets_rows(self):
        self.mgr.create_collection("kb1", dim=4)
        vecs_old = np.array([[1, 0, 0, 0]], dtype=np.float32)
        self.mgr.add_vectors("kb1", vecs_old, ["old"], ["old text"], [{}])

        vecs_new = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        meta = self.mgr.rebuild("kb1", vecs_new,
                                ["n0", "n1"], ["new0", "new1"], [{}, {}])
        self.assertEqual(meta.version, 3)
        self.assertEqual(self.scalar.count("kb1"), 2)
        idx = self.mgr._indexes["kb1"]
        self.assertEqual(idx.ntotal, 2)

    def test_persist_and_reload(self):
        self.mgr.create_collection("kb1", dim=4)
        vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.mgr.add_vectors("kb1", vecs, ["d0", "d1"], ["t0", "t1"], [{}, {}])

        # Reload from disk using the same MetaStore and ScalarStore
        mgr2 = IndexManager(self.store, self.scalar)
        idx2 = mgr2._indexes.get("kb1")
        self.assertIsNotNone(idx2)
        self.assertEqual(idx2.ntotal, 2)

    def test_version_increments_on_each_add(self):
        self.mgr.create_collection("kb1", dim=2)
        v = np.ones((1, 2), dtype=np.float32)
        m1 = self.mgr.add_vectors("kb1", v, ["a"], ["t"], [{}])
        m2 = self.mgr.add_vectors("kb1", v, ["b"], ["t"], [{}])
        self.assertEqual(m1.version, 2)
        self.assertEqual(m2.version, 3)

    def test_delete_collection(self):
        self.mgr.create_collection("kb1", dim=2)
        self.assertTrue(self.mgr.delete_collection("kb1"))
        self.assertIsNone(self.mgr.get_meta("kb1"))
        self.assertEqual(self.scalar.count("kb1"), 0)

    def test_search_missing_collection_raises(self):
        q = np.ones((1, 4), dtype=np.float32)
        with self.assertRaises(KeyError):
            self.mgr.search_with_meta("nonexistent", q, top_k=1)


# ---------------------------------------------------------------------------
# 4. FastAPI routes (via TestClient)
# ---------------------------------------------------------------------------

class TestAPI(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.tmp = Path(tempfile.mkdtemp())
        import main
        main._store   = MetaStore(self.tmp / "collections.json")
        main._scalar  = ScalarStore(self.tmp / "scalar.db")
        main._manager = IndexManager(main._store, main._scalar)
        self.client = TestClient(main.app)

    def tearDown(self):
        import index_manager as im
        for p in im._INDEX_DIR.glob("*.index"):
            p.unlink(missing_ok=True)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_create_and_list(self):
        r = self.client.post("/collections", json={"name": "kb1", "dim": 4})
        self.assertEqual(r.status_code, 201)
        data = r.json()
        self.assertEqual(data["name"], "kb1")
        self.assertEqual(data["version"], 1)

        r2 = self.client.get("/collections")
        self.assertEqual(len(r2.json()), 1)

    def test_version_polling(self):
        self.client.post("/collections", json={"name": "kb2", "dim": 4})
        r = self.client.get("/collections/kb2/version")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["version"], 1)

    def test_version_not_found(self):
        r = self.client.get("/collections/nope/version")
        self.assertEqual(r.status_code, 404)

    def test_sync_check_needs_sync(self):
        self.client.post("/collections", json={"name": "kb3", "dim": 4})
        r = self.client.post("/collections/kb3/sync-check",
                             json={"client_version": 0})
        self.assertTrue(r.json()["needs_sync"])
        self.assertEqual(r.json()["server_version"], 1)

    def test_sync_check_no_sync(self):
        self.client.post("/collections", json={"name": "kb4", "dim": 4})
        r = self.client.post("/collections/kb4/sync-check",
                             json={"client_version": 1})
        self.assertFalse(r.json()["needs_sync"])

    def test_delete(self):
        self.client.post("/collections", json={"name": "kb5", "dim": 4})
        r = self.client.delete("/collections/kb5")
        self.assertEqual(r.status_code, 204)
        r2 = self.client.get("/collections/kb5/version")
        self.assertEqual(r2.status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
