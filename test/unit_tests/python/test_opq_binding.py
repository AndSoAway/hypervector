from __future__ import annotations

import numpy as np
import pytest

import hypervec


def training_data() -> np.ndarray:
    rows = np.arange(64, dtype=np.float32)[:, None]
    columns = np.arange(4, dtype=np.float32)[None, :]
    return ((rows * (columns + 3) + columns * columns) % 37 - 18) / 9


def make_index(use_id_map: bool = False):
    return hypervec.create_index(
        "opq_pq",
        4,
        hypervec.kMetricL2,
        {"m_pq": 2, "nbits": 2, "opq_iterations": 2},
        use_id_map,
    )


def test_opq_pq_factory_builds_and_searches():
    index = make_index()
    database = training_data().astype(np.float32)
    index.build(database)

    distances, labels = index.search(database[:2], 3)
    assert len(distances) == 2
    assert len(labels) == 2
    assert all(len(row) == 3 for row in labels)
    assert all(0 <= label < len(database) for row in labels for label in row)
    assert np.isfinite(np.asarray(distances)).all()
    assert (np.asarray(distances) >= 0).all()


def test_opq_pq_alias_and_id_map_composition():
    alias = hypervec.create_index(
        "IndexOPQPQ",
        4,
        hypervec.kMetricL2,
        {"m_pq": 2, "nbits": 2, "opq_iterations": 1},
    )
    assert isinstance(alias, hypervec.Index)

    mapped = make_index(use_id_map=True)
    database = training_data().astype(np.float32)
    mapped.build(database)
    distances, labels = mapped.search(database[:1], 2)
    assert len(distances[0]) == 2
    assert all(0 <= label < len(database) for label in labels[0])


def test_opq_pq_factory_rejects_invalid_configuration():
    with pytest.raises(RuntimeError, match="kMetricL2"):
        hypervec.create_index(
            "opq_pq", 4, hypervec.kMetricInnerProduct, {"m_pq": 2}
        )
    with pytest.raises(RuntimeError, match="divisible"):
        hypervec.create_index(
            "opq_pq", 4, hypervec.kMetricL2, {"m_pq": 3, "nbits": 2}
        )
    with pytest.raises(RuntimeError, match="opq_iterations"):
        hypervec.create_index(
            "opq_pq",
            4,
            hypervec.kMetricL2,
            {"m_pq": 2, "nbits": 2, "opq_iterations": 0},
        )
    with pytest.raises(RuntimeError, match="unknown parameter"):
        hypervec.create_index(
            "opq_pq", 4, hypervec.kMetricL2, {"m_pq": 2, "typo": 1}
        )
