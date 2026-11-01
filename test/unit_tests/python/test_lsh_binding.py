from __future__ import annotations

import numpy as np
import pytest

import hypervec


def make_index():
    return hypervec.create_index(
        "lsh",
        2,
        hypervec.kMetricInnerProduct,
        {
            "table_count": 4,
            "bits_per_table": 1,
            "probe_count": 2,
            "candidate_limit": 0,
            "random_seed": 42,
        },
    )


def test_lsh_factory_adds_and_searches_with_runtime_parameters():
    index = make_index()
    database = np.array(
        [[1, 0], [0, 2], [2, 1], [3, 4], [1, 3], [6, 2]],
        dtype=np.float32,
    )
    index.add(database)

    distances, labels = index.search_with_params(
        np.array([[1, 1]], dtype=np.float32),
        3,
        {"probe_count": 2, "candidate_limit": 0},
    )
    assert labels == [[5, 3, 4]]
    assert distances == [[8.0, 7.0, 4.0]]


def test_lsh_alias_and_construction_validation_reach_cpp():
    alias = hypervec.create_index(
        "IndexLSH",
        2,
        hypervec.kMetricInnerProduct,
        {"table_count": 2, "bits_per_table": 3, "probe_count": 2},
    )
    assert isinstance(alias, hypervec.Index)

    with pytest.raises(RuntimeError, match="kMetricInnerProduct"):
        hypervec.create_index("lsh", 2, hypervec.kMetricL2, {})
    with pytest.raises(RuntimeError, match="bits_per_table"):
        hypervec.create_index(
            "lsh",
            2,
            hypervec.kMetricInnerProduct,
            {"bits_per_table": 64},
        )
    with pytest.raises(RuntimeError, match="random_seed"):
        hypervec.create_index(
            "lsh",
            2,
            hypervec.kMetricInnerProduct,
            {"random_seed": -1},
        )


def test_lsh_runtime_parameters_reject_invalid_values_and_types():
    index = make_index()
    database = np.array([[1, 0], [0, 1]], dtype=np.float32)
    index.add(database)

    with pytest.raises(RuntimeError, match="probe_count"):
        index.search_with_params(database[:1], 1, {"probe_count": 0})
    with pytest.raises(RuntimeError, match="candidate_limit"):
        index.search_with_params(database[:1], 1, {"candidate_limit": -1})
    with pytest.raises(RuntimeError, match="candidate_limit"):
        index.search_with_params(database[:1], 1, {"candidate_limit": True})


def test_lsh_runtime_parameters_reach_id_map_storage():
    index = hypervec.create_index(
        "lsh",
        2,
        hypervec.kMetricInnerProduct,
        {
            "table_count": 2,
            "bits_per_table": 1,
            "probe_count": 2,
            "candidate_limit": 1,
        },
        True,
    )
    database = np.array([[1, 0], [2, 0], [3, 0]], dtype=np.float32)
    index.add(database)

    distances, labels = index.search_with_params(
        database[:1], 3, {"candidate_limit": 0}
    )
    assert labels == [[2, 1, 0]]
    assert distances == [[3.0, 2.0, 1.0]]


def test_lsh_persistence_roundtrip_remains_incremental(tmp_path):
    index = make_index()
    database = np.array([[1, 0], [2, 0], [3, 0]], dtype=np.float32)
    index.add(database)
    path = tmp_path / "index.lsh"
    hypervec.write_index(index, str(path))

    restored = hypervec.read_index(str(path))
    restored.add(np.array([[4, 0]], dtype=np.float32))
    distances, labels = restored.search_with_params(
        database[:1], 4, {"probe_count": 2, "candidate_limit": 0}
    )
    assert labels == [[3, 2, 1, 0]]
    assert distances == [[4.0, 3.0, 2.0, 1.0]]
