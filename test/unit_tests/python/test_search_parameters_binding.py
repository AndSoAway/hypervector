from __future__ import annotations

import numpy as np
import pytest

import hypervec


def test_hnsw_search_parameters_use_shared_cpp_validation():
    index = hypervec.create_index(
        "hnsw_flat",
        2,
        hypervec.kMetricL2,
        {"m_hnsw": 4},
    )
    database = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
    index.add(database)

    distances, labels = index.search_with_params(
        database[:1],
        2,
        {
            "ef": 8,
            "check_relative_distance": False,
            "bounded_queue": False,
        },
    )
    assert labels[0][0] == 0
    assert distances[0][0] == 0.0

    with pytest.raises(RuntimeError, match="cannot both be set"):
        index.search_with_params(
            database[:1], 1, {"ef": 8, "ef_search": 8}
        )
    with pytest.raises(RuntimeError, match="ef_search"):
        index.search_with_params(database[:1], 1, {"ef_search": True})


def test_search_parameters_reject_unknown_names_and_value_kinds():
    index = hypervec.create_index(
        "flat", 2, hypervec.kMetricL2, {}
    )
    query = np.array([[0, 0]], dtype=np.float32)

    distances, labels = index.search_with_params(query, 1, {})
    assert labels == [[-1]]
    assert distances[0][0] > 1e30

    with pytest.raises(RuntimeError, match="unknown search parameter"):
        index.search_with_params(query, 1, {"nporbe": 2})
    with pytest.raises(RuntimeError, match="bool or int"):
        index.search_with_params(query, 1, {"nprobe": 1.5})


def test_empty_runtime_config_is_supported():
    index = hypervec.create_index(
        "nsw_flat",
        1,
        hypervec.kMetricL2,
        {"max_degree": 4, "ef_construction": 8, "ef_search": 3},
    )
    database = np.array([[0], [2], [5], [9]], dtype=np.float32)
    index.add(database)

    distances, labels = index.search_with_params(database[:1], 2, {})
    assert labels[0][0] == 0
    assert distances[0][0] == 0.0
