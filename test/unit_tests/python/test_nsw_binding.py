from __future__ import annotations

import numpy as np
import pytest

import hypervec


def make_index():
    return hypervec.create_index(
        "nsw_flat",
        1,
        hypervec.kMetricL2,
        {
            "max_degree": 8,
            "ef_construction": 8,
            "ef_search": 8,
            "check_relative_distance": True,
            "fill_to_max_degree": True,
        },
    )


def test_nsw_factory_builds_and_searches_with_runtime_parameters():
    index = make_index()
    assert isinstance(index, hypervec.Index)
    assert index.d == 1

    database = np.array([[0], [2], [5], [9], [14], [20]], dtype=np.float32)
    query = np.array([[4]], dtype=np.float32)
    index.add(database)

    distances, labels = index.search_with_params(
        query,
        3,
        {"ef_search": 8, "check_relative_distance": False},
    )
    assert labels == [[2, 1, 0]]
    assert distances == [[1.0, 4.0, 16.0]]


def test_nsw_alias_and_parameter_validation_reach_cpp():
    alias = hypervec.create_index(
        "NSW",
        2,
        hypervec.kMetricInnerProduct,
        {"max_degree": 4, "ef_construction": 8},
    )
    assert isinstance(alias, hypervec.Index)
    assert alias.metric_type == hypervec.kMetricInnerProduct

    with pytest.raises(RuntimeError, match="ef_construction"):
        hypervec.create_index(
            "nsw_flat",
            2,
            hypervec.kMetricL2,
            {"max_degree": 8, "ef_construction": 4},
        )


def test_nsw_runtime_parameters_reject_invalid_types_and_values():
    index = make_index()
    database = np.array([[0], [1]], dtype=np.float32)
    index.add(database)

    with pytest.raises(RuntimeError, match="ef_search"):
        index.search_with_params(database[:1], 1, {"ef_search": 0})
    with pytest.raises(RuntimeError, match="check_relative_distance"):
        index.search_with_params(
            database[:1], 1, {"check_relative_distance": "yes"}
        )


def test_nsw_persistence_roundtrip_remains_searchable(tmp_path):
    index = make_index()
    database = np.array([[0], [2], [5], [9], [14], [20]], dtype=np.float32)
    index.add(database)
    path = tmp_path / "index.nsw"
    hypervec.write_index(index, str(path))

    restored = hypervec.read_index(str(path))
    distances, labels = restored.search_with_params(
        np.array([[4]], dtype=np.float32), 3, {"ef_search": 8}
    )
    assert labels == [[2, 1, 0]]
    assert distances == [[1.0, 4.0, 16.0]]

    restored.add(np.array([[25]], dtype=np.float32))
    assert restored.n_total == 7
