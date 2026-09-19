from __future__ import annotations

import numpy as np
import pytest

import hypervec


def make_index():
    return hypervec.create_index(
        "nsg_flat",
        1,
        hypervec.kMetricL2,
        {
            "knn_degree": 8,
            "nn_descent_iterations": 8,
            "nn_descent_convergence_threshold": 0.0,
            "random_seed": 42,
            "max_degree": 8,
            "build_search_width": 8,
            "candidate_pool_size": 16,
            "ef_search": 16,
            "check_relative_distance": True,
        },
    )


def test_nsg_factory_builds_and_searches_with_runtime_parameters():
    index = make_index()
    assert isinstance(index, hypervec.Index)
    assert index.d == 1

    database = np.array([[0], [2], [5], [9], [14], [20]], dtype=np.float32)
    query = np.array([[4]], dtype=np.float32)
    index.build(database)

    distances, labels = index.search_with_params(
        query,
        3,
        {"ef_search": 8, "check_relative_distance": False},
    )
    assert labels == [[2, 1, 0]]
    assert distances == [[1.0, 4.0, 16.0]]
    with pytest.raises(RuntimeError, match="incremental insertion"):
        index.add(database[:1])


def test_nsg_alias_and_build_parameter_validation_reach_cpp():
    alias = hypervec.create_index(
        "NSG",
        2,
        hypervec.kMetricInnerProduct,
        {"knn_degree": 4, "max_degree": 4, "candidate_pool_size": 8},
    )
    assert isinstance(alias, hypervec.Index)
    assert alias.metric_type == hypervec.kMetricInnerProduct

    with pytest.raises(RuntimeError, match="convergence_threshold"):
        hypervec.create_index(
            "nsg_flat",
            2,
            hypervec.kMetricL2,
            {"nn_descent_convergence_threshold": 2.0},
        )
    with pytest.raises(RuntimeError, match="random_seed"):
        hypervec.create_index(
            "nsg_flat",
            2,
            hypervec.kMetricL2,
            {"random_seed": -1},
        )


def test_nsg_runtime_parameters_reject_invalid_types_and_values():
    index = make_index()
    database = np.array([[0], [1]], dtype=np.float32)
    index.build(database)

    with pytest.raises(RuntimeError, match="ef_search"):
        index.search_with_params(database[:1], 1, {"ef_search": 0})
    with pytest.raises(RuntimeError, match="check_relative_distance"):
        index.search_with_params(
            database[:1], 1, {"check_relative_distance": "yes"}
        )


def test_nsg_persistence_roundtrip_remains_searchable(tmp_path):
    index = make_index()
    database = np.array([[0], [2], [5], [9], [14], [20]], dtype=np.float32)
    index.build(database)
    path = tmp_path / "index.nsg"
    hypervec.write_index(index, str(path))

    restored = hypervec.read_index(str(path))
    distances, labels = restored.search_with_params(
        np.array([[4]], dtype=np.float32), 3, {"ef_search": 8}
    )
    assert labels == [[2, 1, 0]]
    assert distances == [[1.0, 4.0, 16.0]]
    with pytest.raises(RuntimeError, match="incremental insertion"):
        restored.add(np.array([[25]], dtype=np.float32))


def test_common_build_binding_validates_matrix_shape_and_dtype():
    index = make_index()
    with pytest.raises(RuntimeError, match="float32"):
        index.build(np.array([[0], [1]], dtype=np.float64))
    with pytest.raises(RuntimeError, match="2-D"):
        index.build(np.array([0, 1], dtype=np.float32))
