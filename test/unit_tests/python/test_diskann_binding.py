from __future__ import annotations

import numpy as np
import pytest

import hypervec


def make_index(*, use_id_map: bool = False, node_data_path: str = ""):
    return hypervec.create_index(
        "diskann",
        1,
        hypervec.kMetricL2,
        {
            "max_degree": 4,
            "build_search_width": 8,
            "candidate_pool_size": 8,
            "alpha": 1.2,
            "build_passes": 2,
            "random_seed": 42,
            "search_width": 8,
            "check_relative_distance": True,
            "page_size": 64,
            "cache_capacity_pages": 1,
            "node_data_path": node_data_path,
        },
        use_id_map,
    )


def database() -> np.ndarray:
    return np.array([[0], [2], [5], [9], [14], [20]], dtype=np.float32)


def test_diskann_factory_builds_and_accepts_runtime_parameters():
    index = make_index()
    vectors = database()
    index.build(vectors)

    distances, labels = index.search_with_params(
        np.array([[4]], dtype=np.float32),
        3,
        {"search_width": 8, "check_relative_distance": False},
    )

    assert labels == [[2, 1, 0]]
    assert distances == [[1.0, 4.0, 16.0]]
    with pytest.raises(RuntimeError, match="incremental insertion"):
        index.add(vectors[:1])


def test_diskann_alias_and_file_backing_reach_cpp(tmp_path):
    node_path = tmp_path / "diskann.nodes"
    index = hypervec.create_index(
        "IndexDiskANNFlat",
        1,
        hypervec.kMetricL2,
        {
            "max_degree": 4,
            "build_search_width": 8,
            "candidate_pool_size": 8,
            "page_size": 64,
            "cache_capacity_pages": 1,
            "node_data_path": str(node_path),
        },
    )
    index.build(database())
    assert node_path.is_file()
    assert node_path.stat().st_size > 0

    with pytest.raises(RuntimeError, match="kMetricL2"):
        hypervec.create_index(
            "diskann", 1, hypervec.kMetricInnerProduct, {}
        )
    with pytest.raises(RuntimeError, match="cache_capacity_pages"):
        hypervec.create_index(
            "diskann", 1, hypervec.kMetricL2, {"cache_capacity_pages": 0}
        )
    with pytest.raises(RuntimeError, match="unknown parameter"):
        hypervec.create_index(
            "diskann", 1, hypervec.kMetricL2, {"typo": 1}
        )


def test_diskann_runtime_parameters_reject_invalid_values_and_types():
    index = make_index()
    vectors = database()
    index.build(vectors)

    with pytest.raises(RuntimeError, match="search_width"):
        index.search_with_params(vectors[:1], 1, {"search_width": 0})
    with pytest.raises(RuntimeError, match="search_width"):
        index.search_with_params(vectors[:1], 1, {"search_width": True})
    with pytest.raises(RuntimeError, match="check_relative_distance"):
        index.search_with_params(
            vectors[:1], 1, {"check_relative_distance": "yes"}
        )


def test_diskann_id_map_persistence_keeps_runtime_parameter_path(tmp_path):
    vectors = database()
    index = make_index(use_id_map=True)
    index.build(vectors)
    expected_distances, expected_labels = index.search_with_params(
        vectors[:2], 3, {"search_width": 8}
    )

    path = tmp_path / "diskann.index"
    hypervec.write_index(index, str(path))
    restored = hypervec.read_index(str(path))
    actual_distances, actual_labels = restored.search_with_params(
        vectors[:2], 3, {"search_width": 8}
    )

    np.testing.assert_array_equal(actual_labels, expected_labels)
    np.testing.assert_allclose(
        actual_distances, expected_distances, rtol=0, atol=0
    )
