from __future__ import annotations

import numpy as np
import pytest

import hypervec


def training_data() -> np.ndarray:
    random = np.random.default_rng(2026)
    return random.normal(size=(96, 8)).astype(np.float32)


def make_index(use_id_map: bool = False):
    return hypervec.create_index(
        "ivf_rabitq",
        8,
        hypervec.kMetricL2,
        {"nlist": 4, "random_seed": 42, "rotation_rounds": 3},
        use_id_map,
    )


def test_ivf_rabitq_factory_builds_and_accepts_runtime_nprobe():
    database = training_data()
    index = make_index()
    index.build(database)

    distances, labels = index.search_with_params(
        database[:4], 3, {"nprobe": 4}
    )
    assert len(distances) == 4
    assert len(labels) == 4
    assert all(len(row) == 3 for row in labels)
    assert [row[0] for row in labels] == [0, 1, 2, 3]
    assert np.isfinite(np.asarray(distances)).all()
    assert (np.asarray(distances) >= 0).all()


def test_ivf_rabitq_alias_and_id_map_use_the_common_binding_path():
    alias = hypervec.create_index(
        "IndexIVFRaBitQ",
        8,
        hypervec.kMetricL2,
        {"nlist": 4, "random_seed": 7, "rotation_rounds": 2},
    )
    assert isinstance(alias, hypervec.Index)

    database = training_data()
    mapped = make_index(use_id_map=True)
    mapped.build(database)
    distances, labels = mapped.search_with_params(
        database[:1], 2, {"nprobe": 4}
    )
    assert labels[0][0] == 0
    assert len(distances[0]) == 2


def test_ivf_rabitq_rejects_invalid_construction_and_search_parameters():
    with pytest.raises(RuntimeError, match="kMetricL2"):
        hypervec.create_index(
            "ivf_rabitq", 8, hypervec.kMetricInnerProduct, {"nlist": 4}
        )
    with pytest.raises(RuntimeError, match="random_seed"):
        hypervec.create_index(
            "ivf_rabitq", 8, hypervec.kMetricL2, {"random_seed": -1}
        )
    with pytest.raises(RuntimeError, match="rotation_rounds"):
        hypervec.create_index(
            "ivf_rabitq", 8, hypervec.kMetricL2, {"rotation_rounds": 17}
        )
    with pytest.raises(RuntimeError, match="unknown parameter"):
        hypervec.create_index(
            "ivf_rabitq", 8, hypervec.kMetricL2, {"typo": 1}
        )

    index = make_index()
    database = training_data()
    index.build(database)
    with pytest.raises(RuntimeError, match="nprobe"):
        index.search_with_params(database[:1], 1, {"nprobe": 0})
