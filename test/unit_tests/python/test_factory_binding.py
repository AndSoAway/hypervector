from __future__ import annotations

import json

import numpy as np
import pytest

import hypervec


def _hnsw_config() -> str:
    return json.dumps(
        {
            "type": "hnsw_flat",
            "dimension": 4,
            "metric": "l2",
            "build": {"graph_degree": 4},
            "defaults": {"ef_search": 8},
        }
    )


def test_registry_factory_and_configured_search() -> None:
    config = _hnsw_config()
    normalized = json.loads(hypervec.normalize_index_config(config))

    assert normalized["type"] == "hnsw_flat"
    assert normalized["storage"]["type"] == "flat"
    assert normalized["build"]["graph_degree"] == 4
    assert normalized["build"]["ef_construction"] == 40
    assert normalized["defaults"]["ef_search"] == 8

    index = hypervec.create_index(config)
    vectors = np.eye(4, dtype=np.float32)
    index.add(vectors)
    distances, labels = index.search_with_config(
        vectors[:1], 1, config, json.dumps({"ef_search": 4})
    )

    assert labels == [[0]]
    assert distances[0][0] == pytest.approx(0.0)


def test_registry_factory_supports_external_id_wrapper() -> None:
    config_object = json.loads(_hnsw_config())
    config_object["id_mode"] = "external"
    config = json.dumps(config_object)
    index = hypervec.create_index(config)
    vectors = np.eye(4, dtype=np.float32)
    index.add(vectors)

    distances, labels = index.search_with_config(vectors[:1], 1, config, "{}")

    assert labels == [[0]]
    assert distances[0][0] == pytest.approx(0.0)


def test_configured_search_rejects_wrong_family_and_unknown_parameter() -> None:
    config = _hnsw_config()
    index = hypervec.create_index(config)
    vectors = np.eye(4, dtype=np.float32)
    index.add(vectors)

    flat_config = json.dumps({"type": "flat", "dimension": 4, "metric": "l2"})
    with pytest.raises(RuntimeError, match="does not match index config"):
        index.search_with_config(vectors[:1], 1, flat_config, "{}")

    wrong_dimension = json.dumps(
        {"type": "hnsw_flat", "dimension": 8, "metric": "l2"}
    )
    with pytest.raises(RuntimeError, match="does not match index config"):
        index.search_with_config(vectors[:1], 1, wrong_dimension, "{}")

    with pytest.raises(
        RuntimeError, match="unknown field 'unknown' in search config"
    ):
        index.search_with_config(vectors[:1], 1, config, json.dumps({"unknown": 1}))
