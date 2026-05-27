#!/usr/bin/env python3

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np

from hypervec import HyperVectorIndexBackend


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logger = logging.getLogger("hypervector-backend-demo")

    data_dir = Path("hv_backend_data")
    if data_dir.exists():
        shutil.rmtree(data_dir)

    config = {
        "uri": str(data_dir),
        "collection_name": "demo_collection",
        "metric_type": "L2",
        "index_params": {
            "index_type": "HNSWFlat",
            "M": 16,
        },
        "index_chunk_size": 2,
    }

    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [1.0, 1.0],
            [9.8, 10.1],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )
    ids = np.array(["chunk-0", "chunk-1", "chunk-2", "chunk-3", "chunk-4"])
    contents = [
        "origin point document",
        "near origin document",
        "middle document",
        "near ten document",
        "ten point document",
    ]
    metadatas = [
        {"source": "demo", "doc_id": "zero", "title": "Origin"},
        {"source": "demo", "doc_id": "zero", "title": "Near Origin"},
        {"source": "demo", "doc_id": "middle", "title": "Middle"},
        {"source": "manual", "doc_id": "ten", "title": "Near Ten"},
        {"source": "manual", "doc_id": "ten", "title": "Ten"},
    ]

    backend = HyperVectorIndexBackend(contents=[], config=config, logger=logger)
    backend.build_index(
        embeddings=embeddings,
        ids=ids,
        contents=contents,
        metadatas=metadatas,
        overwrite=True,
    )

    query = np.array([[0.05, 0.05], [10.2, 9.9]], dtype=np.float32)

    print("\nsearch() text-only results:")
    for row in backend.search(query, top_k=2):
        print(row)

    print("\nsearch_with_meta() structured results:")
    for row in backend.search_with_meta(query, top_k=2):
        for item in row:
            print(
                {
                    "content": item["content"],
                    "score": item["score"],
                    "chunk_id": item["chunk_id"],
                    "source": item["source"],
                    "metadata": item["metadata"],
                }
            )

    print("\nfiltered results: source == 'manual'")
    filtered = backend.search_with_meta(
        np.array([[10.2, 9.9]], dtype=np.float32),
        top_k=2,
        filter="source == 'manual'",
    )
    for item in filtered[0]:
        print(item["content"], item["metadata"])

    reloaded = HyperVectorIndexBackend(contents=[], config=config, logger=logger)
    reloaded.load_index()
    print("\nreloaded search:")
    print(reloaded.search(np.array([[0.0, 0.0]], dtype=np.float32), top_k=2))


if __name__ == "__main__":
    main()
