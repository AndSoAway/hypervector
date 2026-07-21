# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the License) found in the
# LICENSE file in the root directory of this source tree.

"""
Collection data bundle I/O.

A bundle is a ZIP archive with a .hypervec-bundle suffix containing:
  manifest.json  — metadata and checksums for consistency verification
  index.hypervec — binary vector index (same bytes as the on-disk index file)
  scalar.jsonl   — one JSON object per line, each row from the scalar store

The scalar.jsonl rows include the raw float vector alongside the scalar
fields so the bundle is self-contained.  The index.hypervec file is also
included because it carries the optimised index structure (HNSW graph, etc.)
that enables fast search — re-loading from scalar rows alone via flush() is
always possible but slower.

Consistency contract
--------------------
All three files must agree on dim / total / row_id ordering.  The manifest
stores SHA-256 checksums of both index.hypervec and scalar.jsonl so callers
can detect corruption or partial uploads before importing.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .hypervec_meta_store import CollectionMeta

BUNDLE_FORMAT = "hypervector.collection.bundle.v1"
_MANIFEST = "manifest.json"
_INDEX = "index.hypervec"
_SCALAR = "scalar.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_bundle(
    collection_name: str,
    index_path: Path,
    scalar_rows: list[dict[str, Any]],
    meta: "CollectionMeta",
    output_path: Path,
) -> dict[str, Any]:
    """Pack index.hypervec + scalar rows + manifest into a ZIP bundle.

    Returns the manifest dict (which includes checksums and sizes).
    The bundle is written atomically via a temporary file.
    """
    index_bytes = index_path.read_bytes()
    index_checksum = _sha256_bytes(index_bytes)
    index_size_bytes = len(index_bytes)

    scalar_lines = [
        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        for row in scalar_rows
    ]
    scalar_bytes = ("\n".join(scalar_lines) + "\n").encode("utf-8") if scalar_lines else b""
    scalar_checksum = _sha256_bytes(scalar_bytes)

    schema_checksum = _sha256_bytes(
        json.dumps(meta.schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )

    manifest: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "collection_name": collection_name,
        "version": meta.version,
        "dim": meta.dim,
        "total": len(scalar_rows),
        "id_field": meta.id_field,
        "vector_field": meta.vector_field,
        "text_field": meta.text_field,
        "index_checksum": index_checksum,
        "index_size_bytes": index_size_bytes,
        "scalar_checksum": scalar_checksum,
        "schema_checksum": schema_checksum,
        "exported_at": time.time(),
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")

    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(_MANIFEST, manifest_bytes)
            zf.writestr(_INDEX, index_bytes)
            zf.writestr(_SCALAR, scalar_bytes)
        tmp.replace(output_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    return manifest


def read_bundle(
    bundle_path: Path,
) -> tuple[dict[str, Any], bytes, list[dict[str, Any]]]:
    """Unpack a bundle ZIP.

    Returns (manifest, index_bytes, scalar_rows).
    Raises ValueError for format errors or checksum mismatches.
    """
    if not bundle_path.exists():
        raise FileNotFoundError(f"bundle file not found: {bundle_path}")

    try:
        with zipfile.ZipFile(bundle_path, "r") as zf:
            names = zf.namelist()
            for required in (_MANIFEST, _INDEX, _SCALAR):
                if required not in names:
                    raise ValueError(
                        f"bundle is missing required file '{required}': {bundle_path}"
                    )
            manifest = json.loads(zf.read(_MANIFEST).decode("utf-8"))
            index_bytes = zf.read(_INDEX)
            scalar_bytes = zf.read(_SCALAR)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"bundle is not a valid ZIP file: {bundle_path}") from exc

    if manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError(
            f"unsupported bundle format '{manifest.get('format')}'; "
            f"expected '{BUNDLE_FORMAT}'"
        )

    # Verify checksums
    actual_index_checksum = _sha256_bytes(index_bytes)
    if manifest.get("index_checksum") and manifest["index_checksum"] != actual_index_checksum:
        raise ValueError(
            f"index checksum mismatch: manifest={manifest['index_checksum']} "
            f"actual={actual_index_checksum}"
        )

    actual_scalar_checksum = _sha256_bytes(scalar_bytes)
    if manifest.get("scalar_checksum") and manifest["scalar_checksum"] != actual_scalar_checksum:
        raise ValueError(
            f"scalar checksum mismatch: manifest={manifest['scalar_checksum']} "
            f"actual={actual_scalar_checksum}"
        )

    scalar_rows: list[dict[str, Any]] = []
    for line in scalar_bytes.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            scalar_rows.append(json.loads(line))

    return manifest, index_bytes, scalar_rows


def bundle_checksum(bundle_path: Path) -> str:
    """Return the SHA-256 checksum of a bundle file on disk."""
    return _sha256_file(bundle_path)
