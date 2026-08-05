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

# v1 manifest fields that must be present for an import to be accepted.
_REQUIRED_MANIFEST_FIELDS = ("index_checksum", "scalar_checksum", "schema_checksum")

# Label strategy describes how index labels map to scalar row_ids.  v1 uses
# "implicit_sequential": index label i corresponds to the row whose row_id == i,
# which requires row_ids to be exactly 0..total-1 with no gaps.  Future versions
# may add explicit label maps; recording the strategy in the manifest lets
# readers pick the right validation without guessing.
LABEL_STRATEGY_IMPLICIT_SEQUENTIAL = "implicit_sequential"
_DEFAULT_LABEL_STRATEGY = LABEL_STRATEGY_IMPLICIT_SEQUENTIAL

# Zip-bomb guards: a bundle contains exactly three known members.  Reject
# anything with extra entries, an implausibly large uncompressed payload, or an
# extreme compression ratio.
_ALLOWED_MEMBERS = frozenset((_MANIFEST, _INDEX, _SCALAR))
_MAX_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024  # 8 GiB total
_MAX_COMPRESSION_RATIO = 200


def _is_int(value: Any) -> bool:
    """True only for genuine ints — bool is explicitly rejected.

    row_id is used as an index label, so a stray ``True`` (which ``isinstance``
    would otherwise accept as ``1``) must not silently become row_id 1.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def validate_index_label_mapping(
    scalar_rows: list[dict[str, Any]],
    *,
    total: int,
    dim: int | None = None,
    label_strategy: str = _DEFAULT_LABEL_STRATEGY,
) -> None:
    """Validate that scalar rows can be addressed by index labels.

    Proving ``index.n_total == total == len(scalar_rows)`` only shows the counts
    agree — it does not show that index label *i* can be resolved to a scalar
    row.  For the v1 ``implicit_sequential`` strategy that resolution is
    ``row_id == label``, so every row must carry a unique integer row_id that
    exactly covers ``0..total-1``.  Each row's vector must also be 1-D and, when
    ``dim`` is given, match it.

    Raises ValueError on any violation.
    """
    if label_strategy != LABEL_STRATEGY_IMPLICIT_SEQUENTIAL:
        raise ValueError(
            f"unsupported label_strategy '{label_strategy}'; "
            f"only '{LABEL_STRATEGY_IMPLICIT_SEQUENTIAL}' is supported."
        )

    if len(scalar_rows) != int(total):
        raise ValueError(
            f"scalar row count {len(scalar_rows)} does not match total {total}."
        )

    seen: set[int] = set()
    for i, row in enumerate(scalar_rows):
        if "row_id" not in row:
            raise ValueError(f"scalar row at position {i} is missing row_id.")
        row_id = row["row_id"]
        if not _is_int(row_id):
            raise ValueError(
                f"scalar row at position {i} has non-integer row_id "
                f"{row_id!r} (type {type(row_id).__name__})."
            )
        if row_id in seen:
            raise ValueError(f"duplicate row_id {row_id} in scalar rows.")
        seen.add(row_id)

        vector = row.get("vector")
        if vector is not None:
            if not isinstance(vector, (list, tuple)):
                raise ValueError(
                    f"scalar row {row_id} vector must be a 1-D list, "
                    f"got {type(vector).__name__}."
                )
            if any(isinstance(v, (list, tuple)) for v in vector):
                raise ValueError(f"scalar row {row_id} vector must be 1-D.")
            if dim is not None and len(vector) != int(dim):
                raise ValueError(
                    f"scalar row {row_id} vector dim {len(vector)} does not "
                    f"match manifest dim {dim}."
                )

    expected = set(range(int(total)))
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            f"row_ids must cover 0..{int(total) - 1} exactly; "
            f"missing={missing[:10]} extra={extra[:10]}."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def schema_checksum(schema: dict[str, Any] | None) -> str:
    """Deterministic SHA-256 of a collection schema.

    Must match the serialization used in create_bundle so import-time schema
    compatibility checks compare like with like.
    """
    return _sha256_bytes(
        json.dumps(schema or {}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


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

    schema_cksum = schema_checksum(meta.schema)

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
        "schema_checksum": schema_cksum,
        "label_strategy": _DEFAULT_LABEL_STRATEGY,
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
            infos = zf.infolist()
            names = [i.filename for i in infos]
            # Zip-bomb / tampering guard: only the three known members are
            # allowed, and the total uncompressed size and per-entry compression
            # ratio must stay within sane bounds.  We never extractall().
            extra = set(names) - _ALLOWED_MEMBERS
            if extra:
                raise ValueError(
                    f"bundle contains unexpected entries {sorted(extra)}: {bundle_path}"
                )
            total_uncompressed = 0
            for info in infos:
                total_uncompressed += info.file_size
                if info.compress_size > 0:
                    ratio = info.file_size / info.compress_size
                    if ratio > _MAX_COMPRESSION_RATIO:
                        raise ValueError(
                            f"bundle entry '{info.filename}' has suspicious "
                            f"compression ratio {ratio:.0f} (limit "
                            f"{_MAX_COMPRESSION_RATIO}): {bundle_path}"
                        )
            if total_uncompressed > _MAX_UNCOMPRESSED_BYTES:
                raise ValueError(
                    f"bundle uncompressed size {total_uncompressed} exceeds limit "
                    f"{_MAX_UNCOMPRESSED_BYTES}: {bundle_path}"
                )
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

    # v1 bundles must carry all integrity checksums — reject partial manifests
    # rather than silently skipping verification.
    missing = [f for f in _REQUIRED_MANIFEST_FIELDS if not manifest.get(f)]
    if missing:
        raise ValueError(
            f"bundle manifest is missing required v1 field(s) {missing}: {bundle_path}"
        )

    # Verify checksums
    actual_index_checksum = _sha256_bytes(index_bytes)
    if manifest["index_checksum"] != actual_index_checksum:
        raise ValueError(
            f"index checksum mismatch: manifest={manifest['index_checksum']} "
            f"actual={actual_index_checksum}"
        )

    actual_scalar_checksum = _sha256_bytes(scalar_bytes)
    if manifest["scalar_checksum"] != actual_scalar_checksum:
        raise ValueError(
            f"scalar checksum mismatch: manifest={manifest['scalar_checksum']} "
            f"actual={actual_scalar_checksum}"
        )

    scalar_rows: list[dict[str, Any]] = []
    for line in scalar_bytes.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            scalar_rows.append(json.loads(line))

    # manifest.total is mandatory: the label-mapping validation below is keyed
    # off it, and a hand-crafted bundle that omits it must fail with a clear
    # error rather than a downstream KeyError.
    if manifest.get("total") is None:
        raise ValueError(
            f"bundle manifest is missing required field 'total': {bundle_path}"
        )

    # Index-label ↔ row_id mapping: counts alone don't prove label i resolves to
    # a scalar row.  Validate row_id type/uniqueness/coverage and vector shape
    # according to the manifest's declared label strategy (default v1).
    try:
        validate_index_label_mapping(
            scalar_rows,
            total=int(manifest["total"]),
            dim=manifest.get("dim"),
            label_strategy=manifest.get("label_strategy", _DEFAULT_LABEL_STRATEGY),
        )
    except ValueError as exc:
        raise ValueError(f"bundle {bundle_path} is invalid: {exc}") from exc

    return manifest, index_bytes, scalar_rows


def bundle_checksum(bundle_path: Path) -> str:
    """Return the SHA-256 checksum of a bundle file on disk."""
    return _sha256_file(bundle_path)
