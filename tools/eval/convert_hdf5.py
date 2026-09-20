#!/usr/bin/env python3
"""Convert an ANN-style HDF5 dataset to HyperVec evaluation files."""

import argparse
import json
import os
from pathlib import Path
import tempfile


def _dependencies():
    try:
        import h5py  # pylint: disable=import-outside-toplevel
        import numpy  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise RuntimeError(
            "HDF5 conversion requires optional packages: "
            "python3 -m pip install h5py numpy"
        ) from error
    return h5py, numpy


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert train/test/neighbors arrays from an ANN HDF5 file to "
            "fvecs/ivecs without adding an HDF5 dependency to HyperVec."
        )
    )
    parser.add_argument("--name", required=True, help="stable dataset name")
    parser.add_argument("--input", required=True, help="source HDF5 file")
    parser.add_argument("--base-output", required=True, help="base fvecs path")
    parser.add_argument(
        "--query-output", required=True, help="query fvecs path"
    )
    parser.add_argument(
        "--ground-truth-output",
        help="optional ivecs path populated from the neighbors dataset",
    )
    parser.add_argument(
        "--metadata-output", required=True, help="conversion metadata JSON"
    )
    parser.add_argument("--base-key", default="train")
    parser.add_argument("--query-key", default="test")
    parser.add_argument("--neighbors-key", default="neighbors")
    parser.add_argument(
        "--semantic-metric",
        choices=("auto", "l2", "inner_product", "cosine"),
        default="auto",
        help="default auto reads the HDF5 distance attribute",
    )
    parser.add_argument(
        "--ground-truth-k",
        type=int,
        help="optional prefix width exported from neighbors",
    )
    parser.add_argument(
        "--chunk-rows", type=int, default=8192, help="streaming chunk size"
    )
    return parser


def _normalized_path(value):
    return Path(value).expanduser().resolve(strict=False)


def _validate_paths(args):
    input_path = _normalized_path(args.input)
    outputs = [
        _normalized_path(args.base_output),
        _normalized_path(args.query_output),
        _normalized_path(args.metadata_output),
    ]
    if args.ground_truth_output:
        outputs.append(_normalized_path(args.ground_truth_output))
    if input_path in outputs or len(outputs) != len(set(outputs)):
        raise ValueError("input and output paths must be distinct")
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")
    if args.ground_truth_k is not None and args.ground_truth_k <= 0:
        raise ValueError("--ground-truth-k must be positive")
    if args.ground_truth_k is not None and not args.ground_truth_output:
        raise ValueError(
            "--ground-truth-k requires --ground-truth-output"
        )


def _distance_attribute(hdf5_file):
    value = hdf5_file.attrs.get("distance")
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _semantic_metric(requested, source_distance):
    if requested != "auto":
        return requested
    if source_distance is None:
        raise ValueError(
            "HDF5 file has no distance attribute; set --semantic-metric"
        )
    aliases = {
        "angular": "cosine",
        "cosine": "cosine",
        "euclidean": "l2",
        "l2": "l2",
        "dot": "inner_product",
        "ip": "inner_product",
        "inner_product": "inner_product",
    }
    metric = aliases.get(source_distance.lower())
    if metric is None:
        raise ValueError(
            f"unsupported HDF5 distance attribute: {source_distance!r}"
        )
    return metric


def _require_dataset(hdf5_file, key, kind, numpy):
    if key not in hdf5_file:
        raise ValueError(f"HDF5 file has no {kind} dataset {key!r}")
    dataset = hdf5_file[key]
    if dataset.ndim != 2 or dataset.shape[0] <= 0 or dataset.shape[1] <= 0:
        raise ValueError(f"{kind} dataset must be a non-empty 2-D matrix")
    if kind == "neighbors":
        if not numpy.issubdtype(dataset.dtype, numpy.integer):
            raise ValueError("neighbors dataset must contain integers")
    elif not (
        numpy.issubdtype(dataset.dtype, numpy.integer)
        or numpy.issubdtype(dataset.dtype, numpy.floating)
    ):
        raise ValueError(f"{kind} dataset must contain real numeric values")
    return dataset


def _temporary_output(destination):
    destination = _normalized_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name), destination


def _write_fvecs(dataset, temporary_path, normalize, chunk_rows, numpy):
    dimension = int(dataset.shape[1])
    if dimension > numpy.iinfo(numpy.int32).max:
        raise ValueError("vector dimension exceeds the fvecs format range")
    with temporary_path.open("wb") as output:
        for start in range(0, int(dataset.shape[0]), chunk_rows):
            end = min(start + chunk_rows, int(dataset.shape[0]))
            chunk = numpy.asarray(dataset[start:end], dtype=numpy.float32)
            if not numpy.isfinite(chunk).all():
                raise ValueError("vector dataset contains a non-finite value")
            if normalize:
                wide = chunk.astype(numpy.float64)
                norms = numpy.sqrt(numpy.einsum("ij,ij->i", wide, wide))
                if not numpy.isfinite(norms).all() or numpy.any(norms == 0.0):
                    raise ValueError(
                        "cosine conversion requires finite nonzero vectors"
                    )
                chunk = numpy.asarray(wide / norms[:, None], dtype="<f4")
            else:
                chunk = numpy.asarray(chunk, dtype="<f4")
            records = numpy.empty(
                (chunk.shape[0], dimension + 1), dtype="<f4"
            )
            records.view("<i4")[:, 0] = dimension
            records[:, 1:] = chunk
            output.write(records.tobytes(order="C"))


def _write_ivecs(
    dataset, temporary_path, base_count, width, chunk_rows, numpy
):
    if width > numpy.iinfo(numpy.int32).max:
        raise ValueError("ground-truth width exceeds the ivecs format range")
    dimension_word = numpy.asarray([width], dtype="<i4")[0]
    with temporary_path.open("wb") as output:
        for start in range(0, int(dataset.shape[0]), chunk_rows):
            end = min(start + chunk_rows, int(dataset.shape[0]))
            chunk = numpy.asarray(dataset[start:end, :width], dtype=numpy.int64)
            if numpy.any(chunk < 0) or numpy.any(chunk >= base_count):
                raise ValueError("neighbors dataset contains an invalid base ID")
            if numpy.any(chunk > numpy.iinfo(numpy.int32).max):
                raise ValueError("neighbor ID exceeds the ivecs format range")
            records = numpy.empty((chunk.shape[0], width + 1), dtype="<i4")
            records[:, 0] = dimension_word
            records[:, 1:] = chunk.astype("<i4")
            output.write(records.tobytes(order="C"))


def _metadata(args, source_distance, metric, base, queries, ground_truth_k):
    return {
        "format": "hypervec-eval-import-v1",
        "name": args.name,
        "source": args.input,
        "source_format": "hdf5",
        "source_distance": source_distance,
        "base_key": args.base_key,
        "query_key": args.query_key,
        "neighbors_key": (
            args.neighbors_key if args.ground_truth_output else None
        ),
        "dtype": "float32",
        "base": args.base_output,
        "queries": args.query_output,
        "ground_truth": args.ground_truth_output,
        "dimension": int(base.shape[1]),
        "base_count": int(base.shape[0]),
        "query_count": int(queries.shape[0]),
        "ground_truth_k": ground_truth_k,
        "semantic_metric": metric,
        "l2_normalized": metric == "cosine",
        "conversion": "hdf5-stream-v1",
    }


def convert(args):
    _validate_paths(args)
    h5py, numpy = _dependencies()
    staged = []
    try:
        with h5py.File(args.input, "r") as hdf5_file:
            base = _require_dataset(
                hdf5_file, args.base_key, "base", numpy
            )
            queries = _require_dataset(
                hdf5_file, args.query_key, "query", numpy
            )
            if base.shape[1] != queries.shape[1]:
                raise ValueError("base and query dimensions must match")
            source_distance = _distance_attribute(hdf5_file)
            metric = _semantic_metric(args.semantic_metric, source_distance)

            base_stage = _temporary_output(args.base_output)
            query_stage = _temporary_output(args.query_output)
            staged.extend((base_stage, query_stage))
            normalize = metric == "cosine"
            _write_fvecs(
                base, base_stage[0], normalize, args.chunk_rows, numpy
            )
            _write_fvecs(
                queries, query_stage[0], normalize, args.chunk_rows, numpy
            )

            ground_truth_k = None
            if args.ground_truth_output:
                neighbors = _require_dataset(
                    hdf5_file, args.neighbors_key, "neighbors", numpy
                )
                if neighbors.shape[0] != queries.shape[0]:
                    raise ValueError(
                        "neighbors and query vector counts must match"
                    )
                ground_truth_k = (
                    args.ground_truth_k
                    if args.ground_truth_k is not None
                    else int(neighbors.shape[1])
                )
                if ground_truth_k > neighbors.shape[1]:
                    raise ValueError(
                        "--ground-truth-k exceeds the neighbors width"
                    )
                ground_truth_stage = _temporary_output(
                    args.ground_truth_output
                )
                staged.append(ground_truth_stage)
                _write_ivecs(
                    neighbors,
                    ground_truth_stage[0],
                    int(base.shape[0]),
                    ground_truth_k,
                    args.chunk_rows,
                    numpy,
                )

            metadata_stage = _temporary_output(args.metadata_output)
            staged.append(metadata_stage)
            metadata = _metadata(
                args,
                source_distance,
                metric,
                base,
                queries,
                ground_truth_k,
            )
            metadata_stage[0].write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        for temporary, destination in staged:
            os.replace(temporary, destination)
        print(f"dataset={args.name}")
        print(f"dimension={metadata['dimension']}")
        print(f"base_count={metadata['base_count']}")
        print(f"query_count={metadata['query_count']}")
        print(f"semantic_metric={metric}")
        print(f"ground_truth_k={ground_truth_k or 0}")
    except Exception:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise


def main():
    args = _parser().parse_args()
    try:
        convert(args)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"hypervec_convert_hdf5: {error}") from error


if __name__ == "__main__":
    main()
