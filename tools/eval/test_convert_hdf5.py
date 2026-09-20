#!/usr/bin/env python3
"""Focused tests for the optional HDF5 converter."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import h5py
    import numpy
except ImportError:
    h5py = None
    numpy = None

CONVERTER_PATH = Path(sys.argv[1]).resolve()


@unittest.skipIf(h5py is None or numpy is None, "h5py and numpy are optional")
class Hdf5ConverterTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.converter = CONVERTER_PATH

    def tearDown(self):
        self.temporary.cleanup()

    def _run(self, source, prefix):
        outputs = {
            "base": self.directory / f"{prefix}-base.fvecs",
            "queries": self.directory / f"{prefix}-query.fvecs",
            "ground_truth": self.directory / f"{prefix}-gt.ivecs",
            "metadata": self.directory / f"{prefix}.json",
        }
        command = [
            sys.executable,
            str(self.converter),
            "--name",
            prefix,
            "--input",
            str(source),
            "--base-output",
            str(outputs["base"]),
            "--query-output",
            str(outputs["queries"]),
            "--ground-truth-output",
            str(outputs["ground_truth"]),
            "--metadata-output",
            str(outputs["metadata"]),
            "--ground-truth-k",
            "1",
            "--chunk-rows",
            "1",
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        return result, outputs

    @staticmethod
    def _read_vectors(path, rows, dimension, dtype):
        records = numpy.fromfile(path, dtype=dtype).reshape(rows, dimension + 1)
        headers = records[:, 0]
        if dtype == "<f4":
            headers = headers.copy().view("<i4")
        numpy.testing.assert_array_equal(headers, dimension)
        return records[:, 1:]

    def test_converts_and_normalizes_angular_dataset(self):
        source = self.directory / "source.hdf5"
        with h5py.File(source, "w") as output:
            output.attrs["distance"] = "angular"
            output.create_dataset(
                "train", data=numpy.asarray([[3, 4], [0, 2], [-2, 0]])
            )
            output.create_dataset(
                "test", data=numpy.asarray([[4, 3], [1, 0]])
            )
            output.create_dataset(
                "neighbors", data=numpy.asarray([[0, 1], [2, 0]])
            )

        result, outputs = self._run(source, "angular")
        self.assertEqual(result.returncode, 0, result.stderr)
        base = self._read_vectors(outputs["base"], 3, 2, "<f4")
        queries = self._read_vectors(outputs["queries"], 2, 2, "<f4")
        numpy.testing.assert_allclose(
            numpy.linalg.norm(base, axis=1), 1.0, rtol=1e-6
        )
        numpy.testing.assert_allclose(
            numpy.linalg.norm(queries, axis=1), 1.0, rtol=1e-6
        )
        neighbors = self._read_vectors(
            outputs["ground_truth"], 2, 1, "<i4"
        )
        numpy.testing.assert_array_equal(neighbors[:, 0], [0, 2])
        metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
        self.assertEqual(metadata["semantic_metric"], "cosine")
        self.assertTrue(metadata["l2_normalized"])
        self.assertEqual(metadata["ground_truth_k"], 1)

    def test_failure_does_not_publish_outputs(self):
        source = self.directory / "zero.hdf5"
        with h5py.File(source, "w") as output:
            output.attrs["distance"] = "angular"
            output.create_dataset("train", data=numpy.asarray([[0.0, 0.0]]))
            output.create_dataset("test", data=numpy.asarray([[1.0, 0.0]]))
            output.create_dataset("neighbors", data=numpy.asarray([[0]]))

        result, outputs = self._run(source, "zero")
        self.assertNotEqual(result.returncode, 0)
        for path in outputs.values():
            self.assertFalse(path.exists())

    def test_converts_euclidean_dataset_without_neighbors(self):
        source = self.directory / "euclidean.hdf5"
        expected_base = numpy.asarray([[3.0, 4.0], [0.0, -2.0]])
        expected_queries = numpy.asarray([[4.0, 3.0]])
        with h5py.File(source, "w") as output:
            output.attrs["distance"] = "euclidean"
            output.create_dataset("train", data=expected_base)
            output.create_dataset("test", data=expected_queries)

        base = self.directory / "l2-base.fvecs"
        queries = self.directory / "l2-query.fvecs"
        metadata = self.directory / "l2.json"
        result = subprocess.run(
            [
                sys.executable,
                str(self.converter),
                "--name",
                "euclidean",
                "--input",
                str(source),
                "--base-output",
                str(base),
                "--query-output",
                str(queries),
                "--metadata-output",
                str(metadata),
            ],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        numpy.testing.assert_array_equal(
            self._read_vectors(base, 2, 2, "<f4"), expected_base
        )
        numpy.testing.assert_array_equal(
            self._read_vectors(queries, 1, 2, "<f4"), expected_queries
        )
        imported = json.loads(metadata.read_text(encoding="utf-8"))
        self.assertEqual(imported["semantic_metric"], "l2")
        self.assertFalse(imported["l2_normalized"])
        self.assertIsNone(imported["ground_truth"])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
