#!/usr/bin/env python3
"""Regression coverage for evaluation artifact identity and output safety."""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

GROUND_TRUTH, BUILD, EVAL = map(Path, sys.argv[1:4])
del sys.argv[1:4]


def write_fvecs(path, rows):
    path.write_bytes(b"".join(struct.pack(f"<i{len(row)}f", len(row), *row)
                              for row in rows))


class ArtifactIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.base = self.directory / "base.fvecs"
        self.queries = self.directory / "queries.fvecs"
        self.index = self.directory / "index.bin"
        self.truth = self.directory / "truth.ivecs"
        self.build_record = self.directory / "build.prov"
        self.gt_record = self.directory / "truth.prov"
        write_fvecs(self.base, [[1, 0], [0, 1], [2, 0], [0, 2]])
        write_fvecs(self.queries, [[1, 0], [0, 1]])
        self.build_args = ["--input", self.base, "--output", self.index,
                           "--index-type", "flat", "--metric", "l2"]
        self.gt_args = ["--base", self.base, "--queries", self.queries,
                       "--output", self.truth, "--k", 1, "--metric", "l2"]
        self.eval_args = ["--index", self.index, "--queries", self.queries,
                         "--ground-truth", self.truth, "--k", 1,
                         "--metric", "l2", "--warmup-runs", 0,
                         "--measured-runs", 1]
        self.call(BUILD, *self.build_args, "--provenance", self.build_record)
        self.call(GROUND_TRUTH, *self.gt_args, "--provenance", self.gt_record)

    def call(self, tool, *args, error=None):
        result = subprocess.run([str(tool), *map(str, args)],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0 if error is None else 1,
                         result.stdout + result.stderr)
        if error:
            self.assertIn(error, result.stderr)
        return result

    def evaluate(self, *args, error=None):
        return self.call(EVAL, *self.eval_args, *args, error=error)

    def both_records(self):
        return ["--build-provenance", self.build_record,
                "--ground-truth-provenance", self.gt_record]

    def test_build_provenance_cannot_overwrite_artifacts(self):
        for destination in [self.base, self.queries, self.index,
                            self.directory / "build.json"]:
            with self.subTest(destination=destination.name):
                destination.touch(exist_ok=True)
                protected = {p: p.read_bytes() for p in
                             [self.base, self.queries, self.index, destination]}
                self.call(BUILD, *self.build_args,
                          "--training-queries", self.queries,
                          "--json-output", self.directory / "build.json",
                          "--provenance", destination,
                          error="artifact output must differ")
                for path, content in protected.items():
                    self.assertEqual(path.read_bytes(), content)

    def test_gt_provenance_cannot_overwrite_inputs_or_truth(self):
        for destination in [self.base, self.queries, self.truth]:
            with self.subTest(destination=destination.name):
                protected = {p: p.read_bytes() for p in
                             [self.base, self.queries, self.truth]}
                self.call(GROUND_TRUTH, *self.gt_args,
                          "--provenance", destination,
                          error="artifact output must differ")
                for path, content in protected.items():
                    self.assertEqual(path.read_bytes(), content)

    def test_alias_paths_are_rejected_before_writing(self):
        for kind in ["lexical", "symlink", "hardlink"]:
            with self.subTest(kind=kind):
                alias = self.directory / (kind + ".fvecs")
                if kind == "lexical":
                    alias = self.directory / "unused" / ".." / self.base.name
                    (self.directory / "unused").mkdir()
                else:
                    try:
                        if kind == "symlink":
                            alias.symlink_to(self.base)
                        else:
                            os.link(self.base, alias)
                    except OSError as error:
                        self.skipTest(f"{kind} not supported: {error}")
                before = self.base.read_bytes()
                self.call(BUILD, *self.build_args, "--provenance", alias,
                          error="artifact output must differ")
                self.assertEqual(self.base.read_bytes(), before)

    def test_report_cannot_overwrite_source_records(self):
        for record in [self.build_record, self.gt_record]:
            before = record.read_bytes()
            self.evaluate(*self.both_records(), "--json-output", record,
                          error="artifact output must differ")
            self.assertEqual(record.read_bytes(), before)

    def test_valid_records_bind_queries_metric_and_width(self):
        raw = self.gt_record.read_bytes()
        self.assertNotIn(b"\r", raw)
        record = dict(line.split("=", 1) for line in raw.decode().splitlines())
        self.assertEqual(record["format"], "hypervec-ground-truth-provenance-v2")
        self.assertEqual(record["semantic_metric"], "l2")
        self.assertEqual(record["ground_truth_width"], "1")
        self.assertEqual(len(record["queries_sha256"]), 64)
        self.assertIn("recall_at_k=1\n",
                      self.evaluate(*self.both_records()).stdout)

    def test_changed_query_order_is_rejected(self):
        write_fvecs(self.queries, [[0, 1], [1, 0]])
        self.evaluate(*self.both_records(), error="queries do not match")

    def test_l2_truth_cannot_score_inner_product_index(self):
        self.call(BUILD, *self.build_args, "--metric", "inner_product",
                  "--provenance", self.build_record)
        self.evaluate(*self.both_records(), "--metric", "inner_product",
                      error="metric does not match")

    def test_legacy_or_incompatible_records_require_regeneration(self):
        original = self.gt_record.read_text()
        for replacement, message in [
            ("", "has no format"),
            ("format=unknown\n", "unsupported ground-truth provenance"),
        ]:
            record = "\n".join(line for line in original.splitlines()
                               if not line.startswith("format="))
            self.gt_record.write_text(replacement + record + "\n")
            self.evaluate(*self.both_records(), error=message)
        self.gt_record.write_text(original.replace("ground_truth_width=1",
                                                  "ground_truth_width=2"))
        self.evaluate(*self.both_records(), error="width does not match")

    def test_crlf_records_match_identical_lf_records(self):
        for record in [self.build_record, self.gt_record]:
            record.write_bytes(record.read_bytes().replace(b"\n", b"\r\n"))
        self.assertIn("recall_at_k=1\n",
                      self.evaluate(*self.both_records()).stdout)

    def test_rerank_checks_each_available_source_record(self):
        wrong = self.directory / "wrong.fvecs"
        write_fvecs(wrong, [[0, 1], [1, 0], [2, 0], [0, 2]])
        for records in [self.both_records(),
                        ["--build-provenance", self.build_record],
                        ["--ground-truth-provenance", self.gt_record]]:
            with self.subTest(records=records[0]):
                self.evaluate(*records, "--rerank-base", wrong,
                              "--rerank-candidates", 4,
                              error="rerank base does not match")
                good = self.evaluate(*records, "--rerank-base", self.base,
                                     "--rerank-candidates", 4)
                self.assertIn("recall_at_k=1\n", good.stdout)

    def test_help_matches_supported_source_record_flags(self):
        help_text = self.call(EVAL, "--help").stdout
        for flag in ["--build-provenance", "--ground-truth-provenance"]:
            self.assertIn(flag, help_text)
        self.assertNotIn("--build-manifest", help_text)
        self.assertNotIn("--ground-truth-manifest", help_text)
        for tool in [BUILD, GROUND_TRUTH]:
            self.assertIn("--provenance", self.call(tool, "--help").stdout)


if __name__ == "__main__":
    unittest.main()
