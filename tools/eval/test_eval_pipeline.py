#!/usr/bin/env python3
"""End-to-end smoke test for the file-based evaluation pipeline."""

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def write_fvecs(path, rows):
    dimension = len(rows[0])
    with path.open("wb") as output:
        for row in rows:
            if len(row) != dimension:
                raise ValueError("inconsistent test vector dimension")
            output.write(struct.pack(f"<i{dimension}f", dimension, *row))


def run(*arguments):
    return subprocess.run(
        [str(argument) for argument in arguments],
        check=True,
        text=True,
        capture_output=True,
    )


def main():
    ground_truth_tool = Path(sys.argv[1]).resolve()
    build_tool = Path(sys.argv[2]).resolve()
    eval_tool = Path(sys.argv[3]).resolve()
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        base = directory / "base.fvecs"
        queries = directory / "queries.fvecs"
        ground_truth = directory / "ground-truth.ivecs"
        index = directory / "flat.index"
        report_path = directory / "report.json"
        write_fvecs(base, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        write_fvecs(queries, [[0.9, 0.1]])

        run(
            ground_truth_tool,
            "--base",
            base,
            "--queries",
            queries,
            "--output",
            ground_truth,
            "--k",
            2,
            "--metric",
            "l2",
        )
        run(
            build_tool,
            "--input",
            base,
            "--output",
            index,
            "--index-type",
            "flat",
            "--metric",
            "l2",
        )
        run(
            eval_tool,
            "--index",
            index,
            "--queries",
            queries,
            "--ground-truth",
            ground_truth,
            "--metric",
            "l2",
            "--k",
            2,
            "--warmup-runs",
            0,
            "--measured-runs",
            1,
            "--json-output",
            report_path,
        )

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["format"] == "hypervec-eval-report-v1"
        assert report["index"]["family"] == "generic"
        assert report["index"]["metric"] == "l2"
        assert report["index"]["vector_count"] == 3
        assert report["workload"]["query_count"] == 1
        assert report["workload"]["semantic_metric"] == "l2"
        assert report["workload"]["k"] == 2
        assert report["execution"]["warmup_runs"] == 0
        assert report["execution"]["measured_runs"] == 1
        assert report["execution"]["search_parameters"] == []
        assert report["metrics"]["recall_at_k"] == 1.0
        assert Path(report["index"]["path"]).is_absolute()
        assert Path(report["workload"]["queries"]).is_absolute()

        cosine_base = directory / "cosine-base.fvecs"
        cosine_queries = directory / "cosine-queries.fvecs"
        cosine_ground_truth = directory / "cosine-ground-truth.ivecs"
        cosine_index = directory / "cosine-flat.index"
        cosine_report_path = directory / "cosine-report.json"
        write_fvecs(cosine_base, [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        write_fvecs(cosine_queries, [[0.8, 0.6]])

        run(
            ground_truth_tool,
            "--base",
            cosine_base,
            "--queries",
            cosine_queries,
            "--output",
            cosine_ground_truth,
            "--k",
            2,
            "--metric",
            "cosine",
        )
        run(
            build_tool,
            "--input",
            cosine_base,
            "--output",
            cosine_index,
            "--index-type",
            "flat",
            "--metric",
            "cosine",
        )
        run(
            eval_tool,
            "--index",
            cosine_index,
            "--queries",
            cosine_queries,
            "--ground-truth",
            cosine_ground_truth,
            "--metric",
            "cosine",
            "--k",
            2,
            "--warmup-runs",
            0,
            "--measured-runs",
            1,
            "--json-output",
            cosine_report_path,
        )
        cosine_report = json.loads(
            cosine_report_path.read_text(encoding="utf-8")
        )
        assert cosine_report["index"]["metric"] == "inner_product"
        assert cosine_report["workload"]["semantic_metric"] == "cosine"
        assert cosine_report["metrics"]["recall_at_k"] == 1.0


if __name__ == "__main__":
    main()
