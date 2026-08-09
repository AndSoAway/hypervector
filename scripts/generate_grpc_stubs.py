#!/usr/bin/env python3
"""Regenerate checked-in gRPC stubs for the server and client packages."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTO_DIR = ROOT / "src" / "python"
PROTO_FILE = PROTO_DIR / "hypervec.proto"
CLIENT_PACKAGE = ROOT / "pyhypervec" / "pyhypervec"


def generate(output_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            str(PROTO_FILE),
        ],
        check=True,
    )


def replace_import(path: Path, replacement: str) -> None:
    generated = "import hypervec_pb2 as hypervec__pb2"
    text = path.read_text(encoding="utf-8")
    if generated not in text:
        raise RuntimeError(f"unexpected generated import in {path}")
    path.write_text(text.replace(generated, replacement, 1), encoding="utf-8")


def main() -> None:
    generate(PROTO_DIR)
    generate(CLIENT_PACKAGE)
    replace_import(
        CLIENT_PACKAGE / "hypervec_pb2_grpc.py",
        "from . import hypervec_pb2 as hypervec__pb2",
    )
    replace_import(
        PROTO_DIR / "hypervec_pb2_grpc.py",
        "try:\n"
        "    from . import hypervec_pb2 as hypervec__pb2\n"
        "except ImportError:  # pragma: no cover - supports direct script execution\n"
        "    import hypervec_pb2 as hypervec__pb2",
    )


if __name__ == "__main__":
    main()
