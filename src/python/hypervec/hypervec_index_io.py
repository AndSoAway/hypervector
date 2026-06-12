# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the License) found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def index_file_info(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return {
        "index_checksum": file_sha256(p) if p.exists() else None,
        "index_size_bytes": p.stat().st_size if p.exists() else None,
    }
