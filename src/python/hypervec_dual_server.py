# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2
# (the "License") found in the LICENSE file in the root directory of this
# source tree.

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

try:
    from .hypervec_grpc_server import (
        DEFAULT_GRPC_MAX_MESSAGE_MB,
        bind_server,
        create_server,
    )
    from .hypervec_http_server import create_app
    from .hypervec_server_engine import HypervecServerEngine
except ImportError:  # pragma: no cover - supports direct file execution
    sys.path.insert(0, str(Path(__file__).parent))
    from hypervec_grpc_server import (
        DEFAULT_GRPC_MAX_MESSAGE_MB,
        bind_server,
        create_server,
    )
    from hypervec_http_server import create_app
    from hypervec_server_engine import HypervecServerEngine


def create_dual_services(
    *,
    data_root: str,
    grpc_workers: int = 10,
    grpc_max_message_mb: int = DEFAULT_GRPC_MAX_MESSAGE_MB,
    engine: HypervecServerEngine | None = None,
) -> tuple[HypervecServerEngine, Any, Any]:
    """Create HTTP and gRPC adapters around exactly one engine instance."""

    shared_engine = engine or HypervecServerEngine(data_root)
    http_app = create_app(data_root=data_root, engine=shared_engine)
    grpc_server = create_server(
        data_root=data_root,
        engine=shared_engine,
        max_workers=grpc_workers,
        max_message_mb=grpc_max_message_mb,
    )
    return shared_engine, http_app, grpc_server


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HTTP and gRPC with one shared HyperVector engine."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--http-host", default="0.0.0.0")
    parser.add_argument("--http-port", default=8080, type=int)
    parser.add_argument("--grpc-host", default="0.0.0.0")
    parser.add_argument("--grpc-port", default=50051, type=int)
    parser.add_argument("--grpc-workers", default=10, type=int)
    parser.add_argument(
        "--grpc-max-message-mb",
        default=DEFAULT_GRPC_MAX_MESSAGE_MB,
        type=int,
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Dual server requires uvicorn and fastapi.") from exc

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    _, http_app, grpc_server = create_dual_services(
        data_root=args.data_root,
        grpc_workers=args.grpc_workers,
        grpc_max_message_mb=args.grpc_max_message_mb,
    )
    grpc_address = f"{args.grpc_host}:{args.grpc_port}"
    bind_server(grpc_server, grpc_address)
    grpc_server.start()
    logging.getLogger("hypervec.dual").info(
        "gRPC listening on %s; HTTP listening on %s:%d",
        grpc_address,
        args.http_host,
        args.http_port,
    )

    try:
        uvicorn.run(
            http_app,
            host=args.http_host,
            port=args.http_port,
            log_level=args.log_level,
        )
    finally:
        grpc_server.stop(grace=5).wait()


if __name__ == "__main__":
    main()
