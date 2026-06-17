# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the License) found in the
# LICENSE file in the root directory of this source tree.

"""
HyperVec gRPC server — wraps HypervecServerEngine with the same pattern
as hypervec_http_server.py.  The engine is the single source of truth;
this module only handles protocol adaptation and error mapping.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import grpc

try:
    from .hypervec_server_engine import HypervecServerEngine
    from . import hypervec_pb2 as pb2
    from . import hypervec_pb2_grpc as pb2_grpc
except ImportError:  # pragma: no cover - supports direct file loading
    sys.path.insert(0, str(Path(__file__).parent))
    from hypervec_server_engine import HypervecServerEngine
    import hypervec_pb2 as pb2
    import hypervec_pb2_grpc as pb2_grpc


def _status_for(exc: Exception) -> grpc.StatusCode:
    if isinstance(exc, FileNotFoundError):
        return grpc.StatusCode.NOT_FOUND
    if isinstance(exc, FileExistsError):
        return grpc.StatusCode.ALREADY_EXISTS
    if isinstance(exc, ValueError):
        return grpc.StatusCode.INVALID_ARGUMENT
    return grpc.StatusCode.INTERNAL


def _abort(context: grpc.ServicerContext, exc: Exception) -> None:
    context.abort(_status_for(exc), str(exc))


class HyperVecServicer(pb2_grpc.HyperVecServicer):
    def __init__(self, engine: HypervecServerEngine) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def Health(self, request, context):
        return pb2.HealthResponse(status="ok")

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ListCollections(self, request, context):
        return pb2.ListCollectionsResponse(collections=self._engine.list_collections())

    def HasCollection(self, request, context):
        try:
            exists = self._engine.has_collection(request.collection_name)
        except Exception as exc:
            _abort(context, exc)
            return pb2.HasCollectionResponse()
        return pb2.HasCollectionResponse(
            collection_name=request.collection_name,
            exists=exists,
        )

    def DescribeCollection(self, request, context):
        try:
            result = self._engine.describe_collection(request.collection_name)
        except Exception as exc:
            _abort(context, exc)
            return pb2.DescribeCollectionResponse()
        return pb2.DescribeCollectionResponse(json_payload=json.dumps(result))

    def CreateCollection(self, request, context):
        try:
            schema = json.loads(request.schema_json)
            index_params = json.loads(request.index_params_json or "{}")
            result = self._engine.create_collection(
                request.collection_name,
                schema=schema,
                index_params=index_params or {"indexes": []},
            )
        except Exception as exc:
            _abort(context, exc)
            return pb2.CreateCollectionResponse()
        return pb2.CreateCollectionResponse(json_payload=json.dumps(result))

    def DropCollection(self, request, context):
        try:
            result = self._engine.drop_collection(request.collection_name)
        except Exception as exc:
            _abort(context, exc)
            return pb2.DropCollectionResponse()
        return pb2.DropCollectionResponse(json_payload=json.dumps(result))

    # ------------------------------------------------------------------
    # Data operations
    # ------------------------------------------------------------------

    def Insert(self, request, context):
        try:
            data = json.loads(request.data_json)
            result = self._engine.insert(request.collection_name, data)
        except Exception as exc:
            _abort(context, exc)
            return pb2.InsertResponse()
        return pb2.InsertResponse(json_payload=json.dumps(result))

    def Flush(self, request, context):
        try:
            result = self._engine.flush(request.collection_name)
        except Exception as exc:
            _abort(context, exc)
            return pb2.FlushResponse()
        return pb2.FlushResponse(json_payload=json.dumps(result))

    def LoadCollection(self, request, context):
        try:
            result = self._engine.load_collection(request.collection_name)
        except Exception as exc:
            _abort(context, exc)
            return pb2.LoadCollectionResponse()
        return pb2.LoadCollectionResponse(json_payload=json.dumps(result))

    def CloseCollection(self, request, context):
        try:
            result = self._engine.close_collection(request.collection_name)
        except Exception as exc:
            _abort(context, exc)
            return pb2.CloseCollectionResponse()
        return pb2.CloseCollectionResponse(json_payload=json.dumps(result))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def Search(self, request, context):
        try:
            data = json.loads(request.query_json)
            search_params = json.loads(request.search_params_json or "{}")
            results = self._engine.search(
                request.collection_name,
                data=data,
                limit=request.limit,
                search_params=search_params,
                output_fields=list(request.output_fields) or None,
                filter=request.filter or "",
                consistency_level=request.consistency_level or None,
            )
        except Exception as exc:
            _abort(context, exc)
            return pb2.SearchResponse()
        return pb2.SearchResponse(results_json=json.dumps(results))

    # ------------------------------------------------------------------
    # Version / sync
    # ------------------------------------------------------------------

    def GetVersion(self, request, context):
        try:
            result = self._engine.get_version(request.collection_name)
        except Exception as exc:
            _abort(context, exc)
            return pb2.GetVersionResponse()
        return pb2.GetVersionResponse(json_payload=json.dumps(result))

    def SyncCheck(self, request, context):
        try:
            result = self._engine.sync_check(
                request.collection_name,
                client_version=request.client_version,
                client_checksum=request.client_checksum or None,
            )
        except Exception as exc:
            _abort(context, exc)
            return pb2.SyncCheckResponse()
        return pb2.SyncCheckResponse(json_payload=json.dumps(result))

    # ------------------------------------------------------------------
    # Index transfer
    # ------------------------------------------------------------------

    def DownloadIndex(self, request, context):
        try:
            path = self._engine.index_path_for_download(request.collection_name)
            version_info = self._engine.get_version(request.collection_name)
            data = path.read_bytes()
        except Exception as exc:
            _abort(context, exc)
            return pb2.DownloadIndexResponse()
        return pb2.DownloadIndexResponse(
            data=data,
            version=int(version_info.get("version") or 0),
            checksum=str(version_info.get("index_checksum") or ""),
            size_bytes=int(version_info.get("index_size_bytes") or 0),
        )

    def UploadIndex(self, request, context):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".hypervec") as f:
                f.write(request.data)
                tmp_path = f.name
            try:
                result = self._engine.upload_index(
                    request.collection_name,
                    tmp_path,
                    version=int(request.version) if request.version else None,
                    checksum=request.checksum or None,
                )
            finally:
                import os
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        except Exception as exc:
            _abort(context, exc)
            return pb2.UploadIndexResponse()
        return pb2.UploadIndexResponse(json_payload=json.dumps(result))


def create_server(
    *,
    data_root: str,
    engine: HypervecServerEngine | None = None,
    max_workers: int = 10,
) -> grpc.Server:
    """Return a configured (but not yet started) gRPC server."""
    engine = engine or HypervecServerEngine(data_root)
    server = grpc.server(ThreadPoolExecutor(max_workers=max_workers))
    pb2_grpc.add_HyperVecServicer_to_server(HyperVecServicer(engine), server)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HyperVec gRPC server.")
    parser.add_argument("--data-root", required=True, help="Collection data root.")
    parser.add_argument("--host", default="[::]", help="Bind host.")
    parser.add_argument("--port", default=50051, type=int, help="Bind port.")
    parser.add_argument("--workers", default=10, type=int, help="Thread pool workers.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    server = create_server(data_root=args.data_root, max_workers=args.workers)
    bind = f"{args.host}:{args.port}"
    server.add_insecure_port(bind)
    server.start()
    logging.getLogger("hypervec.grpc").info("gRPC server listening on %s", bind)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
