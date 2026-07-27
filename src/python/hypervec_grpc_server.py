# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2
# (the "License") found in the LICENSE file in the root directory of this
# source tree.

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

try:
    import grpc
except ImportError as exc:  # pragma: no cover - dependency error path
    raise RuntimeError(
        "HyperVector gRPC server requires grpcio and protobuf. "
        "Install hypervec[grpc-server]."
    ) from exc

try:
    from .hypervec_server_engine import ConflictError, HypervecServerEngine
    from . import hypervec_pb2 as pb2
    from . import hypervec_pb2_grpc as pb2_grpc
except ImportError:  # pragma: no cover - supports direct file execution
    sys.path.insert(0, str(Path(__file__).parent))
    from hypervec_server_engine import ConflictError, HypervecServerEngine
    import hypervec_pb2 as pb2
    import hypervec_pb2_grpc as pb2_grpc


DEFAULT_GRPC_MAX_MESSAGE_MB = 256


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _status_for(exc: Exception) -> grpc.StatusCode:
    if isinstance(exc, FileNotFoundError):
        return grpc.StatusCode.NOT_FOUND
    if isinstance(exc, FileExistsError):
        return grpc.StatusCode.ALREADY_EXISTS
    if isinstance(exc, ConflictError):
        return grpc.StatusCode.FAILED_PRECONDITION
    if isinstance(exc, (ValueError, TypeError, json.JSONDecodeError)):
        return grpc.StatusCode.INVALID_ARGUMENT
    return grpc.StatusCode.INTERNAL


def _abort(context: grpc.ServicerContext, exc: Exception) -> None:
    context.abort(_status_for(exc), str(exc))


class HyperVecServicer(pb2_grpc.HyperVecServicer):
    """Protocol adapter around the shared HypervecServerEngine."""

    def __init__(self, engine: HypervecServerEngine) -> None:
        self._engine = engine

    def Health(self, request, context):
        return pb2.HealthResponse(status="ok")

    def ListCollections(self, request, context):
        return pb2.ListCollectionsResponse(
            collections=self._engine.list_collections()
        )

    def DescribeCollections(self, request, context):
        try:
            result = {"collections": self._engine.describe_collections()}
        except Exception as exc:
            _abort(context, exc)
        return pb2.DescribeCollectionsResponse(json_payload=_json(result))

    def Examples(self, request, context):
        try:
            result = {"examples": self._engine.supported_index_examples()}
        except Exception as exc:
            _abort(context, exc)
        return pb2.ExamplesResponse(json_payload=_json(result))

    def HasCollection(self, request, context):
        try:
            exists = self._engine.has_collection(request.collection_name)
        except Exception as exc:
            _abort(context, exc)
        return pb2.HasCollectionResponse(
            collection_name=request.collection_name,
            exists=exists,
        )

    def DescribeCollection(self, request, context):
        try:
            result = self._engine.describe_collection(request.collection_name)
        except Exception as exc:
            _abort(context, exc)
        return pb2.DescribeCollectionResponse(json_payload=_json(result))

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
        return pb2.CreateCollectionResponse(json_payload=_json(result))

    def DropCollection(self, request, context):
        try:
            result = self._engine.drop_collection(request.collection_name)
        except Exception as exc:
            _abort(context, exc)
        return pb2.DropCollectionResponse(json_payload=_json(result))

    def Insert(self, request, context):
        try:
            data = json.loads(request.data_json)
            result = self._engine.insert(request.collection_name, data)
        except Exception as exc:
            _abort(context, exc)
        return pb2.InsertResponse(json_payload=_json(result))

    def Flush(self, request, context):
        try:
            result = self._engine.flush(request.collection_name)
        except Exception as exc:
            _abort(context, exc)
        return pb2.FlushResponse(json_payload=_json(result))

    def LoadCollection(self, request, context):
        try:
            result = self._engine.load_collection(request.collection_name)
        except Exception as exc:
            _abort(context, exc)
        return pb2.LoadCollectionResponse(json_payload=_json(result))

    def CloseCollection(self, request, context):
        try:
            result = self._engine.close_collection(request.collection_name)
        except Exception as exc:
            _abort(context, exc)
        return pb2.CloseCollectionResponse(json_payload=_json(result))

    def Search(self, request, context):
        try:
            data = json.loads(request.query_json)
            search_params = json.loads(request.search_params_json or "{}")
            result = self._engine.search(
                request.collection_name,
                data=data,
                limit=request.limit,
                search_params=search_params,
                output_fields=list(request.output_fields),
                filter=request.filter or "",
                consistency_level=request.consistency_level or None,
            )
        except Exception as exc:
            _abort(context, exc)
        return pb2.SearchResponse(results_json=_json(result))

    def GetVersion(self, request, context):
        try:
            result = self._engine.get_version(request.collection_name)
        except Exception as exc:
            _abort(context, exc)
        return pb2.GetVersionResponse(json_payload=_json(result))

    def SyncCheck(self, request, context):
        try:
            result = self._engine.sync_check(
                request.collection_name,
                client_version=request.client_version,
                client_checksum=request.client_checksum or None,
            )
        except Exception as exc:
            _abort(context, exc)
        return pb2.SyncCheckResponse(json_payload=_json(result))

    def DownloadIndex(self, request, context):
        try:
            path = self._engine.index_path_for_download(request.collection_name)
            version = self._engine.get_version(request.collection_name)
            data = path.read_bytes()
        except Exception as exc:
            _abort(context, exc)
        return pb2.DownloadIndexResponse(
            data=data,
            version=int(version.get("version") or 0),
            checksum=str(version.get("index_checksum") or ""),
            size_bytes=int(version.get("index_size_bytes") or len(data)),
        )

    def UploadIndex(self, request, context):
        tmp_path = None
        try:
            if not request.data:
                raise ValueError("uploaded index body is empty.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".hypervec") as handle:
                handle.write(request.data)
                tmp_path = handle.name
            result = self._engine.upload_index(
                request.collection_name,
                tmp_path,
                version=request.version if request.HasField("version") else None,
                checksum=request.checksum or None,
            )
        except Exception as exc:
            _abort(context, exc)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return pb2.UploadIndexResponse(json_payload=_json(result))

    def DownloadCollectionBundle(self, request, context):
        bundle_path = None
        try:
            exported = self._engine.export_collection_bundle(request.collection_name)
            bundle_path = exported["path"]
            data = Path(bundle_path).read_bytes()
        except Exception as exc:
            _abort(context, exc)
        finally:
            if bundle_path:
                try:
                    os.unlink(bundle_path)
                except OSError:
                    pass
        return pb2.DownloadCollectionBundleResponse(
            data=data,
            version=int(exported.get("version") or 0),
            bundle_format=str(exported.get("bundle_format") or ""),
            checksum=str(exported.get("bundle_checksum") or ""),
            size_bytes=int(exported.get("bytes") or len(data)),
        )

    def UploadCollectionBundle(self, request, context):
        tmp_path = None
        try:
            if not request.data:
                raise ValueError("uploaded bundle body is empty.")
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".hypervec-bundle",
            ) as handle:
                handle.write(request.data)
                tmp_path = handle.name
            result = self._engine.import_collection_bundle(
                request.collection_name,
                tmp_path,
                checksum=request.checksum or None,
                mode=request.mode or "replace",
            )
        except Exception as exc:
            _abort(context, exc)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return pb2.UploadCollectionBundleResponse(json_payload=_json(result))

    def PurgeCollectionData(self, request, context):
        try:
            require_exported = (
                request.require_exported
                if request.HasField("require_exported")
                else True
            )
            result = self._engine.purge_collection_data(
                request.collection_name,
                require_exported=require_exported,
            )
        except Exception as exc:
            _abort(context, exc)
        return pb2.PurgeCollectionDataResponse(json_payload=_json(result))


def create_server(
    *,
    data_root: str,
    engine: HypervecServerEngine | None = None,
    max_workers: int = 10,
    max_message_mb: int = DEFAULT_GRPC_MAX_MESSAGE_MB,
) -> grpc.Server:
    engine = engine or HypervecServerEngine(data_root)
    max_bytes = int(max_message_mb) * 1024 * 1024
    if max_bytes <= 0:
        raise ValueError("max_message_mb must be positive.")
    server = grpc.server(
        ThreadPoolExecutor(max_workers=max_workers),
        options=(
            ("grpc.max_send_message_length", max_bytes),
            ("grpc.max_receive_message_length", max_bytes),
        ),
    )
    pb2_grpc.add_HyperVecServicer_to_server(HyperVecServicer(engine), server)
    return server


def bind_server(server: grpc.Server, address: str) -> int:
    bound_port = server.add_insecure_port(address)
    if bound_port == 0:
        raise RuntimeError(f"failed to bind gRPC server to {address}")
    return bound_port


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HyperVector gRPC server.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--host", default="[::]")
    parser.add_argument("--port", default=50051, type=int)
    parser.add_argument("--workers", default=10, type=int)
    parser.add_argument(
        "--max-message-mb",
        default=DEFAULT_GRPC_MAX_MESSAGE_MB,
        type=int,
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    server = create_server(
        data_root=args.data_root,
        max_workers=args.workers,
        max_message_mb=args.max_message_mb,
    )
    address = f"{args.host}:{args.port}"
    bind_server(server, address)
    server.start()
    logging.getLogger("hypervec.grpc").info("gRPC server listening on %s", address)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
