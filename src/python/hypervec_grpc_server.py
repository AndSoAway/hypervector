from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from concurrent import futures
from pathlib import Path
from typing import Any

try:
    from .hypervec_server_engine import HypervecServerEngine
    from . import hypervec_service_pb2 as pb2
    from . import hypervec_service_pb2_grpc as pb2_grpc
except ImportError:
    from hypervec_server_engine import HypervecServerEngine
    import hypervec_service_pb2 as pb2
    import hypervec_service_pb2_grpc as pb2_grpc


def _require_grpc():
    try:
        import grpc
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("HyperVec gRPC server requires grpcio.") from exc
    return grpc


def _json_loads(raw: str | bytes | None, default: Any) -> Any:
    if raw in (None, "", b""):
        return default
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _to_json_response(payload: Any) -> pb2.JsonResponse:
    return pb2.JsonResponse(json=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


class HypervecGrpcServicer(pb2_grpc.HypervecServiceServicer):
    def __init__(self, engine: HypervecServerEngine) -> None:
        self.engine = engine

    def Health(self, request, context):
        del request, context
        return pb2.HealthResponse(status="ok")

    def ListCollections(self, request, context):
        del request, context
        return pb2.ListCollectionsResponse(collections=self.engine.list_collections())

    def HasCollection(self, request, context):
        return pb2.HasCollectionResponse(
            collection_name=request.collection_name,
            exists=self.engine.has_collection(request.collection_name),
        )

    def DescribeCollection(self, request, context):
        return _to_json_response(self.engine.describe_collection(request.collection_name))

    def CreateCollection(self, request, context):
        return _to_json_response(
            self.engine.create_collection(
                request.collection_name,
                schema=_json_loads(request.schema_json, {}),
                index_params=_json_loads(request.index_params_json, {"indexes": []}),
            )
        )

    def DropCollection(self, request, context):
        return _to_json_response(self.engine.drop_collection(request.collection_name))

    def Insert(self, request, context):
        return _to_json_response(self.engine.insert(request.collection_name, _json_loads(request.data_json, [])))

    def Flush(self, request, context):
        return _to_json_response(self.engine.flush(request.collection_name))

    def LoadCollection(self, request, context):
        return _to_json_response(self.engine.load_collection(request.collection_name))

    def CloseCollection(self, request, context):
        return _to_json_response(self.engine.close_collection(request.collection_name))

    def Search(self, request, context):
        if request.data_bytes:
            import numpy as np
            query = np.frombuffer(request.data_bytes, dtype=np.float32)
            if request.dim and request.num_rows:
                query = query.reshape(request.num_rows, request.dim)
            data = query
        else:
            data = _json_loads(request.data_json, [])
        output_fields = list(request.output_fields)
        id_only = output_fields == ["id"] and not request.filter
        results = self.engine.search(
            request.collection_name,
            data=data,
            limit=request.limit,
            search_params=_json_loads(request.search_params_json, {}),
            output_fields=output_fields,
            filter=request.filter,
            consistency_level=request.consistency_level or None,
        )
        return pb2.SearchResponse(
            results=[
                pb2.SearchResult(
                    hits=[
                        pb2.SearchHit(
                            id=str(hit.get("id", "")),
                            distance=float(hit.get("distance", 0.0)),
                            entity_json="" if id_only else json.dumps(hit.get("entity", {}), ensure_ascii=False, separators=(",", ":")),
                        )
                        for hit in row
                    ]
                )
                for row in results
            ]
        )

    def GetVersion(self, request, context):
        return _to_json_response(self.engine.get_version(request.collection_name))

    def SyncCheck(self, request, context):
        return _to_json_response(
            self.engine.sync_check(
                request.collection_name,
                client_version=request.client_version,
                client_checksum=request.client_checksum or None,
            )
        )

    def DownloadIndex(self, request, context):
        path = self.engine.index_path_for_download(request.collection_name)
        version = self.engine.get_version(request.collection_name)
        return pb2.DownloadIndexResponse(
            data=Path(path).read_bytes(),
            version=int(version.get("version") or 0),
            index_checksum=str(version.get("index_checksum") or ""),
            index_size_bytes=int(version.get("index_size_bytes") or 0),
        )

    def UploadIndex(self, request, context):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(request.data)
            tmp_path = f.name
        try:
            return _to_json_response(
                self.engine.upload_index(
                    request.collection_name,
                    tmp_path,
                    version=request.version or None,
                    checksum=request.checksum or None,
                )
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)



def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HyperVec gRPC server.")
    parser.add_argument("--data-root", required=True, help="Collection data root.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", default=50051, type=int, help="Bind port.")
    parser.add_argument("--workers", default=64, type=int, help="Thread pool workers.")
    args = parser.parse_args()

    grpc = _require_grpc()
    logging.basicConfig(level=logging.INFO)
    engine = HypervecServerEngine(args.data_root)
    if os.environ.get("HYPERVEC_PRELOAD_COLLECTIONS", "1") not in {"0", "false", "False"}:
        for collection_name in engine.list_collections():
            try:
                engine.load_collection(collection_name)
                logging.info("Preloaded collection '%s'.", collection_name)
            except Exception as exc:
                logging.warning("Failed to preload collection '%s': %s", collection_name, exc)
    servicer = HypervecGrpcServicer(engine)
    max_message_size = int(os.environ.get("HYPERVEC_GRPC_MAX_MESSAGE_SIZE", str(1024 * 1024 * 1024)))
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.workers),
        options=[
            ("grpc.max_send_message_length", max_message_size),
            ("grpc.max_receive_message_length", max_message_size),
            ("grpc.max_concurrent_streams", args.workers * 2),
            ("grpc.so_reuseport", 1),
        ],
    )
    pb2_grpc.add_HypervecServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    logging.info("HyperVec gRPC server listening on %s:%s", args.host, args.port)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
