from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .exceptions import HypervecClientError

# The generated stubs live alongside the server in src/python/
_SRC_PYTHON = Path(__file__).parents[3] / "src" / "python"
if _SRC_PYTHON.exists() and str(_SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SRC_PYTHON))


def _import_stubs():
    try:
        import grpc
        import hypervec_pb2 as pb2
        import hypervec_pb2_grpc as pb2_grpc
    except ImportError as exc:
        raise HypervecClientError(
            "gRPC transport requires grpcio and the generated stubs. "
            "Install grpcio: pip install grpcio"
        ) from exc
    return grpc, pb2, pb2_grpc


class GrpcTransport:
    """Thin wrapper that turns HypervecClient calls into gRPC requests."""

    def __init__(self, address: str, *, timeout: float = 30.0) -> None:
        grpc, pb2, pb2_grpc = _import_stubs()
        self._grpc = grpc
        self._pb2 = pb2
        self._channel = grpc.insecure_channel(address)
        self._stub = pb2_grpc.HyperVecStub(self._channel)
        self._timeout = timeout

    def close(self) -> None:
        self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _call(self, rpc_name: str, request):
        try:
            return getattr(self._stub, rpc_name)(request, timeout=self._timeout)
        except self._grpc.RpcError as exc:
            code = exc.code()
            detail = exc.details() or str(exc)
            raise HypervecClientError(f"gRPC {rpc_name} [{code}]: {detail}") from exc

    @staticmethod
    def _parse(payload: str) -> Any:
        return json.loads(payload) if payload else {}

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        resp = self._call("Health", self._pb2.HealthRequest())
        return {"status": resp.status}

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        resp = self._call("ListCollections", self._pb2.ListCollectionsRequest())
        return list(resp.collections)

    def has_collection(self, collection_name: str) -> bool:
        resp = self._call(
            "HasCollection",
            self._pb2.HasCollectionRequest(collection_name=collection_name),
        )
        return bool(resp.exists)

    def describe_collection(self, collection_name: str) -> dict[str, Any]:
        resp = self._call(
            "DescribeCollection",
            self._pb2.DescribeCollectionRequest(collection_name=collection_name),
        )
        return self._parse(resp.json_payload)

    def create_collection(
        self,
        collection_name: str,
        schema: Any,
        index_params: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        from .schema import CollectionSchema, IndexParams

        schema_dict = schema.to_dict() if isinstance(schema, CollectionSchema) else dict(schema)
        if index_params is None:
            ip_dict = IndexParams().to_dict()
        else:
            ip_dict = (
                index_params.to_dict() if isinstance(index_params, IndexParams) else dict(index_params)
            )
        resp = self._call(
            "CreateCollection",
            self._pb2.CreateCollectionRequest(
                collection_name=collection_name,
                schema_json=json.dumps(schema_dict),
                index_params_json=json.dumps(ip_dict),
            ),
        )
        return self._parse(resp.json_payload)

    def drop_collection(self, collection_name: str) -> dict[str, Any]:
        resp = self._call(
            "DropCollection",
            self._pb2.DropCollectionRequest(collection_name=collection_name),
        )
        return self._parse(resp.json_payload)

    # ------------------------------------------------------------------
    # Data operations
    # ------------------------------------------------------------------

    def insert(self, collection_name: str, data: list[dict[str, Any]]) -> dict[str, Any]:
        resp = self._call(
            "Insert",
            self._pb2.InsertRequest(
                collection_name=collection_name,
                data_json=json.dumps(data),
            ),
        )
        return self._parse(resp.json_payload)

    def flush(self, collection_name: str) -> dict[str, Any]:
        resp = self._call("Flush", self._pb2.FlushRequest(collection_name=collection_name))
        return self._parse(resp.json_payload)

    def load_collection(self, collection_name: str) -> dict[str, Any]:
        resp = self._call(
            "LoadCollection",
            self._pb2.LoadCollectionRequest(collection_name=collection_name),
        )
        return self._parse(resp.json_payload)

    def close_collection(self, collection_name: str) -> dict[str, Any]:
        resp = self._call(
            "CloseCollection",
            self._pb2.CloseCollectionRequest(collection_name=collection_name),
        )
        return self._parse(resp.json_payload)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        *,
        collection_name: str,
        data: Any,
        limit: int,
        search_params: dict[str, Any] | None = None,
        output_fields: list[str] | None = None,
        filter: str | None = None,
        consistency_level: str | None = None,
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        resp = self._call(
            "Search",
            self._pb2.SearchRequest(
                collection_name=collection_name,
                query_json=json.dumps(data),
                limit=int(limit),
                search_params_json=json.dumps(search_params or {}),
                output_fields=list(output_fields or []),
                filter=filter or "",
                consistency_level=consistency_level or "",
            ),
        )
        return json.loads(resp.results_json)

    # ------------------------------------------------------------------
    # Version / sync
    # ------------------------------------------------------------------

    def get_version(self, collection_name: str) -> dict[str, Any]:
        resp = self._call(
            "GetVersion",
            self._pb2.GetVersionRequest(collection_name=collection_name),
        )
        return self._parse(resp.json_payload)

    def sync_check(
        self,
        collection_name: str,
        client_version: int,
        client_checksum: str | None = None,
    ) -> dict[str, Any]:
        resp = self._call(
            "SyncCheck",
            self._pb2.SyncCheckRequest(
                collection_name=collection_name,
                client_version=int(client_version),
                client_checksum=client_checksum or "",
            ),
        )
        return self._parse(resp.json_payload)

    # ------------------------------------------------------------------
    # Index transfer
    # ------------------------------------------------------------------

    def download_index(self, collection_name: str, target_path: Any) -> dict[str, Any]:
        resp = self._call(
            "DownloadIndex",
            self._pb2.DownloadIndexRequest(collection_name=collection_name),
        )
        from pathlib import Path as _Path

        target = _Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resp.data)
        return {
            "collection_name": collection_name,
            "path": str(target),
            "bytes": len(resp.data),
            "version": str(resp.version) if resp.version else None,
            "index_checksum": resp.checksum or None,
            "index_size_bytes": str(resp.size_bytes) if resp.size_bytes else None,
        }

    def upload_index(
        self,
        collection_name: str,
        index_path: Any,
        *,
        version: int | None = None,
        checksum: str | None = None,
    ) -> dict[str, Any]:
        from pathlib import Path as _Path

        data = _Path(index_path).read_bytes()
        resp = self._call(
            "UploadIndex",
            self._pb2.UploadIndexRequest(
                collection_name=collection_name,
                data=data,
                version=int(version) if version is not None else 0,
                checksum=checksum or "",
            ),
        )
        return self._parse(resp.json_payload)
