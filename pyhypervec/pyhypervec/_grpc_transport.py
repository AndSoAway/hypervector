from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .exceptions import HypervecClientError, HypervecGrpcError


DEFAULT_GRPC_MAX_MESSAGE_MB = 256


def grpc_max_message_bytes() -> int:
    raw = os.getenv("HYPERVEC_GRPC_MAX_MESSAGE_MB", str(DEFAULT_GRPC_MAX_MESSAGE_MB))
    try:
        value = int(raw)
    except ValueError as exc:
        raise HypervecClientError(
            f"HYPERVEC_GRPC_MAX_MESSAGE_MB must be an integer, got {raw!r}."
        ) from exc
    if value <= 0:
        raise HypervecClientError("HYPERVEC_GRPC_MAX_MESSAGE_MB must be positive.")
    return value * 1024 * 1024


def _import_grpc():
    try:
        import grpc

        from . import hypervec_pb2 as pb2
        from . import hypervec_pb2_grpc as pb2_grpc
    except ImportError as exc:
        raise HypervecClientError(
            "gRPC transport requires pyhypervec's gRPC dependencies. "
            "Install them with: pip install 'pyhypervec[grpc]'"
        ) from exc
    return grpc, pb2, pb2_grpc


class GrpcTransport:
    def __init__(
        self,
        address: str,
        *,
        timeout: float = 30.0,
        token: str | None = None,
    ) -> None:
        grpc, pb2, pb2_grpc = _import_grpc()
        max_bytes = grpc_max_message_bytes()
        self._grpc = grpc
        self._pb2 = pb2
        self._timeout = timeout
        self._metadata = (("authorization", f"Bearer {token}"),) if token else None
        self._channel = grpc.insecure_channel(
            address,
            options=(
                ("grpc.max_send_message_length", max_bytes),
                ("grpc.max_receive_message_length", max_bytes),
            ),
        )
        self._stub = pb2_grpc.HyperVecStub(self._channel)

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> "GrpcTransport":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _call(self, rpc_name: str, request: Any) -> Any:
        try:
            return getattr(self._stub, rpc_name)(
                request,
                timeout=self._timeout,
                metadata=self._metadata,
            )
        except self._grpc.RpcError as exc:
            code = exc.code()
            code_name = getattr(code, "name", str(code))
            detail = exc.details() or str(exc)
            raise HypervecGrpcError(code_name, detail) from exc

    @staticmethod
    def _json(payload: str) -> Any:
        return json.loads(payload) if payload else {}

    @staticmethod
    def _dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def health(self) -> dict[str, Any]:
        response = self._call("Health", self._pb2.HealthRequest())
        return {"status": response.status}

    def list_collections(self) -> list[str]:
        response = self._call("ListCollections", self._pb2.ListCollectionsRequest())
        return list(response.collections)

    def describe_collections(self) -> list[dict[str, Any]]:
        response = self._call(
            "DescribeCollections", self._pb2.DescribeCollectionsRequest()
        )
        return list(self._json(response.json_payload).get("collections", []))

    def examples(self) -> list[dict[str, Any]]:
        response = self._call("Examples", self._pb2.ExamplesRequest())
        return list(self._json(response.json_payload).get("examples", []))

    def has_collection(self, collection_name: str) -> bool:
        response = self._call(
            "HasCollection",
            self._pb2.HasCollectionRequest(collection_name=collection_name),
        )
        return bool(response.exists)

    def describe_collection(self, collection_name: str) -> dict[str, Any]:
        response = self._call(
            "DescribeCollection",
            self._pb2.DescribeCollectionRequest(collection_name=collection_name),
        )
        return dict(self._json(response.json_payload))

    def create_collection(
        self,
        collection_name: str,
        schema: Any,
        index_params: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        from .schema import CollectionSchema, IndexParams

        schema_dict = schema.to_dict() if isinstance(schema, CollectionSchema) else dict(schema)
        params_dict = (
            IndexParams().to_dict()
            if index_params is None
            else index_params.to_dict()
            if isinstance(index_params, IndexParams)
            else dict(index_params)
        )
        response = self._call(
            "CreateCollection",
            self._pb2.CreateCollectionRequest(
                collection_name=collection_name,
                schema_json=self._dump(schema_dict),
                index_params_json=self._dump(params_dict),
                extra_json=self._dump(kwargs),
            ),
        )
        return dict(self._json(response.json_payload))

    def drop_collection(self, collection_name: str) -> dict[str, Any]:
        response = self._call(
            "DropCollection",
            self._pb2.DropCollectionRequest(collection_name=collection_name),
        )
        return dict(self._json(response.json_payload))

    def insert(self, collection_name: str, data: list[dict[str, Any]]) -> dict[str, Any]:
        response = self._call(
            "Insert",
            self._pb2.InsertRequest(
                collection_name=collection_name,
                data_json=self._dump(data),
            ),
        )
        return dict(self._json(response.json_payload))

    def flush(self, collection_name: str) -> dict[str, Any]:
        response = self._call(
            "Flush", self._pb2.FlushRequest(collection_name=collection_name)
        )
        return dict(self._json(response.json_payload))

    def load_collection(self, collection_name: str) -> dict[str, Any]:
        response = self._call(
            "LoadCollection",
            self._pb2.LoadCollectionRequest(collection_name=collection_name),
        )
        return dict(self._json(response.json_payload))

    def close_collection(self, collection_name: str) -> dict[str, Any]:
        response = self._call(
            "CloseCollection",
            self._pb2.CloseCollectionRequest(collection_name=collection_name),
        )
        return dict(self._json(response.json_payload))

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
        response = self._call(
            "Search",
            self._pb2.SearchRequest(
                collection_name=collection_name,
                query_json=self._dump(data),
                limit=int(limit),
                search_params_json=self._dump(search_params or {}),
                output_fields=list(output_fields or []),
                filter=filter or "",
                consistency_level=consistency_level or "",
                extra_json=self._dump(kwargs),
            ),
        )
        return list(self._json(response.results_json))

    def get_version(self, collection_name: str) -> dict[str, Any]:
        response = self._call(
            "GetVersion",
            self._pb2.GetVersionRequest(collection_name=collection_name),
        )
        return dict(self._json(response.json_payload))

    def sync_check(
        self,
        collection_name: str,
        client_version: int,
        client_checksum: str | None = None,
    ) -> dict[str, Any]:
        response = self._call(
            "SyncCheck",
            self._pb2.SyncCheckRequest(
                collection_name=collection_name,
                client_version=int(client_version),
                client_checksum=client_checksum or "",
            ),
        )
        return dict(self._json(response.json_payload))

    def download_index(self, collection_name: str, target_path: str | Path) -> dict[str, Any]:
        response = self._call(
            "DownloadIndex",
            self._pb2.DownloadIndexRequest(collection_name=collection_name),
        )
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.data)
        return {
            "collection_name": collection_name,
            "path": str(target),
            "bytes": len(response.data),
            "version": str(response.version),
            "index_checksum": response.checksum or None,
            "index_size_bytes": str(response.size_bytes),
        }

    def upload_index(
        self,
        collection_name: str,
        index_path: str | Path,
        *,
        version: int | None = None,
        checksum: str | None = None,
    ) -> dict[str, Any]:
        request = self._pb2.UploadIndexRequest(
            collection_name=collection_name,
            data=Path(index_path).read_bytes(),
            checksum=checksum or "",
        )
        if version is not None:
            request.version = int(version)
        response = self._call("UploadIndex", request)
        return dict(self._json(response.json_payload))

    def download_collection_bundle(
        self,
        collection_name: str,
        target_path: str | Path,
    ) -> dict[str, Any]:
        response = self._call(
            "DownloadCollectionBundle",
            self._pb2.DownloadCollectionBundleRequest(
                collection_name=collection_name
            ),
        )
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.data)
        return {
            "collection_name": collection_name,
            "path": str(target),
            "bytes": len(response.data),
            "version": str(response.version),
            "bundle_format": response.bundle_format or None,
            "bundle_checksum": response.checksum or None,
        }

    def upload_collection_bundle(
        self,
        collection_name: str,
        bundle_path: str | Path,
        checksum: str | None = None,
        mode: str = "replace",
    ) -> dict[str, Any]:
        response = self._call(
            "UploadCollectionBundle",
            self._pb2.UploadCollectionBundleRequest(
                collection_name=collection_name,
                data=Path(bundle_path).read_bytes(),
                checksum=checksum or "",
                mode=mode,
            ),
        )
        return dict(self._json(response.json_payload))

    def purge_collection_data(
        self,
        collection_name: str,
        require_exported: bool = True,
    ) -> dict[str, Any]:
        request = self._pb2.PurgeCollectionDataRequest(
            collection_name=collection_name
        )
        request.require_exported = bool(require_exported)
        response = self._call("PurgeCollectionData", request)
        return dict(self._json(response.json_payload))
