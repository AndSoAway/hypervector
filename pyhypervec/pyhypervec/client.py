from __future__ import annotations

import json
from pathlib import Path
import socket
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

from .exceptions import HypervecClientError, HypervecHTTPError
from .schema import CollectionSchema, IndexParams

try:
    from . import hypervec_service_pb2 as pb2
    from . import hypervec_service_pb2_grpc as pb2_grpc
except Exception:  # pragma: no cover
    pb2 = None
    pb2_grpc = None


class HypervecClient:
    def __init__(
        self,
        uri: str,
        token: str | None = None,
        timeout: float = 30.0,
        http2: bool = False,
    ) -> None:
        parsed = urlparse(uri)
        scheme = parsed.scheme.lower()
        if not scheme:
            raise HypervecClientError(f"invalid HyperVec server URI: {uri}")
        self.transport = "grpc" if scheme == "tcp" else "http"
        normalized_scheme = "http" if scheme == "tcp" else scheme
        if normalized_scheme not in {"http", "https"}:
            raise HypervecClientError(f"unsupported HyperVec server URI scheme: {parsed.scheme}")
        normalized = parsed._replace(scheme=normalized_scheme).geturl()
        self.uri = normalized.rstrip("/") + "/"
        self.token = token
        self.timeout = timeout
        self.http2 = http2 or normalized_scheme == "https"
        self._grpc_stub_instance = None

    @staticmethod
    def create_schema(
        *,
        auto_id: bool = False,
        enable_dynamic_field: bool = True,
        description: str = "",
    ) -> CollectionSchema:
        return CollectionSchema(
            auto_id=auto_id,
            enable_dynamic_field=enable_dynamic_field,
            description=description,
        )

    def prepare_index_params(self) -> IndexParams:
        return IndexParams()

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
    ) -> Any:
        if self.transport == "grpc":
            return self._request_grpc_json(method, path, body=body)

        if self.http2:
            data = None
            if body is not None:
                data = json.dumps(body, separators=(",", ":")).encode("utf-8")
            raw, _ = self._request_http2(
                method,
                path,
                body=data,
                content_type="application/json",
            )
            if not raw:
                return None
            return json.loads(raw.decode("utf-8"))

        data = None
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if body is not None:
            data = json.dumps(body, separators=(",", ":")).encode("utf-8")

        req = Request(
            urljoin(self.uri, path.lstrip("/")),
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(message)
                message = str(parsed.get("detail", message))
            except Exception:
                pass
            raise HypervecHTTPError(exc.code, message) from exc

        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))

    def _request_bytes(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str = "application/octet-stream",
    ) -> tuple[bytes, dict[str, str]]:
        if self.transport == "grpc":
            return self._request_grpc_bytes(method, path, body=body, content_type=content_type)

        if self.http2:
            return self._request_http2(
                method,
                path,
                body=body,
                content_type=content_type,
            )

        headers = {"Content-Type": content_type}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = Request(
            urljoin(self.uri, path.lstrip("/")),
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                resp_headers = dict(resp.headers.items())
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(message)
                message = str(parsed.get("detail", message))
            except Exception:
                pass
            raise HypervecHTTPError(exc.code, message) from exc
        return raw, resp_headers

    def _request_http2(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str = "application/octet-stream",
    ) -> tuple[bytes, dict[str, str]]:
        parsed = urlparse(urljoin(self.uri, path.lstrip("/")))
        if parsed.scheme == "https":
            return self._request_h2_tls(
                method,
                path,
                body=body,
                content_type=content_type,
            )
        if parsed.scheme == "http":
            return self._request_h2c(
                method,
                path,
                body=body,
                content_type=content_type,
            )
        raise HypervecClientError(f"unsupported URI scheme for HTTP/2: {parsed.scheme}")

    def _request_h2_tls(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        content_type: str,
    ) -> tuple[bytes, dict[str, str]]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise HypervecClientError(
                "HTTP/2 over TLS requires httpx[http2]. Install pyhypervec with dependencies."
            ) from exc

        headers = {"Content-Type": content_type}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        with httpx.Client(http2=True, timeout=self.timeout) as client:
            response = client.request(
                method,
                urljoin(self.uri, path.lstrip("/")),
                content=body,
                headers=headers,
            )
        if response.status_code >= 400:
            raise HypervecHTTPError(response.status_code, self._error_message(response.content))
        return response.content, dict(response.headers)

    def _request_h2c(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        content_type: str,
    ) -> tuple[bytes, dict[str, str]]:
        try:
            from h2.config import H2Configuration
            from h2.connection import H2Connection
            from h2.events import DataReceived, ResponseReceived, StreamEnded, TrailersReceived
        except ImportError as exc:  # pragma: no cover
            raise HypervecClientError(
                "HTTP/2 cleartext requires h2. Install pyhypervec with dependencies."
            ) from exc

        parsed = urlparse(urljoin(self.uri, path.lstrip("/")))
        host = parsed.hostname
        if not host:
            raise HypervecClientError(f"invalid HyperVec server URI: {self.uri}")
        port = parsed.port or 80
        request_path = parsed.path or "/"
        if parsed.query:
            request_path += "?" + parsed.query
        authority = host if parsed.port is None else f"{host}:{port}"

        conn = H2Connection(config=H2Configuration(client_side=True, header_encoding="utf-8"))
        response_headers: dict[str, str] = {}
        response_body = bytearray()
        stream_ended = False

        with socket.create_connection((host, port), timeout=self.timeout) as sock:
            sock.settimeout(self.timeout)
            conn.initiate_connection()
            sock.sendall(conn.data_to_send())

            stream_id = conn.get_next_available_stream_id()
            headers = [
                (":method", method.upper()),
                (":authority", authority),
                (":scheme", "http"),
                (":path", request_path),
                ("content-type", content_type),
            ]
            if self.token:
                headers.append(("authorization", f"Bearer {self.token}"))
            conn.send_headers(stream_id, headers, end_stream=body is None)
            sock.sendall(conn.data_to_send())
            if body is not None:
                self._send_h2c_body(sock, conn, stream_id, body)

            while not stream_ended:
                data = sock.recv(65535)
                if not data:
                    break
                events = conn.receive_data(data)
                for event in events:
                    if isinstance(event, ResponseReceived):
                        response_headers.update(dict(event.headers))
                    elif isinstance(event, TrailersReceived):
                        response_headers.update(dict(event.headers))
                    elif isinstance(event, DataReceived):
                        response_body.extend(event.data)
                        conn.acknowledge_received_data(event.flow_controlled_length, event.stream_id)
                    elif isinstance(event, StreamEnded):
                        stream_ended = True
                out = conn.data_to_send()
                if out:
                    sock.sendall(out)

        status = int(response_headers.get(":status", "0") or "0")
        raw = bytes(response_body)
        if status >= 400:
            raise HypervecHTTPError(status, self._error_message(raw))
        if status == 0:
            raise HypervecClientError("HTTP/2 response did not include a status code.")
        return raw, response_headers

    def _send_h2c_body(self, sock: socket.socket, conn: Any, stream_id: int, body: bytes) -> None:
        offset = 0
        while offset < len(body):
            window = conn.local_flow_control_window(stream_id)
            if window <= 0:
                data = sock.recv(65535)
                if not data:
                    raise HypervecClientError("HTTP/2 connection closed while sending request body.")
                conn.receive_data(data)
                continue
            chunk_size = min(window, conn.max_outbound_frame_size, len(body) - offset)
            conn.send_data(stream_id, body[offset:offset + chunk_size], end_stream=False)
            offset += chunk_size
            out = conn.data_to_send()
            if out:
                sock.sendall(out)
        conn.end_stream(stream_id)
        out = conn.data_to_send()
        if out:
            sock.sendall(out)

    def _grpc_channel_target(self) -> str:
        parsed = urlparse(self.uri)
        if not parsed.netloc:
            raise HypervecClientError(f"invalid HyperVec server URI: {self.uri}")
        return parsed.netloc

    def _grpc_stub(self):
        if pb2 is None or pb2_grpc is None:
            raise HypervecClientError("gRPC transport requires grpcio and generated HyperVec protobuf bindings.")
        if self._grpc_stub_instance is None:
            import grpc

            channel = grpc.insecure_channel(
                self._grpc_channel_target(),
                options=[
                    ("grpc.max_send_message_length", 1024 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
                    ("grpc.max_concurrent_streams", 256),
                    ("grpc.keepalive_time_ms", 60000),
                    ("grpc.keepalive_timeout_ms", 20000),
                    ("grpc.http2.min_time_between_pings_ms", 30000),
                    ("grpc.http2.max_pings_without_data", 0),
                ],
            )
            self._grpc_stub_instance = pb2_grpc.HypervecServiceStub(channel)
        return self._grpc_stub_instance

    @staticmethod
    def _collection_from_path(path: str) -> str:
        parts = urlparse(path).path.strip("/").split("/")
        if len(parts) < 2 or parts[0] != "collections":
            raise HypervecClientError(f"invalid collection path: {path}")
        return parts[1]

    @staticmethod
    def _json_response(response: Any) -> Any:
        return json.loads(response.json or "{}")

    def _request_grpc_json(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
    ) -> Any:
        stub = self._grpc_stub()
        payload = body or {}
        if path == "/health":
            response = stub.Health(pb2.HealthRequest())
            return {"status": response.status}
        if path == "/collections":
            response = stub.ListCollections(pb2.ListCollectionsRequest())
            return {"collections": list(response.collections)}
        if method == "DELETE" and path.startswith("/collections/"):
            response = stub.DropCollection(pb2.DropCollectionRequest(collection_name=self._collection_from_path(path)))
            return self._json_response(response)
        if path.endswith("/exists"):
            response = stub.HasCollection(pb2.HasCollectionRequest(collection_name=self._collection_from_path(path)))
            return {"collection_name": response.collection_name, "exists": bool(response.exists)}
        if path.endswith("/describe"):
            return self._json_response(stub.DescribeCollection(pb2.DescribeCollectionRequest(collection_name=self._collection_from_path(path))))
        if path.endswith("/create"):
            response = stub.CreateCollection(
                pb2.CreateCollectionRequest(
                    collection_name=self._collection_from_path(path),
                    schema_json=json.dumps(payload.get("schema", {}), separators=(",", ":")),
                    index_params_json=json.dumps(payload.get("index_params", {}), separators=(",", ":")),
                )
            )
            return self._json_response(response)
        if path.endswith("/insert"):
            response = stub.Insert(
                pb2.InsertRequest(
                    collection_name=self._collection_from_path(path),
                    data_json=json.dumps(payload.get("data", []), separators=(",", ":")),
                )
            )
            return self._json_response(response)
        if path.endswith("/flush"):
            return self._json_response(stub.Flush(pb2.CollectionRequest(collection_name=self._collection_from_path(path))))
        if path.endswith("/load"):
            return self._json_response(stub.LoadCollection(pb2.CollectionRequest(collection_name=self._collection_from_path(path))))
        if path.endswith("/close"):
            return self._json_response(stub.CloseCollection(pb2.CollectionRequest(collection_name=self._collection_from_path(path))))
        if path.endswith("/version"):
            return self._json_response(stub.GetVersion(pb2.CollectionRequest(collection_name=self._collection_from_path(path))))
        if path.endswith("/sync-check"):
            response = stub.SyncCheck(
                pb2.SyncCheckRequest(
                    collection_name=self._collection_from_path(path),
                    client_version=int(payload.get("client_version", 0)),
                    client_checksum=str(payload.get("client_checksum", "")),
                )
            )
            return self._json_response(response)
        if path.endswith("/search"):
            search_data = payload.get("data", [])
            data_bytes = b""
            dim = 0
            num_rows = 0
            if search_data is not None:
                import numpy as np
                arr = np.asarray(search_data, dtype=np.float32)
                if arr.size:
                    if arr.ndim == 1:
                        arr = arr.reshape(1, arr.shape[0])
                    data_bytes = arr.tobytes()
                    dim = arr.shape[1] if arr.ndim == 2 else 0
                    num_rows = arr.shape[0]
            response = stub.Search(
                pb2.SearchRequest(
                    collection_name=self._collection_from_path(path),
                    data_json="" if data_bytes else json.dumps(search_data, separators=(",", ":")),
                    limit=int(payload.get("limit", 0)),
                    search_params_json=json.dumps(payload.get("search_params", {}), separators=(",", ":")),
                    output_fields=list(payload.get("output_fields", [])),
                    filter=str(payload.get("filter", "")),
                    consistency_level=str(payload.get("consistency_level", "")),
                    data_bytes=data_bytes,
                    dim=dim,
                    num_rows=num_rows,
                )
            )
            return {
                "results": [
                    [
                        {
                            "id": hit.id,
                            "distance": hit.distance,
                            "entity": json.loads(hit.entity_json) if hit.entity_json else {},
                        }
                        for hit in row.hits
                    ]
                    for row in response.results
                ]
            }
        raise HypervecClientError(f"unsupported gRPC JSON path: {method} {path}")

    def _request_grpc_bytes(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str = "application/octet-stream",
    ) -> tuple[bytes, dict[str, str]]:
        del content_type
        if not urlparse(path).path.endswith("/index"):
            raise HypervecClientError(f"unsupported gRPC bytes path: {method} {path}")
        stub = self._grpc_stub()
        collection_name = self._collection_from_path(path)
        if method == "GET":
            response = stub.DownloadIndex(pb2.CollectionRequest(collection_name=collection_name))
            return response.data, {
                "X-Hypervec-Collection-Version": str(response.version),
                "X-Hypervec-Index-Checksum": response.index_checksum,
                "X-Hypervec-Index-Size": str(response.index_size_bytes),
            }
        if method == "PUT":
            query = parse_qs(urlparse(path).query)
            response = stub.UploadIndex(
                pb2.UploadIndexRequest(
                    collection_name=collection_name,
                    data=body or b"",
                    version=int((query.get("version") or ["0"])[0] or 0),
                    checksum=(query.get("checksum") or [""])[0],
                )
            )
            return (response.json or "{}").encode("utf-8"), {}
        raise HypervecClientError(f"unsupported gRPC bytes method: {method}")

    @staticmethod
    def _error_message(raw: bytes) -> str:
        message = raw.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(message)
            return str(parsed.get("detail", message))
        except Exception:
            return message

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def list_collections(self) -> list[str]:
        res = self._request("GET", "/collections")
        return list(res.get("collections", []))

    def has_collection(self, collection_name: str) -> bool:
        res = self._request("GET", f"/collections/{collection_name}/exists")
        return bool(res.get("exists", False))

    def describe_collection(self, collection_name: str) -> dict[str, Any]:
<<<<<<< ours
        desc = self._request("GET", f"/collections/{collection_name}/describe")
        return self._normalize_description(desc)

    def describe_collections(self) -> list[dict[str, Any]]:
        res = self._request("GET", "/collections/describe")
        return [
            self._normalize_description(desc)
            for desc in list(res.get("collections", []))
        ]

    def examples(self) -> list[dict[str, Any]]:
        res = self._request("GET", "/examples")
        return list(res.get("examples", []))

    def get_collection_stats(
        self,
        collection_name: str,
        timeout: float | None = None,
    ) -> dict[str, int]:
        del timeout
        desc = self.describe_collection(collection_name)
        return {"row_count": int(desc.get("total") or 0)}
=======
        return self._request("GET", f"/collections/{collection_name}/describe")
>>>>>>> theirs

    def create_collection(
        self,
        collection_name: str,
        *,
        schema: CollectionSchema,
        index_params: IndexParams | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        body = {
            "schema": schema.to_dict(),
            "index_params": (index_params or IndexParams()).to_dict(),
        }
        body.update(kwargs)
        return self._request("POST", f"/collections/{collection_name}/create", body=body)

    def drop_collection(self, collection_name: str) -> dict[str, Any]:
        return self._request("DELETE", f"/collections/{collection_name}")

    def insert(self, collection_name: str, data: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/collections/{collection_name}/insert",
            body={"data": data},
        )

    def flush(self, collection_name: str) -> dict[str, Any]:
        return self._request("POST", f"/collections/{collection_name}/flush", body={})

    def load_collection(self, collection_name: str) -> dict[str, Any]:
        return self._request("POST", f"/collections/{collection_name}/load", body={})

    def close_collection(self, collection_name: str) -> dict[str, Any]:
        return self._request("POST", f"/collections/{collection_name}/close", body={})

    def get_version(self, collection_name: str) -> dict[str, Any]:
        return self._request("GET", f"/collections/{collection_name}/version")

    def sync_check(
        self,
        collection_name: str,
        client_version: int,
        client_checksum: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"client_version": int(client_version)}
        if client_checksum:
            body["client_checksum"] = client_checksum
        return self._request(
            "POST",
            f"/collections/{collection_name}/sync-check",
            body=body,
        )

    def download_index(self, collection_name: str, target_path: str | Path) -> dict[str, Any]:
        raw, headers = self._request_bytes("GET", f"/collections/{collection_name}/index")
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return {
            "collection_name": collection_name,
            "path": str(target),
            "bytes": len(raw),
            "version": self._header_value(headers, "X-Hypervec-Collection-Version"),
            "index_checksum": self._header_value(headers, "X-Hypervec-Index-Checksum"),
            "index_size_bytes": self._header_value(headers, "X-Hypervec-Index-Size"),
        }

    @staticmethod
    def _header_value(headers: dict[str, str], name: str) -> str | None:
        return headers.get(name) or headers.get(name.lower())

    def upload_index(
        self,
        collection_name: str,
        index_path: str | Path,
        *,
        version: int | None = None,
        checksum: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if version is not None:
            params["version"] = int(version)
        if checksum:
            params["checksum"] = checksum
        query = f"?{urlencode(params)}" if params else ""
        raw = Path(index_path).read_bytes()
        body, _ = self._request_bytes(
            "PUT",
            f"/collections/{collection_name}/index{query}",
            body=raw,
        )
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

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
        body = {
            "data": data,
            "limit": int(limit),
            "search_params": dict(search_params or {}),
            "output_fields": list(output_fields or []),
        }
        if filter:
            body["filter"] = filter
        if consistency_level:
            body["consistency_level"] = consistency_level
        body.update(kwargs)
        res = self._request("POST", f"/collections/{collection_name}/search", body=body)
        return list(res.get("results", []))
