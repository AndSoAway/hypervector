from __future__ import annotations

import json
from pathlib import Path
import socket
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

from .exceptions import HypervecClientError, HypervecHTTPError
from .schema import CollectionSchema, IndexParams


class HypervecClient:
    def __init__(
        self,
        uri: str,
        token: str | None = None,
        timeout: float = 30.0,
        http2: bool = False,
    ) -> None:
        self.uri = uri.rstrip("/") + "/"
        self.token = token
        self.timeout = timeout
        self.http2 = http2

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

    @staticmethod
    def _normalize_description(desc: dict[str, Any]) -> dict[str, Any]:
        if isinstance(desc, dict) and "description" not in desc:
            schema = desc.get("schema")
            if isinstance(schema, dict):
                desc["description"] = str(schema.get("description") or "")
        return desc

    def describe_collection(self, collection_name: str) -> dict[str, Any]:
        desc = self._request("GET", f"/collections/{collection_name}/describe")
        return self._normalize_description(desc)

    def describe_collections(self) -> list[dict[str, Any]]:
        res = self._request("GET", "/collections/describe")
        return [
            self._normalize_description(desc)
            for desc in list(res.get("collections", []))
        ]

    def get_collection_stats(
        self,
        collection_name: str,
        timeout: float | None = None,
    ) -> dict[str, int]:
        del timeout
        desc = self.describe_collection(collection_name)
        return {"row_count": int(desc.get("total") or 0)}

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

    def get_examples(self, index_type: str | None = None) -> dict[str, Any]:
        """
        Get usage examples for index types.

        Args:
            index_type: The index type (e.g., "HNSW", "Flat").
                       If None, returns list of all supported index types.

        Returns:
            dict: Example data including description, parameters, and code samples.

        Example:
            # List all supported index types
            examples = client.get_examples()
            print(examples["supported_indexes"])

            # Get HNSW examples
            hnsw_examples = client.get_examples("HNSW")
            print(hnsw_examples["description"])
            print(hnsw_examples["example_code"]["python"]["create"])
        """
        if index_type:
            return self._request("GET", f"/examples/{index_type}")
        else:
            return self._request("GET", "/examples")
