from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from .exceptions import HypervecHTTPError
from .schema import CollectionSchema, IndexParams


class HypervecClient:
    def __init__(
        self,
        uri: str,
        token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.uri = uri.rstrip("/") + "/"
        self.token = token
        self.timeout = timeout

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

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def list_collections(self) -> list[str]:
        res = self._request("GET", "/collections")
        return list(res.get("collections", []))

    def has_collection(self, collection_name: str) -> bool:
        res = self._request("GET", f"/collections/{collection_name}/exists")
        return bool(res.get("exists", False))

    def describe_collection(self, collection_name: str) -> dict[str, Any]:
        return self._request("GET", f"/collections/{collection_name}/describe")

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
