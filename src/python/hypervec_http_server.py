# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the License) found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import tempfile
from typing import Any

from .hypervec_server_engine import ConflictError, HypervecServerEngine


def _require_fastapi():
    try:
        from fastapi import FastAPI, HTTPException, Query, Request, Response
        from fastapi.responses import FileResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "HyperVec HTTP server requires fastapi and pydantic."
        ) from exc
    return FastAPI, HTTPException, Query, Request, Response, FileResponse, BaseModel, Field


def create_app(
    *,
    data_root: str,
    engine: HypervecServerEngine | None = None,
) -> Any:
    FastAPI, HTTPException, Query, Request, Response, FileResponse, BaseModel, Field = _require_fastapi()
    engine = engine or HypervecServerEngine(data_root)

    class CreateCollectionRequest(BaseModel):
        collection_schema: dict[str, Any] = Field(alias="schema")
        index_params: dict[str, Any] = Field(default_factory=lambda: {"indexes": []})

    class InsertRequest(BaseModel):
        data: list[dict[str, Any]]

    class SearchRequest(BaseModel):
        data: list[list[float]]
        limit: int
        search_params: dict[str, Any] = Field(default_factory=dict)
        output_fields: list[str] = Field(default_factory=list)
        filter: str = ""
        consistency_level: str | None = None

    class SyncCheckRequest(BaseModel):
        client_version: int
        client_checksum: str | None = None

    def fail(exc: Exception) -> HTTPException:
        if isinstance(exc, FileNotFoundError):
            return HTTPException(status_code=404, detail=str(exc))
        if isinstance(exc, (FileExistsError, ConflictError)):
            return HTTPException(status_code=409, detail=str(exc))
        if isinstance(exc, ValueError):
            return HTTPException(status_code=400, detail=str(exc))
        return HTTPException(status_code=500, detail=str(exc))

    app = FastAPI(title="HyperVec HTTP Server", version="1")
    app.state.hypervec_engine = engine

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/collections")
    def list_collections() -> dict[str, list[str]]:
        return {"collections": engine.list_collections()}

    @app.get("/collections/describe")
    def describe_collections() -> dict[str, Any]:
        try:
            return {"collections": engine.describe_collections()}
        except Exception as exc:
            raise fail(exc)

    @app.get("/collections/{collection_name}/exists")
    def has_collection(collection_name: str) -> dict[str, Any]:
        try:
            return {
                "collection_name": collection_name,
                "exists": engine.has_collection(collection_name),
            }
        except Exception as exc:
            raise fail(exc)

    @app.get("/collections/{collection_name}/describe")
    def describe_collection(collection_name: str) -> dict[str, Any]:
        try:
            return engine.describe_collection(collection_name)
        except Exception as exc:
            raise fail(exc)

    @app.post("/collections/{collection_name}/create")
    def create_collection(
        collection_name: str,
        request: CreateCollectionRequest,
    ) -> dict[str, Any]:
        try:
            return engine.create_collection(
                collection_name,
                schema=request.collection_schema,
                index_params=request.index_params,
            )
        except Exception as exc:
            raise fail(exc)

    @app.delete("/collections/{collection_name}")
    def drop_collection(collection_name: str) -> dict[str, Any]:
        try:
            return engine.drop_collection(collection_name)
        except Exception as exc:
            raise fail(exc)

    @app.post("/collections/{collection_name}/insert")
    def insert(collection_name: str, request: InsertRequest) -> dict[str, Any]:
        try:
            return engine.insert(collection_name, request.data)
        except Exception as exc:
            raise fail(exc)

    @app.post("/collections/{collection_name}/flush")
    def flush(collection_name: str) -> dict[str, Any]:
        try:
            return engine.flush(collection_name)
        except Exception as exc:
            raise fail(exc)

    @app.post("/collections/{collection_name}/load")
    def load_collection(collection_name: str) -> dict[str, Any]:
        try:
            return engine.load_collection(collection_name)
        except Exception as exc:
            raise fail(exc)

    @app.post("/collections/{collection_name}/close")
    def close_collection(collection_name: str) -> dict[str, Any]:
        try:
            return engine.close_collection(collection_name)
        except Exception as exc:
            raise fail(exc)

    @app.post("/collections/{collection_name}/search")
    def search(collection_name: str, request: SearchRequest) -> dict[str, Any]:
        try:
            return {
                "results": engine.search(
                    collection_name,
                    data=request.data,
                    limit=request.limit,
                    search_params=request.search_params,
                    output_fields=request.output_fields,
                    filter=request.filter,
                    consistency_level=request.consistency_level,
                )
            }
        except Exception as exc:
            raise fail(exc)

    @app.get("/collections/{collection_name}/version")
    def get_version(collection_name: str) -> dict[str, Any]:
        try:
            return engine.get_version(collection_name)
        except Exception as exc:
            raise fail(exc)

    @app.post("/collections/{collection_name}/sync-check")
    def sync_check(collection_name: str, request: SyncCheckRequest) -> dict[str, Any]:
        try:
            return engine.sync_check(
                collection_name,
                client_version=request.client_version,
                client_checksum=request.client_checksum,
            )
        except Exception as exc:
            raise fail(exc)

    @app.get("/collections/{collection_name}/index")
    def download_index(collection_name: str):
        try:
            path = engine.index_path_for_download(collection_name)
            version = engine.get_version(collection_name)
            headers = {}
            if version.get("version") is not None:
                headers["X-Hypervec-Collection-Version"] = str(version["version"])
            if version.get("index_checksum"):
                headers["X-Hypervec-Index-Checksum"] = str(version["index_checksum"])
            if version.get("index_size_bytes") is not None:
                headers["X-Hypervec-Index-Size"] = str(version["index_size_bytes"])
            return FileResponse(
                str(path),
                media_type="application/octet-stream",
                filename=f"{collection_name}.hypervec",
                headers=headers,
            )
        except Exception as exc:
            raise fail(exc)

    @app.put("/collections/{collection_name}/index")
    async def upload_index(
        collection_name: str,
        request: Request,
        version: int | None = Query(default=None),
        checksum: str | None = Query(default=None),
    ) -> dict[str, Any]:
        try:
            body = await request.body()
            if not body:
                raise ValueError("uploaded index body is empty.")
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(body)
                tmp_path = f.name
            try:
                return engine.upload_index(
                    collection_name,
                    tmp_path,
                    version=version,
                    checksum=checksum,
                )
            finally:
                import os

                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        except Exception as exc:
            raise fail(exc)

    # ------------------------------------------------------------------
    # Bundle download / upload / purge-data
    # ------------------------------------------------------------------

    class PurgeDataRequest(BaseModel):
        require_exported: bool = True

    @app.get("/collections/{collection_name}/bundle")
    def download_bundle(collection_name: str) -> Response:
        """Download a collection data bundle (scalar rows + index).

        Returns a ZIP file with manifest.json, index.hypervec, scalar.jsonl.
        The bundle is generated on-the-fly into the collection's directory.
        """
        try:
            result = engine.export_collection_bundle(collection_name)
        except Exception as exc:
            raise fail(exc)
        bundle_path = result["path"]
        with open(bundle_path, "rb") as f:
            data = f.read()
        import os as _os
        try:
            _os.unlink(bundle_path)
        except OSError:
            pass
        headers = {
            "Content-Disposition": (
                f'attachment; filename="{collection_name}.hypervec-bundle"'
            ),
            "X-Hypervec-Collection-Version": str(result["version"]),
            "X-Hypervec-Bundle-Format": result["bundle_format"],
            "X-Hypervec-Bundle-Checksum": result["bundle_checksum"],
            "X-Hypervec-Bundle-Size": str(result["bytes"]),
        }
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers=headers,
        )

    @app.put("/collections/{collection_name}/bundle")
    async def upload_bundle(
        collection_name: str,
        request: Request,
        checksum: str | None = Query(default=None),
        mode: str = Query(default="replace"),
    ) -> dict[str, Any]:
        """Upload (restore) a collection data bundle.

        Restores scalar rows and index.hypervec from the uploaded bundle.
        The collection metadata entry must already exist (created beforehand).
        """
        try:
            body = await request.body()
            if not body:
                raise ValueError("uploaded bundle body is empty.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".hypervec-bundle") as f:
                f.write(body)
                tmp_path = f.name
            try:
                return engine.import_collection_bundle(
                    collection_name,
                    tmp_path,
                    checksum=checksum,
                    mode=mode,
                )
            finally:
                import os
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        except Exception as exc:
            raise fail(exc)

    @app.post("/collections/{collection_name}/purge-data")
    def purge_data(
        collection_name: str,
        request: PurgeDataRequest = PurgeDataRequest(),
    ) -> dict[str, Any]:
        """Purge user data (index + scalar rows) while keeping collection metadata.

        Safe for UltraRAG user-exit flow: call download_bundle first, then
        purge-data.  Set require_exported=false only if you intentionally want
        to destroy data without a prior bundle export.
        """
        try:
            return engine.purge_collection_data(
                collection_name,
                require_exported=request.require_exported,
            )
        except Exception as exc:
            raise fail(exc)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HyperVec HTTP server.")
    parser.add_argument("--data-root", required=True, help="Collection data root.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", default=8080, type=int, help="Bind port.")
    parser.add_argument(
        "--server",
        choices=("hypercorn", "uvicorn"),
        default="hypercorn",
        help="ASGI server implementation. Hypercorn is the default because it supports HTTP/2.",
    )
    parser.add_argument("--log-level", default="info", help="ASGI server log level.")
    parser.add_argument("--certfile", default=None, help="TLS certificate file for HTTP/2 over TLS.")
    parser.add_argument("--keyfile", default=None, help="TLS private key file for HTTP/2 over TLS.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    app = create_app(data_root=args.data_root)

    if args.server == "uvicorn":
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("HyperVec HTTP server requires uvicorn.") from exc

        if args.certfile or args.keyfile:
            if not (args.certfile and args.keyfile):
                raise RuntimeError("--certfile and --keyfile must be provided together.")
            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                log_level=args.log_level,
                ssl_certfile=args.certfile,
                ssl_keyfile=args.keyfile,
            )
        else:
            uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
        return

    try:
        import asyncio
        from hypercorn.asyncio import serve
        from hypercorn.config import Config
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "HyperVec HTTP/2 server requires hypercorn. Install hypervec[server] "
            "or run: python -m pip install hypercorn h2"
        ) from exc

    if bool(args.certfile) != bool(args.keyfile):
        raise RuntimeError("--certfile and --keyfile must be provided together.")

    config = Config()
    config.bind = [f"{args.host}:{args.port}"]
    config.loglevel = args.log_level
    config.certfile = args.certfile
    config.keyfile = args.keyfile
    config.alpn_protocols = ["h2", "http/1.1"]
    asyncio.run(serve(app, config))


if __name__ == "__main__":
    main()
