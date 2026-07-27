# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the License) found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from .hypervec_config import (
    CONFIG_OPTIONS,
    ConfigError,
    HypervecConfig,
    cli_overrides_from_namespace,
    configure_logging,
    export_sample_config,
    resolve_config,
)

if TYPE_CHECKING:
    from .hypervec_server_engine import HypervecServerEngine


def _require_fastapi():
    try:
        from fastapi import FastAPI, HTTPException, Query, Request
        from fastapi.responses import FileResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "HyperVec HTTP server requires fastapi and pydantic."
        ) from exc
    return FastAPI, HTTPException, Query, Request, FileResponse, BaseModel, Field


def create_app(
    *,
    data_root: str,
    engine: Optional["HypervecServerEngine"] = None,
) -> Any:
    FastAPI, HTTPException, Query, Request, FileResponse, BaseModel, Field = _require_fastapi()
    if engine is None:
        from .hypervec_server_engine import HypervecServerEngine

        engine = HypervecServerEngine(data_root)

    class CreateCollectionRequest(BaseModel):
        collection_schema: Dict[str, Any] = Field(alias="schema")
        index_params: Dict[str, Any] = Field(default_factory=lambda: {"indexes": []})

    class InsertRequest(BaseModel):
        data: List[Dict[str, Any]]

    class SearchRequest(BaseModel):
        data: List[List[float]]
        limit: int
        search_params: Dict[str, Any] = Field(default_factory=dict)
        output_fields: List[str] = Field(default_factory=list)
        filter: str = ""
        consistency_level: Optional[str] = None

    class SyncCheckRequest(BaseModel):
        client_version: int
        client_checksum: Optional[str] = None

    def fail(exc: Exception) -> HTTPException:
        if isinstance(exc, FileNotFoundError):
            return HTTPException(status_code=404, detail=str(exc))
        if isinstance(exc, FileExistsError):
            return HTTPException(status_code=409, detail=str(exc))
        if isinstance(exc, ValueError):
            return HTTPException(status_code=400, detail=str(exc))
        return HTTPException(status_code=500, detail=str(exc))

    app = FastAPI(title="HyperVec HTTP Server", version="1")
    app.state.hypervec_engine = engine

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/collections")
    def list_collections() -> Dict[str, List[str]]:
        return {"collections": engine.list_collections()}

    @app.get("/collections/describe")
    def describe_collections() -> Dict[str, Any]:
        try:
            return {"collections": engine.describe_collections()}
        except Exception as exc:
            raise fail(exc)

    @app.get("/examples")
    def examples() -> Dict[str, Any]:
        try:
            return {"examples": engine.supported_index_examples()}
        except Exception as exc:
            raise fail(exc)

    @app.get("/collections/{collection_name}/exists")
    def has_collection(collection_name: str) -> Dict[str, Any]:
        try:
            return {
                "collection_name": collection_name,
                "exists": engine.has_collection(collection_name),
            }
        except Exception as exc:
            raise fail(exc)

    @app.get("/collections/{collection_name}/describe")
    def describe_collection(collection_name: str) -> Dict[str, Any]:
        try:
            return engine.describe_collection(collection_name)
        except Exception as exc:
            raise fail(exc)

    @app.post("/collections/{collection_name}/create")
    def create_collection(
        collection_name: str,
        request: CreateCollectionRequest,
    ) -> Dict[str, Any]:
        try:
            return engine.create_collection(
                collection_name,
                schema=request.collection_schema,
                index_params=request.index_params,
            )
        except Exception as exc:
            raise fail(exc)

    @app.delete("/collections/{collection_name}")
    def drop_collection(collection_name: str) -> Dict[str, Any]:
        try:
            return engine.drop_collection(collection_name)
        except Exception as exc:
            raise fail(exc)

    @app.post("/collections/{collection_name}/insert")
    def insert(collection_name: str, request: InsertRequest) -> Dict[str, Any]:
        try:
            return engine.insert(collection_name, request.data)
        except Exception as exc:
            raise fail(exc)

    @app.post("/collections/{collection_name}/flush")
    def flush(collection_name: str) -> Dict[str, Any]:
        try:
            return engine.flush(collection_name)
        except Exception as exc:
            raise fail(exc)

    @app.post("/collections/{collection_name}/load")
    def load_collection(collection_name: str) -> Dict[str, Any]:
        try:
            return engine.load_collection(collection_name)
        except Exception as exc:
            raise fail(exc)

    @app.post("/collections/{collection_name}/close")
    def close_collection(collection_name: str) -> Dict[str, Any]:
        try:
            return engine.close_collection(collection_name)
        except Exception as exc:
            raise fail(exc)

    @app.post("/collections/{collection_name}/search")
    def search(collection_name: str, request: SearchRequest) -> Dict[str, Any]:
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
    def get_version(collection_name: str) -> Dict[str, Any]:
        try:
            return engine.get_version(collection_name)
        except Exception as exc:
            raise fail(exc)

    @app.post("/collections/{collection_name}/sync-check")
    def sync_check(collection_name: str, request: SyncCheckRequest) -> Dict[str, Any]:
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
        version: Optional[int] = Query(default=None),
        checksum: Optional[str] = Query(default=None),
    ) -> Dict[str, Any]:
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

    return app


def _config_option(cli_dest: str):
    """Return CLI-facing metadata without duplicating choices in the parser."""

    return next(option for option in CONFIG_OPTIONS if option.cli_dest == cli_dest)


def _add_boolean_option(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    help_text: str,
    disable_help_text: str,
) -> None:
    """Add a bool option while preserving explicit-vs-omitted semantics."""

    option = f"--{name}"
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            option,
            action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
            help=help_text,
        )
        return

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        option,
        dest=name.replace("-", "_"),
        action="store_true",
        default=argparse.SUPPRESS,
        help=help_text,
    )
    group.add_argument(
        f"--no-{name}",
        dest=name.replace("-", "_"),
        action="store_false",
        default=argparse.SUPPRESS,
        help=disable_help_text,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the backward-compatible CLI and suppress business defaults."""

    parser = argparse.ArgumentParser(description="Run the HyperVec HTTP server.")
    parser.add_argument("--config", help="INI configuration file.")
    parser.add_argument(
        "--export-sample-config",
        metavar="PATH",
        help="Write a sample INI configuration and exit.",
    )
    parser.add_argument(
        "--data-root",
        default=argparse.SUPPRESS,
        help="Collection data root.",
    )
    parser.add_argument(
        "--host",
        default=argparse.SUPPRESS,
        help="Bind host.",
    )
    parser.add_argument(
        "--port",
        default=argparse.SUPPRESS,
        type=int,
        help="Bind port.",
    )
    parser.add_argument(
        "--server",
        choices=_config_option("server").choices,
        default=argparse.SUPPRESS,
        help="ASGI server implementation. Hypercorn is the default because it supports HTTP/2.",
    )
    _add_boolean_option(
        parser,
        "enable-http2",
        help_text="Enable HTTP/2 when supported by the selected ASGI server.",
        disable_help_text="Disable HTTP/2 protocol advertisement.",
    )
    parser.add_argument(
        "--log-level",
        choices=_config_option("log_level").choices,
        default=argparse.SUPPRESS,
        help="ASGI server and HyperVector log level.",
    )
    parser.add_argument(
        "--certfile",
        default=argparse.SUPPRESS,
        help="TLS certificate file for HTTP/2 over TLS.",
    )
    parser.add_argument(
        "--keyfile",
        default=argparse.SUPPRESS,
        help="TLS private key file for HTTP/2 over TLS.",
    )
    parser.add_argument(
        "--default-index-type",
        choices=_config_option("default_index_type").choices,
        default=argparse.SUPPRESS,
        help="Reserved default index type for collection creation.",
    )
    parser.add_argument(
        "--default-metric-type",
        choices=_config_option("default_metric_type").choices,
        default=argparse.SUPPRESS,
        help="Reserved default metric type for collection creation.",
    )
    _add_boolean_option(
        parser,
        "enable-logging",
        help_text="Enable or disable HyperVector Python logging.",
        disable_help_text="Disable HyperVector Python logging.",
    )
    _add_boolean_option(
        parser,
        "log-to-stderr",
        help_text="Enable or disable HyperVector stderr logging.",
        disable_help_text="Disable HyperVector stderr logging.",
    )
    _add_boolean_option(
        parser,
        "log-to-file",
        help_text="Enable or disable HyperVector file logging.",
        disable_help_text="Disable HyperVector file logging.",
    )
    parser.add_argument(
        "--log-file-path",
        default=argparse.SUPPRESS,
        help="HyperVector log file path.",
    )
    return parser


def run_server(config: HypervecConfig) -> None:
    """Start the selected ASGI server from an already validated config."""

    data_root = config.server.data_root
    if data_root is None:  # pragma: no cover - resolve_config enforces this
        raise RuntimeError("server.data_root must be resolved before starting the server")

    app = create_app(data_root=data_root)

    if config.server.server == "uvicorn":
        # Uvicorn remains the HTTP/1.1 compatibility path. enable_http2 is
        # consumed only by the Hypercorn branch below.
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("HyperVec HTTP server requires uvicorn.") from exc

        if config.server.certfile:
            uvicorn.run(
                app,
                host=config.server.host,
                port=config.server.port,
                log_level=config.logging.log_level,
                access_log=config.logging.enable_logging,
                ssl_certfile=config.server.certfile,
                ssl_keyfile=config.server.keyfile,
            )
        else:
            uvicorn.run(
                app,
                host=config.server.host,
                port=config.server.port,
                log_level=config.logging.log_level,
                access_log=config.logging.enable_logging,
            )
        return

    try:
        import asyncio
        from hypercorn.asyncio import serve
        from hypercorn.config import Config
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The HyperVec Hypercorn server requires hypercorn. Install hypervec[server] "
            "or run: python -m pip install hypercorn h2"
        ) from exc

    hypercorn_config = Config()
    hypercorn_config.bind = [f"{config.server.host}:{config.server.port}"]
    hypercorn_config.loglevel = config.logging.log_level
    hypercorn_config.certfile = config.server.certfile
    hypercorn_config.keyfile = config.server.keyfile
    # Preserve the historical HTTP/2 default while allowing HTTP/1.1-only TLS
    # negotiation. Hypercorn may still support cleartext h2c internally.
    hypercorn_config.alpn_protocols = (
        ["h2", "http/1.1"] if config.server.enable_http2 else ["http/1.1"]
    )
    if not config.logging.enable_logging:
        hypercorn_config.accesslog = None
    asyncio.run(serve(app, hypercorn_config))


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
