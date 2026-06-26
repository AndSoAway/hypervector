# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the License) found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import tempfile
from typing import Any

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
    engine: HypervecServerEngine | None = None,
) -> Any:
    FastAPI, HTTPException, Query, Request, FileResponse, BaseModel, Field = _require_fastapi()
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
        consistency_level: Optional[str] = None

    class SyncCheckRequest(BaseModel):
        client_version: int
        client_checksum: str | None = None

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

    examples = {
        "HNSW": {
            "name": "HNSW",
            "full_name": "Hierarchical Navigable Small World",
            "description": "基于多层小世界图的近似最近邻索引，适合高召回、低延迟向量检索。",
            "use_case": ["百万级以上向量检索", "低延迟在线搜索", "高召回召回阶段"],
            "advantages": ["查询速度快", "召回率高", "无需训练"],
            "limitations": ["索引内存占用较高", "构建耗时随 M 和 ef_construction 增加"],
            "parameters": {"M": "图连接数", "ef_construction": "构建搜索宽度", "ef_search": "查询搜索宽度"},
            "example_code": {"Python": {"create": "index_params.add_index(field_name='vector', index_type='HNSW', metric_type='L2', params={'M': 32, 'ef_construction': 200})", "search": "client.search(collection_name='wiki_hnsw_1m', data=[query], limit=10, search_params={'ef_search': 128})"}},
            "performance_tips": ["提高 ef_search 可提升召回但增加延迟", "提高 M 可提升图质量但增加内存"],
        },
        "Flat": {
            "name": "Flat",
            "full_name": "Brute-force Flat Index",
            "description": "精确暴力搜索索引，遍历全部向量计算距离。",
            "use_case": ["小规模数据", "召回率基准", "验证近似索引效果"],
            "advantages": ["结果精确", "无需训练", "实现简单"],
            "limitations": ["大规模数据查询较慢"],
            "parameters": {"metric_type": "L2/IP/COSINE"},
        },
        "IVF": {
            "name": "IVF",
            "full_name": "Inverted File Index",
            "description": "倒排聚类索引，通过只搜索部分聚类降低查询开销。",
            "use_case": ["大规模向量粗召回", "可接受近似结果的搜索"],
            "advantages": ["查询成本可控", "适合大规模数据"],
            "limitations": ["需要训练", "召回受 nprobe 影响"],
            "parameters": {"nlist": "聚类中心数", "nprobe": "查询探测聚类数"},
        },
        "IVFPQ": {
            "name": "IVFPQ",
            "full_name": "IVF with Product Quantization",
            "description": "倒排索引结合乘积量化，降低内存占用。",
            "use_case": ["超大规模向量检索", "内存敏感场景"],
            "advantages": ["内存占用低", "查询速度快"],
            "limitations": ["量化会损失精度", "需要训练"],
            "parameters": {"nlist": "聚类中心数", "m": "子量化器数量", "nbits": "编码位数"},
        },
        "IVFLVQ": {
            "name": "IVFLVQ",
            "full_name": "IVF with LVQ",
            "description": "倒排索引结合 LVQ 量化，兼顾压缩和查询效率。",
            "use_case": ["大规模压缩检索", "内存受限场景"],
            "advantages": ["压缩率高", "适合批量检索"],
            "limitations": ["参数调优复杂", "存在量化误差"],
            "parameters": {"nlist": "聚类中心数", "nlocal": "局部量化参数", "nbits": "量化位数"},
        },
        "PQ": {
            "name": "PQ",
            "full_name": "Product Quantization",
            "description": "乘积量化索引，将向量压缩为短编码以降低存储成本。",
            "use_case": ["内存压缩", "大规模候选集重排前筛选"],
            "advantages": ["存储成本低", "适合大规模数据"],
            "limitations": ["召回低于精确索引", "需要训练码本"],
            "parameters": {"m": "子空间数量", "nbits": "每个子空间编码位数"},
        },
        "LVQ": {
            "name": "LVQ",
            "full_name": "Locally-adaptive Vector Quantization",
            "description": "局部自适应向量量化索引，用于压缩向量并加速检索。",
            "use_case": ["压缩检索", "内存带宽敏感场景"],
            "advantages": ["压缩友好", "可降低内存访问"],
            "limitations": ["近似结果", "需要根据数据调参"],
            "parameters": {"nlocal": "局部分组数", "nbits": "量化位数"},
        },
        "HNSWFlat": {
            "name": "HNSWFlat",
            "full_name": "HNSW with Flat vectors",
            "description": "HNSW 图结构配合原始向量精确距离计算，是通用在线 ANN 索引。",
            "use_case": ["通用向量搜索服务", "高召回在线检索"],
            "advantages": ["召回和延迟表现均衡", "适合作为默认 HNSW 配置"],
            "limitations": ["内存占用高于量化索引"],
            "parameters": {"M": "图连接数", "ef_construction": "构建搜索宽度", "ef_search": "查询搜索宽度"},
        },
    }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/examples")
    def list_examples() -> dict[str, Any]:
        return {"supported_indexes": list(examples.keys()), "examples": list(examples.values())}

    @app.get("/examples/{index_name}")
    def get_example(index_name: str) -> dict[str, Any]:
        for name, example in examples.items():
            if name.lower() == index_name.lower():
                return example
        raise HTTPException(status_code=404, detail=f"example index '{index_name}' does not exist.")

    @app.get("/collections")
    def list_collections() -> dict[str, list[str]]:
        return {"collections": engine.list_collections()}

    @app.get("/collections/describe")
    def describe_collections() -> dict[str, Any]:
        try:
            return {"collections": engine.describe_collections()}
        except Exception as exc:
            raise fail(exc)

    @app.get("/examples")
    def examples() -> dict[str, Any]:
        try:
            return {"examples": engine.supported_index_examples()}
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
        request: CreateCollectionRequest = Body(...),
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
