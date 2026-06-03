"""
HyperVec FastAPI server.

Core polling endpoint:
  GET /collections/{name}/version  — frontend polls this to detect index updates
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel

from index_manager import IndexManager
from scalar_store import ScalarStore
from store import MetaStore

_store = MetaStore()
_scalar = ScalarStore()
_manager = IndexManager(_store, _scalar)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="HyperVec Index Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CreateCollectionRequest(BaseModel):
    name: str
    dim: int


class CollectionResponse(BaseModel):
    collection_id: str
    name: str
    version: int
    updated_at: float


class VersionResponse(BaseModel):
    collection_id: str
    version: int
    updated_at: float


class SyncCheckRequest(BaseModel):
    client_version: int


class SyncCheckResponse(BaseModel):
    needs_sync: bool
    server_version: int
    client_version: int


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

@app.get("/collections", response_model=list[CollectionResponse])
def list_collections():
    return [
        CollectionResponse(
            collection_id=m.collection_id,
            name=m.name,
            version=m.version,
            updated_at=m.updated_at,
        )
        for m in _store.list_all()
    ]


@app.post("/collections", response_model=CollectionResponse, status_code=201)
def create_collection(body: CreateCollectionRequest):
    meta = _manager.create_collection(body.name, body.dim)
    return CollectionResponse(
        collection_id=meta.collection_id,
        name=meta.name,
        version=meta.version,
        updated_at=meta.updated_at,
    )


@app.delete("/collections/{collection_id}", status_code=204)
def delete_collection(collection_id: str):
    if not _manager.delete_collection(collection_id):
        raise HTTPException(404, "Collection not found")


# ---------------------------------------------------------------------------
# Version polling  ← frontend polls this
# ---------------------------------------------------------------------------

@app.get("/collections/{collection_id}/version", response_model=VersionResponse)
def get_version(collection_id: str):
    """
    Lightweight polling endpoint.
    Frontend compares returned version with its locally cached version.
    If server_version > client_version → pull /collections/{id}/index.
    """
    meta = _store.get(collection_id)
    if meta is None:
        raise HTTPException(404, "Collection not found")
    return VersionResponse(
        collection_id=collection_id,
        version=meta.version,
        updated_at=meta.updated_at,
    )


# ---------------------------------------------------------------------------
# Reconnect sync check
# ---------------------------------------------------------------------------

@app.post("/collections/{collection_id}/sync-check", response_model=SyncCheckResponse)
def sync_check(collection_id: str, body: SyncCheckRequest):
    """
    Called on reconnect. Client sends its cached version.
    needs_sync=true → client should fetch /collections/{id}/index.
    """
    meta = _store.get(collection_id)
    if meta is None:
        raise HTTPException(404, "Collection not found")
    return SyncCheckResponse(
        needs_sync=meta.version != body.client_version,
        server_version=meta.version,
        client_version=body.client_version,
    )


# ---------------------------------------------------------------------------
# Index file download
# ---------------------------------------------------------------------------

@app.get("/collections/{collection_id}/index")
def download_index(collection_id: str):
    """Download serialized HyperVec index file for client-side caching."""
    meta = _store.get(collection_id)
    if meta is None:
        raise HTTPException(404, "Collection not found")
    path = Path(meta.index_path)
    if not path.exists():
        raise HTTPException(503, "Index file not available")
    return FileResponse(
        path=str(path),
        media_type="application/octet-stream",
        filename=f"{collection_id}.index",
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "collections": len(_store.list_all())}
