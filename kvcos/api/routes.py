"""
ENGRAM Protocol — API Routes


FastAPI route handlers for the ENGRAM REST API.
All endpoints under /v1/ prefix.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile, File

from kvcos.api.schemas import (
    DeleteResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    StatsResponse,
    StoreResponse,
)
from kvcos.core.types import ENGRAM_VERSION

router = APIRouter(prefix="/v1")


# ── Dependency stubs ──────────────────────────────────────────────────────────
# These are replaced by real instances in server.py lifespan.
# Using module-level state that the server sets during startup.

_retriever = None
_storage = None
_index = None


def _get_retriever():
    if _retriever is None:
        raise HTTPException(503, "ENGRAM not initialized. Server starting up.")
    return _retriever


def _get_storage():
    if _storage is None:
        raise HTTPException(503, "ENGRAM not initialized. Server starting up.")
    return _storage


def _get_index():
    if _index is None:
        raise HTTPException(503, "ENGRAM not initialized. Server starting up.")
    return _index


# ── Health ────────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    index = _get_index()
    storage = _get_storage()
    return HealthResponse(
        status="ok",
        version=ENGRAM_VERSION,
        index_entries=index.n_entries,
        storage_backend="local",
    )


# ── Stats (must come before /cache/{cache_id} to avoid route shadowing) ──────


@router.get("/cache/stats", response_model=StatsResponse)
async def cache_stats():
    """Get aggregate statistics for the engram store."""
    storage = _get_storage()
    stats = storage.stats()
    return StatsResponse(
        total_entries=stats["total_entries"],
        total_size_bytes=stats["total_size_bytes"],
        total_size_mb=round(stats["total_size_bytes"] / (1024 * 1024), 2),
        avg_compression_ratio=stats["avg_compression_ratio"],
        model_breakdown=stats["model_breakdown"],
    )


# ── Store ─────────────────────────────────────────────────────────────────────


@router.post("/cache", response_model=StoreResponse)
async def store_cache(
    agent_id: str,
    task_description: str,
    model_id: str,
    file: UploadFile = File(...),
    compression: str = "q8_0",
):
    """Store a .eng file in the engram store.

    Accepts a pre-serialized .eng file upload.
    The file is stored and its metadata indexed for EGR retrieval.
    """
    storage = _get_storage()

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "Empty file upload")

    import uuid
    cache_id = str(uuid.uuid4())

    from kvcos.core.types import EngramMetadata
    from datetime import datetime, timezone

    metadata: EngramMetadata = {
        "engram_version": ENGRAM_VERSION,
        "cache_id": cache_id,
        "compression": compression,
        "model_id": model_id,
        "model_family": "",
        "n_layers": "0",
        "n_heads": "0",
        "n_kv_heads": "0",
        "head_dim": "0",
        "context_len": "0",
        "agent_id": agent_id,
        "task_description": task_description,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    path = storage.store(cache_id, data, metadata)

    return StoreResponse(
        cache_id=cache_id,
        size_bytes=len(data),
        compression_ratio=1.0,
        path=path,
    )


# ── Retrieve by ID ────────────────────────────────────────────────────────────


@router.get("/cache/{cache_id}")
async def get_cache(cache_id: str):
    """Retrieve a .eng file by cache ID.

    Returns the raw .eng file bytes (application/octet-stream).
    """
    storage = _get_storage()

    data = storage.get(cache_id)
    if data is None:
        raise HTTPException(404, f"Cache entry not found: {cache_id}")

    from fastapi.responses import Response
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{cache_id}.eng"'},
    )


# ── Search ────────────────────────────────────────────────────────────────────


@router.post("/cache/search", response_model=SearchResponse)
async def search_cache(req: SearchRequest):
    """Search for similar engram states via EGR manifold search.

    Uses inner product similarity (MIPS) in the model's pre-RoPE
    key manifold. D2: K→K retrieval only.
    """
    index = _get_index()

    # For text-only search without a KV query vector, we need the
    # retriever to extract a state vector first. This endpoint
    # currently returns index entries matching by metadata filter.
    # Full EGR vector search requires a query KV cache (via /egr/retrieve).

    # Metadata-based listing with optional filters
    storage = _get_storage()
    entries = storage.list_entries(model_family=None, limit=req.top_k)

    results = [
        SearchResultItem(
            cache_id=e.get("cache_id", ""),
            similarity=0.0,
            task_description=e.get("task_description", ""),
            model_id=e.get("model_id", ""),
            created_at=e.get("created_at", ""),
            context_len=int(e.get("context_len", "0")),
        )
        for e in entries
        if (req.model_id is None or e.get("model_id") == req.model_id)
    ]

    return SearchResponse(results=results[:req.top_k], n_searched=index.n_entries)


# ── Delete ────────────────────────────────────────────────────────────────────


@router.delete("/cache/{cache_id}", response_model=DeleteResponse)
async def delete_cache(cache_id: str):
    """Delete an engram from storage and index."""
    retriever = _get_retriever()
    deleted = retriever.delete_engram(cache_id)
    return DeleteResponse(deleted=deleted, cache_id=cache_id)


