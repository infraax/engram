"""
ENGRAM Protocol — API Schemas


Pydantic models for all REST API request/response payloads.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Store ─────────────────────────────────────────────────────────────────────


class StoreRequest(BaseModel):
    agent_id: str
    task_description: str
    model_id: str
    compression: str = "q8_0"


class StoreResponse(BaseModel):
    cache_id: str
    size_bytes: int
    compression_ratio: float
    path: str


# ── Retrieve ──────────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    task_description: str
    model_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=100)
    min_similarity: float | None = None


class SearchResultItem(BaseModel):
    cache_id: str
    similarity: float
    task_description: str
    model_id: str
    created_at: str
    context_len: int


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    n_searched: int


# ── Extend ────────────────────────────────────────────────────────────────────


class ExtendResponse(BaseModel):
    cache_id: str
    new_context_len: int


# ── Delete ────────────────────────────────────────────────────────────────────


class DeleteResponse(BaseModel):
    deleted: bool
    cache_id: str


# ── Stats ─────────────────────────────────────────────────────────────────────


class StatsResponse(BaseModel):
    total_entries: int
    total_size_bytes: int
    total_size_mb: float
    avg_compression_ratio: float
    model_breakdown: dict[str, int]


# ── Health ────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    index_entries: int
    storage_backend: str
