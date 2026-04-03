"""
ENGRAM Protocol — ENGRAM Python Client


Async HTTP client wrapping all ENGRAM API endpoints.
This is what agents import to interact with the engram store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx


class EngramClient:
    """Python client for the ENGRAM REST API.

    Usage:
        client = EngramClient("http://localhost:8080")
        result = client.store_file(path, agent_id="worker", task="analyze code", model_id="llama-3.1-8b")
        matches = client.search(task_description="debug auth error", top_k=3)
        data = client.get(matches[0]["cache_id"])
    """

    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=f"{self.base_url}/v1", timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Health ────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        """Check ENGRAM server health."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    # ── Store ─────────────────────────────────────────────────

    def store_file(
        self,
        file_path: Path,
        agent_id: str,
        task_description: str,
        model_id: str,
        compression: str = "q8_0",
    ) -> dict[str, Any]:
        """Upload a .eng file to the engram store.

        Args:
            file_path: Path to the .eng file
            agent_id: Agent identifier
            task_description: Human-readable description
            model_id: Model identifier
            compression: Compression method used

        Returns:
            Dict with cache_id, size_bytes, compression_ratio, path
        """
        with open(file_path, "rb") as f:
            resp = self._client.post(
                "/cache",
                params={
                    "agent_id": agent_id,
                    "task_description": task_description,
                    "model_id": model_id,
                    "compression": compression,
                },
                files={"file": (file_path.name, f, "application/octet-stream")},
            )
        resp.raise_for_status()
        return resp.json()

    def store_bytes(
        self,
        data: bytes,
        agent_id: str,
        task_description: str,
        model_id: str,
        compression: str = "q8_0",
        filename: str = "cache.eng",
    ) -> dict[str, Any]:
        """Upload raw .eng bytes to the engram store."""
        resp = self._client.post(
            "/cache",
            params={
                "agent_id": agent_id,
                "task_description": task_description,
                "model_id": model_id,
                "compression": compression,
            },
            files={"file": (filename, data, "application/octet-stream")},
        )
        resp.raise_for_status()
        return resp.json()

    # ── Retrieve ──────────────────────────────────────────────

    def get(self, cache_id: str) -> bytes:
        """Retrieve a .eng file by cache ID.

        Returns raw bytes of the .eng file.
        """
        resp = self._client.get(f"/cache/{cache_id}")
        resp.raise_for_status()
        return resp.content

    # ── Search ────────────────────────────────────────────────

    def search(
        self,
        task_description: str,
        model_id: str | None = None,
        top_k: int = 5,
        min_similarity: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar engram states.

        Returns list of search result dicts with cache_id, similarity, etc.
        """
        body: dict[str, Any] = {
            "task_description": task_description,
            "top_k": top_k,
        }
        if model_id:
            body["model_id"] = model_id
        if min_similarity is not None:
            body["min_similarity"] = min_similarity

        resp = self._client.post("/cache/search", json=body)
        resp.raise_for_status()
        return resp.json()["results"]

    # ── Delete ────────────────────────────────────────────────

    def delete(self, cache_id: str) -> bool:
        """Delete an engram from storage and index."""
        resp = self._client.delete(f"/cache/{cache_id}")
        resp.raise_for_status()
        return resp.json()["deleted"]

    # ── Stats ─────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Get aggregate engram store statistics."""
        resp = self._client.get("/cache/stats")
        resp.raise_for_status()
        return resp.json()
