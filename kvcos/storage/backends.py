"""
ENGRAM Protocol — Abstract Storage Backend


All storage backends (local, redis, S3) implement this interface.
Phase 1 uses local disk only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from kvcos.core.types import CacheStats, EngramMetadata


class StorageBackend(ABC):
    """Abstract interface for engram storage backends.

    All operations are synchronous in Phase 1.
    """

    @abstractmethod
    def store(self, cache_id: str, data: bytes, metadata: EngramMetadata) -> str:
        """Store a .eng file. Returns storage path/key."""
        ...

    @abstractmethod
    def store_file(self, cache_id: str, source_path: Path, metadata: EngramMetadata) -> str:
        """Store a .eng file from a local path (zero-copy when possible)."""
        ...

    @abstractmethod
    def get(self, cache_id: str) -> bytes | None:
        """Retrieve a .eng file as bytes. None if not found."""
        ...

    @abstractmethod
    def get_path(self, cache_id: str) -> Path | None:
        """Get local filesystem path for a cache entry. None if not found."""
        ...

    @abstractmethod
    def get_metadata(self, cache_id: str) -> EngramMetadata | None:
        """Read only metadata (header-only, no tensor data loaded)."""
        ...

    @abstractmethod
    def delete(self, cache_id: str) -> bool:
        """Delete a cache entry. Returns True if deleted."""
        ...

    @abstractmethod
    def list_entries(
        self,
        agent_id: str | None = None,
        model_family: str | None = None,
        limit: int = 100,
    ) -> list[EngramMetadata]:
        """List cache entries with optional filters."""
        ...

    @abstractmethod
    def exists(self, cache_id: str) -> bool:
        """Check if a cache entry exists."""
        ...

    @abstractmethod
    def stats(self) -> CacheStats:
        """Get aggregate statistics for the store."""
        ...
