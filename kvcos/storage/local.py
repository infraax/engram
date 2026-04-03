"""
ENGRAM Protocol — Local Disk Storage Backend


Directory layout:
    {data_dir}/{model_family}/{agent_id}/{date}/{cache_id}.eng

Phase 1 production backend. Zero infrastructure dependencies.
Uses safetensors header-only read for metadata operations.
D7: One safetensors file per 256-token block.
"""

from __future__ import annotations

import logging
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

from kvcos.core.serializer import EngramSerializer
from kvcos.core.types import ENG_FILE_EXTENSION, CacheStats, EngramMetadata
from kvcos.storage.backends import StorageBackend


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage for .eng files.

    Files organized by model family, agent ID, and date.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._serializer = EngramSerializer()
        self._index: dict[str, Path] = {}  # cache_id → file path
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Scan data directory and rebuild in-memory path index."""
        self._index.clear()
        for eng_file in self.data_dir.rglob(f"*{ENG_FILE_EXTENSION}"):
            cache_id = eng_file.stem
            try:
                meta = self._serializer.read_metadata_only(eng_file)
                if "cache_id" in meta:
                    cache_id = meta["cache_id"]
            except Exception as e:
                logger.debug("Skipping metadata for %s: %s", eng_file.name, e)
            self._index[cache_id] = eng_file

    def _resolve_path(self, metadata: EngramMetadata) -> Path:
        """Determine storage path from metadata."""
        model_family = metadata.get("model_family", "unknown")
        agent_id = metadata.get("agent_id", "default")
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cache_id = metadata.get("cache_id", "unknown")
        path = self.data_dir / model_family / agent_id / date_str / f"{cache_id}{ENG_FILE_EXTENSION}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def store(self, cache_id: str, data: bytes, metadata: EngramMetadata) -> str:
        metadata_copy = dict(metadata)
        metadata_copy["cache_id"] = cache_id
        path = self._resolve_path(metadata_copy)  # type: ignore[arg-type]

        tmp_path = path.with_suffix(f"{ENG_FILE_EXTENSION}.tmp")
        try:
            tmp_path.write_bytes(data)
            tmp_path.rename(path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        self._index[cache_id] = path
        return str(path)

    def store_file(self, cache_id: str, source_path: Path, metadata: EngramMetadata) -> str:
        metadata_copy = dict(metadata)
        metadata_copy["cache_id"] = cache_id
        dest_path = self._resolve_path(metadata_copy)  # type: ignore[arg-type]

        if source_path == dest_path:
            self._index[cache_id] = dest_path
            return str(dest_path)

        tmp_path = dest_path.with_suffix(f"{ENG_FILE_EXTENSION}.tmp")
        try:
            shutil.copy2(str(source_path), str(tmp_path))
            tmp_path.rename(dest_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        self._index[cache_id] = dest_path
        return str(dest_path)

    def get(self, cache_id: str) -> bytes | None:
        path = self._index.get(cache_id)
        if path is None or not path.exists():
            return None
        return path.read_bytes()

    def get_path(self, cache_id: str) -> Path | None:
        path = self._index.get(cache_id)
        if path is None or not path.exists():
            return None
        return path

    def get_metadata(self, cache_id: str) -> EngramMetadata | None:
        path = self._index.get(cache_id)
        if path is None or not path.exists():
            return None
        try:
            return self._serializer.read_metadata_only(path)
        except Exception as e:
            logger.warning("Failed to read metadata for %s: %s", cache_id, e)
            return None

    def delete(self, cache_id: str) -> bool:
        path = self._index.pop(cache_id, None)
        if path is None or not path.exists():
            return False

        path.unlink()

        parent = path.parent
        try:
            while parent != self.data_dir:
                if not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
                else:
                    break
        except OSError:
            pass

        return True

    def list_entries(
        self,
        agent_id: str | None = None,
        model_family: str | None = None,
        limit: int = 100,
    ) -> list[EngramMetadata]:
        results: list[EngramMetadata] = []

        for cache_id, path in self._index.items():
            if len(results) >= limit:
                break
            if not path.exists():
                continue
            try:
                meta = self._serializer.read_metadata_only(path)
            except Exception as e:
                logger.debug("Skipping %s in list_entries: %s", cache_id, e)
                continue
            if agent_id and meta.get("agent_id") != agent_id:
                continue
            if model_family and meta.get("model_family") != model_family:
                continue
            results.append(meta)

        results.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return results[:limit]

    def exists(self, cache_id: str) -> bool:
        path = self._index.get(cache_id)
        return path is not None and path.exists()

    def stats(self) -> CacheStats:
        total_entries = 0
        total_size = 0
        model_counts: dict[str, int] = defaultdict(int)

        for cache_id, path in self._index.items():
            if not path.exists():
                continue
            total_entries += 1
            total_size += path.stat().st_size
            try:
                meta = self._serializer.read_metadata_only(path)
                model_counts[meta.get("model_family", "unknown")] += 1
            except Exception as e:
                logger.debug("Metadata read failed for %s: %s", cache_id, e)
                model_counts["unknown"] += 1

        return CacheStats(
            total_entries=total_entries,
            total_size_bytes=total_size,
            avg_compression_ratio=0.0,
            model_breakdown=dict(model_counts),
        )

    def vacuum(self) -> int:
        """Remove stale index entries for deleted files."""
        stale = [cid for cid, path in self._index.items() if not path.exists()]
        for cid in stale:
            del self._index[cid]
        return len(stale)
