"""
kvcos/engram/manifest.py — Knowledge index manifest registry.

Tracks which source files have been indexed into .eng files,
their content hashes for incremental re-indexing, and chunk
metadata for multi-chunk files.

Storage: JSON file at ~/.engram/manifest.json (human-readable,
git-friendly, easily inspectable).

Thread safety: reads are lock-free, writes use atomic rename.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class ChunkRecord:
    """One indexed chunk from a source file."""
    eng_path: str          # Absolute path to .eng file
    chunk_index: int       # 0-based chunk index within source
    chunk_total: int       # Total chunks for this source
    char_start: int        # Start offset in source content
    char_end: int          # End offset in source content
    indexed_at: float      # Unix timestamp of indexing


@dataclass(frozen=True)
class SourceRecord:
    """Registry entry for one indexed source file."""
    source_path: str       # Absolute path to original .md file
    content_hash: str      # SHA-256 of file content at index time
    project: str           # Project namespace (e.g., "engram", "_global")
    file_size: int         # Bytes at index time
    chunks: tuple[ChunkRecord, ...] = ()
    indexed_at: float = 0.0
    last_verified: float = 0.0

    @property
    def eng_paths(self) -> list[str]:
        """All .eng file paths for this source."""
        return [c.eng_path for c in self.chunks]


def _content_hash(content: str) -> str:
    """SHA-256 hex digest of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    """SHA-256 hex digest of file on disk."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Manifest:
    """
    Knowledge index manifest — tracks source-to-.eng mappings.

    Immutable-style operations: all mutations return new state
    and write atomically to disk.

    Usage:
        m = Manifest.load()
        m = m.register(source_path, content_hash, project, chunks)
        # m is now updated and persisted to disk
    """

    def __init__(
        self,
        records: dict[str, SourceRecord],
        manifest_path: Path,
    ) -> None:
        self._records = dict(records)  # defensive copy
        self._path = manifest_path

    @classmethod
    def load(cls, manifest_path: Path | None = None) -> Manifest:
        """Load manifest from disk, or create empty if not found."""
        if manifest_path is None:
            manifest_path = Path(
                os.environ.get("ENGRAM_MANIFEST_PATH",
                               "~/.engram/manifest.json")
            ).expanduser()

        if manifest_path.exists():
            data = json.loads(manifest_path.read_text())
            records = {}
            for key, rec_data in data.get("sources", {}).items():
                chunks = tuple(
                    ChunkRecord(**c) for c in rec_data.pop("chunks", [])
                )
                records[key] = SourceRecord(**rec_data, chunks=chunks)
            return cls(records, manifest_path)

        return cls({}, manifest_path)

    def register(
        self,
        source_path: str,
        content_hash: str,
        project: str,
        file_size: int,
        chunks: list[ChunkRecord],
    ) -> Manifest:
        """
        Register a newly indexed source file. Returns updated Manifest.

        Overwrites any existing record for the same source_path
        (re-index scenario).
        """
        now = time.time()
        record = SourceRecord(
            source_path=source_path,
            content_hash=content_hash,
            project=project,
            file_size=file_size,
            chunks=tuple(chunks),
            indexed_at=now,
            last_verified=now,
        )

        new_records = dict(self._records)
        new_records[source_path] = record

        new_manifest = Manifest(new_records, self._path)
        new_manifest._persist()
        return new_manifest

    def unregister(self, source_path: str) -> Manifest:
        """Remove a source from the manifest. Returns updated Manifest."""
        new_records = {
            k: v for k, v in self._records.items()
            if k != source_path
        }
        new_manifest = Manifest(new_records, self._path)
        new_manifest._persist()
        return new_manifest

    def needs_reindex(self, source_path: str, current_hash: str) -> bool:
        """Check if a source file needs re-indexing (content changed)."""
        record = self._records.get(source_path)
        if record is None:
            return True
        return record.content_hash != current_hash

    def get_record(self, source_path: str) -> SourceRecord | None:
        """Look up a source record by path."""
        return self._records.get(source_path)

    def get_project_records(self, project: str) -> list[SourceRecord]:
        """All records for a given project namespace."""
        return [
            r for r in self._records.values()
            if r.project == project
        ]

    def all_records(self) -> Iterator[SourceRecord]:
        """Iterate over all registered source records."""
        yield from self._records.values()

    @property
    def total_sources(self) -> int:
        return len(self._records)

    @property
    def total_chunks(self) -> int:
        return sum(len(r.chunks) for r in self._records.values())

    @property
    def projects(self) -> set[str]:
        return {r.project for r in self._records.values()}

    def summary(self) -> dict:
        """Quick stats for display."""
        return {
            "total_sources": self.total_sources,
            "total_chunks": self.total_chunks,
            "projects": sorted(self.projects),
            "manifest_path": str(self._path),
        }

    def _persist(self) -> None:
        """Atomic write to disk via tempfile + rename."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {
            "version": 1,
            "updated_at": time.time(),
            "sources": {},
        }
        for key, rec in self._records.items():
            rec_dict = asdict(rec)
            serializable["sources"][key] = rec_dict

        # Atomic write: write to temp, then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent),
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(serializable, f, indent=2)
            os.replace(tmp_path, str(self._path))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def __len__(self) -> int:
        return self.total_sources

    def __contains__(self, source_path: str) -> bool:
        return source_path in self._records

    def __repr__(self) -> str:
        return (
            f"Manifest({self.total_sources} sources, "
            f"{self.total_chunks} chunks, "
            f"projects={sorted(self.projects)})"
        )
