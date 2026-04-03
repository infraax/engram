"""
Engrammatic Geometry Retrieval — Manifold Index


FAISS-backed MIPS (Maximum Inner Product Search) index for EGR retrieval.
Indexes state vectors extracted from .eng files by MARStateExtractor.

D2: FAISS IndexFlatIP for K→K retrieval only. Never Q→K.
    faiss.serialize_index() for persistence (not write_index — avoids
    platform incompatibility Issue #3888). Atomic write via temp + rename.
    MKL build enforced at import time.

D4: No L2 normalization. True MIPS. Raw inner product scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
import torch

from kvcos.core.types import CacheSearchResult


@dataclass
class IndexEntry:
    """Metadata associated with an indexed state vector."""

    cache_id: str
    task_description: str
    model_id: str
    created_at: str
    context_len: int
    l2_norm: float  # D4: stored for optional downstream use


class ManifoldIndex:
    """FAISS-backed inner product index for EGR state vectors.

    Stores state vectors and associated metadata for MIPS retrieval.
    Persistence via faiss.serialize_index() with atomic file writes.

    Usage:
        index = ManifoldIndex(dim=160)
        index.add(state_vec, entry)
        results = index.search(query_vec, top_k=5)
        index.save(Path("~/.engram/index/egr.faiss"))
    """

    def __init__(self, dim: int, index_path: Path | None = None):
        """Initialize the manifold index.

        Args:
            dim: Dimension of state vectors (must match MARStateExtractor output).
            index_path: Optional path to load an existing index from disk.
        """
        self.dim = dim
        self._entries: list[IndexEntry] = []
        self._id_to_position: dict[str, int] = {}  # cache_id → FAISS row position

        if index_path and index_path.exists():
            self._index = self._load_index(index_path)
        else:
            # D2: IndexFlatIP — exact MIPS, correct for Phase 1 corpus sizes (<100K)
            self._index = faiss.IndexFlatIP(dim)

    @property
    def n_entries(self) -> int:
        """Number of indexed state vectors."""
        return self._index.ntotal

    def add(
        self,
        state_vec: torch.Tensor | np.ndarray,
        entry: IndexEntry,
    ) -> None:
        """Add a state vector and its metadata to the index.

        Args:
            state_vec: [dim] state vector (D4: NOT normalized)
            entry: Associated metadata for this engram
        """
        vec = self._to_numpy(state_vec)

        if vec.shape != (self.dim,):
            raise ValueError(
                f"State vector dim {vec.shape} != index dim ({self.dim},)"
            )

        # Check for duplicate cache_id
        if entry.cache_id in self._id_to_position:
            # Update: remove old entry position tracking, add at new position
            # FAISS IndexFlat doesn't support in-place update, so we just
            # track the latest position. Old vector remains but is shadowed.
            pass

        position = self._index.ntotal
        self._index.add(vec.reshape(1, -1).astype(np.float32))
        self._entries.append(entry)
        self._id_to_position[entry.cache_id] = position

    def search(
        self,
        query_vec: torch.Tensor | np.ndarray,
        top_k: int = 5,
        min_similarity: float | None = None,
        model_id: str | None = None,
    ) -> list[CacheSearchResult]:
        """Search for the most similar engram states via MIPS.

        Args:
            query_vec: [dim] query state vector
            top_k: Number of results to return
            min_similarity: Minimum inner product score threshold
            model_id: Optional filter by model ID

        Returns:
            List of CacheSearchResult sorted by similarity (descending)
        """
        if self._index.ntotal == 0:
            return []

        vec = self._to_numpy(query_vec)
        if vec.shape != (self.dim,):
            raise ValueError(
                f"Query vector dim {vec.shape} != index dim ({self.dim},)"
            )

        # Search more than top_k to account for filtering
        search_k = min(top_k * 3, self._index.ntotal) if model_id else min(top_k, self._index.ntotal)
        scores, indices = self._index.search(
            vec.reshape(1, -1).astype(np.float32), search_k
        )

        results: list[CacheSearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._entries):
                continue

            entry = self._entries[idx]

            # Skip if this cache_id has been superseded by a later add
            if self._id_to_position.get(entry.cache_id) != idx:
                continue

            # Apply filters
            if model_id and entry.model_id != model_id:
                continue
            if min_similarity is not None and score < min_similarity:
                continue

            results.append(CacheSearchResult(
                cache_id=entry.cache_id,
                similarity=float(score),
                task_description=entry.task_description,
                model_id=entry.model_id,
                created_at=entry.created_at,
                context_len=entry.context_len,
            ))

            if len(results) >= top_k:
                break

        return results

    def remove(self, cache_id: str) -> bool:
        """Mark a cache entry as removed from the index.

        FAISS IndexFlat doesn't support deletion. We remove from the
        metadata tracking so the entry is filtered out of search results.
        The vector remains in FAISS until the next rebuild.

        Args:
            cache_id: ID to remove

        Returns:
            True if the entry was found and removed from tracking
        """
        if cache_id in self._id_to_position:
            del self._id_to_position[cache_id]
            return True
        return False

    def rebuild(self) -> int:
        """Rebuild the index from only active entries.

        Removes gaps left by remove() calls. Returns count of active entries.
        """
        active_positions = set(self._id_to_position.values())
        if len(active_positions) == len(self._entries):
            return len(active_positions)  # No gaps

        # Collect active vectors and entries
        new_entries: list[IndexEntry] = []
        vectors: list[np.ndarray] = []

        for pos, entry in enumerate(self._entries):
            if pos in active_positions and entry.cache_id in self._id_to_position:
                if self._id_to_position[entry.cache_id] == pos:
                    vec = faiss.rev_swig_ptr(
                        self._index.get_xb(), self._index.ntotal * self.dim
                    ).reshape(-1, self.dim)[pos]
                    vectors.append(vec.copy())
                    new_entries.append(entry)

        # Rebuild
        self._index = faiss.IndexFlatIP(self.dim)
        self._entries = []
        self._id_to_position = {}

        for vec, entry in zip(vectors, new_entries):
            self.add(torch.from_numpy(vec), entry)

        return self.n_entries

    def save(self, path: Path) -> None:
        """Persist the index to disk.

        D2: Uses faiss.serialize_index() (not write_index) to avoid
        platform incompatibility. Atomic write via temp file + rename.
        Metadata saved as a sidecar .json file.
        """
        import json

        path.parent.mkdir(parents=True, exist_ok=True)

        # D2: serialize_index returns numpy uint8 array — write raw bytes
        index_bytes: np.ndarray = faiss.serialize_index(self._index)

        # Atomic write for FAISS index
        tmp_path = path.with_suffix(".faiss.tmp")
        try:
            tmp_path.write_bytes(index_bytes.tobytes())
            tmp_path.rename(path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        # Save metadata sidecar
        meta_path = path.with_suffix(".meta.json")
        meta_tmp = meta_path.with_suffix(".json.tmp")
        try:
            sidecar = {
                "dim": self.dim,
                "entries": [
                    {
                        "cache_id": e.cache_id,
                        "task_description": e.task_description,
                        "model_id": e.model_id,
                        "created_at": e.created_at,
                        "context_len": e.context_len,
                        "l2_norm": e.l2_norm,
                    }
                    for e in self._entries
                ],
                "id_to_position": self._id_to_position,
            }
            meta_tmp.write_text(json.dumps(sidecar, indent=2))
            meta_tmp.rename(meta_path)
        except Exception:
            meta_tmp.unlink(missing_ok=True)
            raise

    def _load_index(self, path: Path) -> faiss.IndexFlatIP:
        """Load a FAISS index and its metadata sidecar from disk.

        D2: Uses faiss.deserialize_index() from raw bytes (not read_index).
        """
        import json

        raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        index = faiss.deserialize_index(raw)

        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            sidecar = json.loads(meta_path.read_text())
            self._entries = [
                IndexEntry(**e) for e in sidecar.get("entries", [])
            ]
            self._id_to_position = {
                k: int(v) for k, v in sidecar.get("id_to_position", {}).items()
            }

        return index

    @staticmethod
    def _to_numpy(vec: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convert a vector to numpy float32."""
        if isinstance(vec, torch.Tensor):
            return vec.detach().cpu().float().numpy()
        return vec.astype(np.float32)
