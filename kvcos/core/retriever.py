"""
Engrammatic Geometry Retrieval — Retriever


Orchestrates the full EGR retrieval pipeline:
  1. Extract state vector from query KV cache (MARStateExtractor)
  2. Search manifold index for similar engram states (ManifoldIndex)
  3. Load matched .eng files from storage (StorageBackend)
  4. Return ranked results with KV tensors ready for injection

This is the primary interface agents use for retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from kvcos.core.serializer import EngramSerializer
from kvcos.core.types import (
    CacheSearchResult,
    CompressionMethod,
    EngramMetadata,
    ModelCacheSpec,
    StateExtractionMode,
)
from kvcos.core.manifold_index import IndexEntry, ManifoldIndex
from kvcos.core.state_extractor import ExtractionResult, MARStateExtractor
from kvcos.storage.backends import StorageBackend


@dataclass
class RetrievalResult:
    """A single retrieval result with loaded KV tensors."""

    cache_id: str
    similarity: float
    task_description: str
    model_id: str
    keys: torch.Tensor  # [n_layers, n_kv_heads, ctx_len, head_dim]
    values: torch.Tensor  # [n_layers, n_kv_heads, ctx_len, head_dim]
    metadata: EngramMetadata


@dataclass
class RetrievalResponse:
    """Full response from a retrieval query."""

    query_extraction: ExtractionResult
    results: list[RetrievalResult]
    n_searched: int  # total entries in the index


class EGRRetriever:
    """Engrammatic Geometry Retrieval — full pipeline.

    Connects MARStateExtractor → ManifoldIndex → StorageBackend
    into a single retrieval call.

    Usage:
        retriever = EGRRetriever(extractor, index, storage)

        # Store an engram
        retriever.index_engram(keys, values, spec, agent_id, task_desc, model_id)

        # Retrieve similar engrams
        response = retriever.retrieve(query_keys, spec, top_k=3)
        for result in response.results:
            print(result.similarity, result.task_description)
            # result.keys / result.values ready for injection
    """

    def __init__(
        self,
        extractor: MARStateExtractor,
        index: ManifoldIndex,
        storage: StorageBackend,
        serializer: EngramSerializer | None = None,
    ):
        self.extractor = extractor
        self.index = index
        self.storage = storage
        self._serializer = serializer or EngramSerializer()

    def index_engram(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        spec: ModelCacheSpec,
        agent_id: str,
        task_description: str,
        model_id: str,
        cache_id: str | None = None,
        compression: CompressionMethod = CompressionMethod.Q8_0,
        output_dir: Path | None = None,
        extra_metadata: dict[str, str] | None = None,
    ) -> str:
        """Extract state vector, store .eng file, and add to index.

        This is the "write" path: compute once → store → index → reuse forever.

        Args:
            keys: [n_layers, n_kv_heads, ctx_len, head_dim]
            values: same shape as keys
            spec: Model architecture spec
            agent_id: Agent identifier
            task_description: Human-readable task description (searchable)
            model_id: Full model identifier
            cache_id: Explicit ID (auto-generated if None)
            compression: Compression method for storage
            output_dir: Directory for .eng file (uses storage backend default if None)
            extra_metadata: Additional metadata key-value pairs

        Returns:
            cache_id of the stored engram
        """
        import uuid
        from datetime import datetime, timezone

        from kvcos.core.types import ENG_FILE_EXTENSION

        cid = cache_id or str(uuid.uuid4())

        # 1. Extract state vector
        extraction = self.extractor.extract(keys, spec)

        # 2. Serialize to .eng file
        if output_dir:
            output_path = output_dir / f"{cid}{ENG_FILE_EXTENSION}"
        else:
            # Use a temp path; storage backend will move it
            import tempfile
            output_path = Path(tempfile.mkdtemp()) / f"{cid}{ENG_FILE_EXTENSION}"

        merge_meta = {
            "state_vec_norm": str(extraction.l2_norm),
            "extraction_mode": extraction.mode.value,
        }
        if extra_metadata:
            merge_meta.update(extra_metadata)

        result = self._serializer.serialize(
            keys=keys,
            values=values,
            agent_id=agent_id,
            task_description=task_description,
            model_id=model_id,
            output_path=output_path,
            compression=compression,
            cache_id=cid,
            extra_metadata=merge_meta,
        )

        # 3. Store in backend
        metadata = self._serializer.read_metadata_only(output_path)
        self.storage.store_file(cid, output_path, metadata)

        # 4. Add to manifold index
        now = datetime.now(timezone.utc).isoformat()
        entry = IndexEntry(
            cache_id=cid,
            task_description=task_description,
            model_id=model_id,
            created_at=now,
            context_len=keys.shape[2],
            l2_norm=extraction.l2_norm,
        )
        self.index.add(extraction.state_vec, entry)

        return cid

    def retrieve(
        self,
        query_keys: torch.Tensor,
        spec: ModelCacheSpec,
        top_k: int = 5,
        min_similarity: float | None = None,
        model_id: str | None = None,
        load_tensors: bool = True,
    ) -> RetrievalResponse:
        """Retrieve similar engram states for a query KV cache.

        This is the "read" path: extract query vector → search index →
        load matching .eng files.

        Args:
            query_keys: [n_layers, n_kv_heads, ctx_len, head_dim] query K cache
            spec: Model architecture spec
            top_k: Number of results to return
            min_similarity: Minimum MIPS score threshold
            model_id: Filter by model ID
            load_tensors: If True, load full KV tensors from storage.
                If False, return metadata only (faster for previewing).

        Returns:
            RetrievalResponse with ranked results
        """
        # 1. Extract query state vector
        query_extraction = self.extractor.extract(query_keys, spec)

        # 2. Search manifold index
        search_results = self.index.search(
            query_vec=query_extraction.state_vec,
            top_k=top_k,
            min_similarity=min_similarity,
            model_id=model_id,
        )

        # 3. Load matching engrams from storage
        results: list[RetrievalResult] = []
        for sr in search_results:
            if load_tensors:
                path = self.storage.get_path(sr["cache_id"])
                if path is None:
                    continue

                try:
                    keys, values, metadata = self._serializer.deserialize(path)
                except Exception:
                    continue

                results.append(RetrievalResult(
                    cache_id=sr["cache_id"],
                    similarity=sr["similarity"],
                    task_description=sr["task_description"],
                    model_id=sr["model_id"],
                    keys=keys,
                    values=values,
                    metadata=metadata,
                ))
            else:
                # Metadata-only mode
                metadata = self.storage.get_metadata(sr["cache_id"])
                if metadata is None:
                    continue

                results.append(RetrievalResult(
                    cache_id=sr["cache_id"],
                    similarity=sr["similarity"],
                    task_description=sr["task_description"],
                    model_id=sr["model_id"],
                    keys=torch.empty(0),
                    values=torch.empty(0),
                    metadata=metadata,
                ))

        return RetrievalResponse(
            query_extraction=query_extraction,
            results=results,
            n_searched=self.index.n_entries,
        )

    def delete_engram(self, cache_id: str) -> bool:
        """Remove an engram from both index and storage."""
        idx_removed = self.index.remove(cache_id)
        store_removed = self.storage.delete(cache_id)
        return idx_removed or store_removed

    def save_index(self, path: Path) -> None:
        """Persist the manifold index to disk."""
        self.index.save(path)
