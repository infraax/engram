"""
kvcos/engram/knowledge_index.py — HNSW index over the knowledge store.

Builds and maintains a faiss HNSW index over all .eng files in
~/.engram/knowledge/. Supports dynamic dimension (384 for sbert,
2048 for llama_cpp/hash) — determined at build time from the first
.eng file.

Usage:
    # Build from all knowledge .eng files
    kidx = KnowledgeIndex.build_from_knowledge_dir()
    results = kidx.search("HNSW recall benchmark", k=5)
    kidx.save()

    # Load pre-built index
    kidx = KnowledgeIndex.load()
    results = kidx.search("testing patterns", k=3)

Index files:
    ~/.engram/index/knowledge.faiss
    ~/.engram/index/knowledge.meta
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F

from kvcos.engram.embedder import get_fingerprint
from kvcos.engram.format import EigramEncoder

logger = logging.getLogger(__name__)


INDEX_DIR = Path(
    os.environ.get("ENGRAM_INDEX_DIR", "~/.engram/index")
).expanduser()

KNOWLEDGE_DIR = Path(
    os.environ.get("ENGRAM_KNOWLEDGE_DIR", "~/.engram/knowledge")
).expanduser()

INDEX_NAME = "knowledge"

_encoder = EigramEncoder()


@dataclass(frozen=True)
class KnowledgeResult:
    """Single search result from the knowledge index."""
    doc_id: str
    score: float
    rank: int
    source_path: str
    project: str
    content: str
    chunk_info: str     # "2/5" format
    headers: list[str]
    margin: float = 0.0


class KnowledgeIndex:
    """HNSW index over the ENGRAM knowledge store.

    Parameters match EngramIndex for consistency:
        M=32, efConstruction=200, efSearch=64
    """

    M = 32
    EF_CONSTRUCTION = 200
    EF_SEARCH = 64

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        self._index: faiss.IndexHNSWFlat | None = None
        self._meta: list[dict] = []   # per-vector metadata
        self._n_docs: int = 0

    @classmethod
    def build_from_knowledge_dir(
        cls,
        knowledge_dir: Path | None = None,
        verbose: bool = True,
    ) -> KnowledgeIndex:
        """Build HNSW index from all .eng files in the knowledge directory."""
        if knowledge_dir is None:
            knowledge_dir = KNOWLEDGE_DIR

        eng_files = sorted(knowledge_dir.rglob("*.eng"), key=os.path.getmtime)
        eng_files = [p for p in eng_files if p.suffix == ".eng"]

        if not eng_files:
            raise ValueError(f"No .eng files found in {knowledge_dir}")

        vectors: list[torch.Tensor] = []
        metas: list[dict] = []
        skipped = 0

        for p in eng_files:
            try:
                data = _encoder.decode(p.read_bytes())

                fp = data.get("vec_fourier_v2")
                if fp is None:
                    fp = data.get("vec_fourier")
                if fp is None:
                    skipped += 1
                    continue

                # Load sidecar metadata
                meta_path = Path(str(p) + ".meta.json")
                meta = {}
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())

                # Use sidecar description if longer than binary
                description = meta.get("task_description", "") or \
                    data.get("task_description", "")

                vectors.append(fp.float())
                metas.append({
                    "doc_id": data.get("cache_id", p.stem),
                    "source_path": meta.get("source_path", ""),
                    "project": meta.get("project", ""),
                    "content": description,
                    "chunk_index": meta.get("chunk_index", 0),
                    "chunk_total": meta.get("chunk_total", 1),
                    "headers": meta.get("headers", []),
                    "fp_source": meta.get("fp_source", "unknown"),
                })
            except Exception as exc:
                logger.debug("Skipping %s: %s", p, exc)
                skipped += 1

        if not vectors:
            raise ValueError(
                f"No valid fingerprints in {len(eng_files)} .eng files"
            )

        # Stack and determine dimension from actual data
        matrix = torch.stack(vectors)
        dim = matrix.shape[1]

        # Normalize for cosine similarity via L2
        matrix = F.normalize(matrix, dim=-1).numpy().astype("float32")

        # Build HNSW
        obj = cls(dim=dim)
        obj._index = faiss.IndexHNSWFlat(dim, cls.M)
        obj._index.hnsw.efConstruction = cls.EF_CONSTRUCTION
        obj._index.hnsw.efSearch = cls.EF_SEARCH
        obj._index.add(matrix)
        obj._meta = metas
        obj._n_docs = len(metas)

        if verbose:
            projects = {m["project"] for m in metas}
            logger.info("Knowledge HNSW: %d vectors, dim=%d", obj._n_docs, dim)
            logger.info("Projects: %s", sorted(projects))
            if skipped:
                logger.warning("Skipped: %d files (no fingerprint)", skipped)

        return obj

    def search(
        self,
        query: str | torch.Tensor,
        k: int = 5,
    ) -> list[KnowledgeResult]:
        """
        Search the knowledge index.

        Args:
            query: Search text (will be fingerprinted) or pre-computed tensor.
            k: Number of results to return.

        Returns:
            List of KnowledgeResult sorted by score descending.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_from_knowledge_dir() first.")

        if isinstance(query, str):
            query_fp, _ = get_fingerprint(query)
        else:
            query_fp = query

        qn = F.normalize(
            query_fp.float().unsqueeze(0), dim=-1
        ).numpy().astype("float32")

        top = min(k + 1, self._n_docs)
        D, I = self._index.search(qn, top)

        results: list[KnowledgeResult] = []
        for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx < 0 or idx >= len(self._meta):
                continue
            meta = self._meta[idx]
            cosine = float(1.0 - dist / 2.0)
            ci = meta.get("chunk_index", 0)
            ct = meta.get("chunk_total", 1)

            results.append(KnowledgeResult(
                doc_id=meta["doc_id"],
                score=cosine,
                rank=rank,
                source_path=meta.get("source_path", ""),
                project=meta.get("project", ""),
                content=meta.get("content", ""),
                chunk_info=f"{ci + 1}/{ct}",
                headers=meta.get("headers", []),
            ))

        # Set margin on top result
        if len(results) >= 2:
            results[0] = KnowledgeResult(
                doc_id=results[0].doc_id,
                score=results[0].score,
                rank=results[0].rank,
                source_path=results[0].source_path,
                project=results[0].project,
                content=results[0].content,
                chunk_info=results[0].chunk_info,
                headers=results[0].headers,
                margin=results[0].score - results[1].score,
            )

        return results[:k]

    def save(self, index_dir: Path | None = None) -> Path:
        """Save index to disk."""
        if index_dir is None:
            index_dir = INDEX_DIR
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss_path = index_dir / f"{INDEX_NAME}.faiss"
        meta_path = index_dir / f"{INDEX_NAME}.meta.json"

        faiss.write_index(self._index, str(faiss_path))
        with open(meta_path, "w") as f:
            json.dump({
                "meta": self._meta,
                "dim": self._dim,
                "n_docs": self._n_docs,
            }, f, indent=2)

        return faiss_path

    @classmethod
    def load(cls, index_dir: Path | None = None) -> KnowledgeIndex:
        """Load pre-built index from disk."""
        if index_dir is None:
            index_dir = INDEX_DIR

        faiss_path = index_dir / f"{INDEX_NAME}.faiss"
        meta_path = index_dir / f"{INDEX_NAME}.meta.json"

        if not faiss_path.exists():
            raise FileNotFoundError(
                f"No knowledge index at {faiss_path}. "
                "Build with KnowledgeIndex.build_from_knowledge_dir()"
            )

        obj = cls()
        obj._index = faiss.read_index(str(faiss_path))
        with open(meta_path, "r") as f:
            data = json.load(f)
        obj._meta = data["meta"]
        obj._dim = data["dim"]
        obj._n_docs = data["n_docs"]
        return obj

    def __len__(self) -> int:
        return self._n_docs

    def __repr__(self) -> str:
        return (
            f"KnowledgeIndex(n={self._n_docs}, dim={self._dim}, "
            f"M={self.M})"
        )
