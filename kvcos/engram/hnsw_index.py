"""
ENGRAM HNSW Index — O(log N) approximate nearest neighbor retrieval.

Wraps faiss.IndexHNSWFlat for production-scale ENGRAM search.
Primary fingerprint: v2 layer-normalized Fourier f0+f1.

Usage:
    idx = EngramIndex(dim=2048)
    idx.add_batch(doc_ids, vectors)
    results = idx.search(query_fp, top_k=5)
    idx.save('index/hnsw')
    idx2 = EngramIndex.load('index/hnsw')
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import faiss
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class HNSWResult:
    """Single HNSW search result."""

    doc_id: str
    score: float
    rank: int
    margin: float = 0.0


class EngramIndex:
    """HNSW-backed ENGRAM retrieval index.

    HNSW parameters:
        M=32: graph degree (higher = better recall, more memory)
        efConstruction=200: build-time search width
        efSearch=64: query-time search width
    """

    M = 32
    EF_CONSTRUCTION = 200
    EF_SEARCH = 64

    def __init__(self, dim: int = 2048):
        self._dim = dim
        self._index: faiss.IndexHNSWFlat | None = None
        self._ids: list[str] = []
        self._id_to_pos: dict[str, int] = {}
        self._n_docs: int = 0

    def add_batch(
        self,
        doc_ids: list[str],
        vectors: torch.Tensor,
    ) -> None:
        """Build HNSW index from vectors.

        Args:
            doc_ids: list of document identifiers
            vectors: [N, dim] tensor of fingerprints
        """
        matrix = F.normalize(vectors.float(), dim=-1).numpy().astype("float32")
        self._dim = matrix.shape[1]
        self._ids = list(doc_ids)
        self._id_to_pos = {cid: i for i, cid in enumerate(doc_ids)}
        self._n_docs = len(doc_ids)

        self._index = faiss.IndexHNSWFlat(self._dim, self.M)
        self._index.hnsw.efConstruction = self.EF_CONSTRUCTION
        self._index.hnsw.efSearch = self.EF_SEARCH
        self._index.add(matrix)

    def build(
        self,
        eng_files: list[str],
        fp_key: str = "vec_fourier_v2",
        verbose: bool = True,
    ) -> None:
        """Build HNSW index from list of .eng file paths.

        Args:
            eng_files: List of paths to .eng encoded files.
            fp_key:    Fingerprint field to index.
                       Default 'vec_fourier_v2' (S3 validated, 99.5% recall).
                       Falls back to 'vec_fourier' if v2 not present.
        """
        from kvcos.engram.reader import read_eigengram

        doc_ids = []
        vecs = []
        missing_v2 = 0

        for fp in eng_files:
            data = read_eigengram(fp)
            cid = data.get("cache_id")
            if not cid:
                continue

            vec = data.get(fp_key)
            if vec is None:
                vec = data.get("vec_fourier")
                missing_v2 += 1
            if vec is None:
                continue

            doc_ids.append(cid)
            vecs.append(vec.float())

        if not vecs:
            raise ValueError(
                f"No valid fingerprints found in {len(eng_files)} files"
            )

        if missing_v2 > 0 and verbose:
            logger.warning(
                "%d docs missing %s, used vec_fourier fallback",
                missing_v2, fp_key,
            )

        self.add_batch(doc_ids, torch.stack(vecs))

        if verbose:
            logger.info("HNSW index built: %d docs, dim=%d", self._n_docs, self._dim)
            logger.info("M=%d, efC=%d, efS=%d", self.M, self.EF_CONSTRUCTION, self.EF_SEARCH)

    def search(
        self,
        query_fp: torch.Tensor,
        top_k: int = 5,
    ) -> list[HNSWResult]:
        """Search the HNSW index.

        Returns list of HNSWResult sorted by score descending.
        HNSW uses L2 on normalized vectors: cosine = 1 - L2^2/2.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call add_batch() or load() first.")

        qn = F.normalize(query_fp.float().unsqueeze(0), dim=-1).numpy().astype("float32")
        D, I = self._index.search(qn, min(top_k + 1, self._n_docs))

        results = []
        for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx < 0:
                continue
            cosine_sim = float(1.0 - dist / 2.0)
            results.append(HNSWResult(
                doc_id=self._ids[idx], score=cosine_sim, rank=rank,
            ))

        if len(results) >= 2:
            results[0].margin = results[0].score - results[1].score
        return results[:top_k]

    def save(self, path: str) -> None:
        """Save index to disk (faiss + JSON metadata)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index(self._index, path + ".faiss")
        meta_path = path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "ids": self._ids,
                "id_to_pos": self._id_to_pos,
                "dim": self._dim,
                "n_docs": self._n_docs,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> EngramIndex:
        """Load index from disk."""
        obj = cls()
        obj._index = faiss.read_index(path + ".faiss")
        meta_path = path + ".meta.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
        obj._ids = meta["ids"]
        obj._id_to_pos = meta["id_to_pos"]
        obj._dim = meta["dim"]
        obj._n_docs = meta["n_docs"]
        return obj

    def __len__(self) -> int:
        return self._n_docs


    def get_vector(self, doc_id: str) -> torch.Tensor | None:
        """Return stored vector for doc_id, or None if not found."""
        pos = self._id_to_pos.get(doc_id)
        if pos is None:
            return None
        vec_np = np.zeros(self._dim, dtype="float32")
        self._index.reconstruct(pos, vec_np)
        return torch.from_numpy(vec_np)

    def __repr__(self) -> str:
        return f"EngramIndex(n={self._n_docs}, dim={self._dim}, M={self.M})"
