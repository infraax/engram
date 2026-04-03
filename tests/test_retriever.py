"""
ENGRAM Protocol — Retriever Tests
Tests for EGRRetriever: store → index → query → retrieve pipeline.
"""

from __future__ import annotations

from pathlib import Path

import torch

from kvcos.core.cache_spec import LLAMA_3_1_8B
from kvcos.core.serializer import EngramSerializer
from kvcos.core.types import CompressionMethod, StateExtractionMode
from kvcos.core.manifold_index import ManifoldIndex
from kvcos.core.retriever import EGRRetriever, RetrievalResponse
from kvcos.core.state_extractor import MARStateExtractor
from kvcos.storage.local import LocalStorageBackend
from tests.conftest import make_synthetic_kv


def _build_retriever(
    data_dir: Path, mode: StateExtractionMode = StateExtractionMode.MEAN_POOL,
) -> EGRRetriever:
    ext = MARStateExtractor(mode=mode, rank=128)
    dim = ext.output_dim(LLAMA_3_1_8B)
    idx = ManifoldIndex(dim=dim)
    storage = LocalStorageBackend(data_dir=data_dir)
    return EGRRetriever(ext, idx, storage)


class TestIndexAndRetrieve:
    """Full store → search → load pipeline."""

    def test_index_returns_cache_id(self, tmp_data_dir: Path) -> None:
        keys, values = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=64)
        retriever = _build_retriever(tmp_data_dir)

        cid = retriever.index_engram(
            keys=keys, values=values, spec=LLAMA_3_1_8B,
            agent_id="test", task_description="test engram",
            model_id=LLAMA_3_1_8B["model_id"],
            output_dir=tmp_data_dir,
        )
        assert isinstance(cid, str)
        assert len(cid) > 0

    def test_retrieve_finds_stored(self, tmp_data_dir: Path) -> None:
        keys, values = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=64)
        retriever = _build_retriever(tmp_data_dir)

        retriever.index_engram(
            keys=keys, values=values, spec=LLAMA_3_1_8B,
            agent_id="test", task_description="findable engram",
            model_id=LLAMA_3_1_8B["model_id"],
            output_dir=tmp_data_dir,
        )

        query_keys, _ = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=64, seed=99)
        response = retriever.retrieve(query_keys, LLAMA_3_1_8B, top_k=1)

        assert isinstance(response, RetrievalResponse)
        assert len(response.results) == 1
        assert response.results[0].keys.shape == keys.shape

    def test_retrieve_empty_index(self, tmp_data_dir: Path) -> None:
        retriever = _build_retriever(tmp_data_dir)
        query_keys, _ = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=64)
        response = retriever.retrieve(query_keys, LLAMA_3_1_8B, top_k=5)
        assert len(response.results) == 0

    def test_delete_removes(self, tmp_data_dir: Path) -> None:
        keys, values = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=64)
        retriever = _build_retriever(tmp_data_dir)

        cid = retriever.index_engram(
            keys=keys, values=values, spec=LLAMA_3_1_8B,
            agent_id="test", task_description="deletable",
            model_id=LLAMA_3_1_8B["model_id"],
            output_dir=tmp_data_dir,
        )
        assert retriever.delete_engram(cid)

        query_keys, _ = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=64)
        response = retriever.retrieve(query_keys, LLAMA_3_1_8B, top_k=5)
        assert len(response.results) == 0
