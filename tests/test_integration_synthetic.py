"""
ENGRAM Protocol — Synthetic Integration Test
Full pipeline E2E with synthetic tensors — no real model needed.

Pipeline: create KV → extract state → serialize .eng → load → index → query → retrieve
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from kvcos.core.cache_spec import LLAMA_3_1_8B
from kvcos.core.serializer import EngramSerializer
from kvcos.core.types import CompressionMethod, StateExtractionMode
from kvcos.core.manifold_index import ManifoldIndex
from kvcos.core.retriever import EGRRetriever
from kvcos.core.state_extractor import MARStateExtractor
from kvcos.storage.local import LocalStorageBackend
from tests.conftest import make_synthetic_kv


class TestFullPipeline:
    """End-to-end: store → index → query → retrieve using synthetic data."""

    def test_serialize_round_trip(self, tmp_data_dir: Path) -> None:
        """Step 1-4: Create → serialize → load → verify shape."""
        keys, values = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=256)
        assert keys.shape == (32, 8, 256, 128)

        serializer = EngramSerializer()
        eng_path = tmp_data_dir / "roundtrip.eng"

        serializer.serialize(
            keys=keys, values=values,
            agent_id="integration-test", task_description="round-trip test",
            model_id=LLAMA_3_1_8B["model_id"], output_path=eng_path,
            compression=CompressionMethod.FP16,
        )
        assert eng_path.exists()

        # Verify valid safetensors
        tensors = load_file(str(eng_path))
        assert "layer_0_keys" in tensors

        k_out, v_out, meta = serializer.deserialize(eng_path)
        assert k_out.shape == keys.shape
        assert v_out.shape == values.shape

    @pytest.mark.parametrize("mode", list(StateExtractionMode))
    def test_extraction_all_modes(self, mode: StateExtractionMode) -> None:
        """Step 2: Extract state vector in all 3 modes."""
        keys, _ = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=256)
        extractor = MARStateExtractor(mode=mode, rank=128)
        result = extractor.extract(keys, LLAMA_3_1_8B)

        assert result.state_vec.dim() == 1
        assert result.state_vec.shape[0] > 0
        assert result.l2_norm > 0
        assert result.mode == mode

    def test_index_and_query(self, tmp_data_dir: Path) -> None:
        """Step 5-6: Index state vector → query with different tensor → get result."""
        keys_a, _ = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=256, seed=42)
        keys_b, _ = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=256, seed=99)

        extractor = MARStateExtractor(
            mode=StateExtractionMode.MEAN_POOL,
        )
        dim = extractor.output_dim(LLAMA_3_1_8B)
        index = ManifoldIndex(dim=dim)

        # Extract and index first tensor
        from kvcos.core.manifold_index import IndexEntry

        result_a = extractor.extract(keys_a, LLAMA_3_1_8B)
        index.add(
            result_a.state_vec,
            IndexEntry(
                cache_id="test-cache-a",
                task_description="indexed engram",
                model_id=LLAMA_3_1_8B["model_id"],
                created_at="2026-01-01T00:00:00Z",
                context_len=256,
                l2_norm=result_a.l2_norm,
            ),
        )

        # Query with second tensor
        result_b = extractor.extract(keys_b, LLAMA_3_1_8B)
        results = index.search(result_b.state_vec, top_k=1)

        assert len(results) >= 1
        assert results[0]["cache_id"] == "test-cache-a"

    def test_full_egr_pipeline(self, tmp_data_dir: Path) -> None:
        """Step 7: Full EGR retrieval — store → index → query → retrieve."""
        keys, values = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=256, seed=42)
        query_keys, _ = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=256, seed=99)

        extractor = MARStateExtractor(
            mode=StateExtractionMode.MEAN_POOL,
        )
        dim = extractor.output_dim(LLAMA_3_1_8B)
        index = ManifoldIndex(dim=dim)
        storage = LocalStorageBackend(data_dir=tmp_data_dir)
        retriever = EGRRetriever(extractor, index, storage)

        # Store
        cache_id = retriever.index_engram(
            keys=keys, values=values, spec=LLAMA_3_1_8B,
            agent_id="integration-test",
            task_description="full pipeline test",
            model_id=LLAMA_3_1_8B["model_id"],
            output_dir=tmp_data_dir,
        )
        assert isinstance(cache_id, str)
        assert index.n_entries == 1

        # Retrieve
        response = retriever.retrieve(query_keys, LLAMA_3_1_8B, top_k=1)
        assert len(response.results) >= 1

        result = response.results[0]
        assert result.cache_id == cache_id
        assert result.keys.shape == keys.shape
        assert result.values.shape == values.shape
        assert result.similarity != 0.0

    def test_multi_engram_ranking(self, tmp_data_dir: Path) -> None:
        """Store 3 engrams, query, verify results are ranked by similarity."""
        extractor = MARStateExtractor(mode=StateExtractionMode.MEAN_POOL)
        dim = extractor.output_dim(LLAMA_3_1_8B)
        index = ManifoldIndex(dim=dim)
        storage = LocalStorageBackend(data_dir=tmp_data_dir)
        retriever = EGRRetriever(extractor, index, storage)

        for seed in (10, 20, 30):
            keys, values = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=64, seed=seed)
            retriever.index_engram(
                keys=keys, values=values, spec=LLAMA_3_1_8B,
                agent_id="test", task_description=f"seed-{seed}",
                model_id=LLAMA_3_1_8B["model_id"],
                output_dir=tmp_data_dir,
            )

        assert index.n_entries == 3

        query_keys, _ = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=64, seed=10)
        response = retriever.retrieve(query_keys, LLAMA_3_1_8B, top_k=3)

        assert len(response.results) == 3
        # Results should be sorted by descending similarity
        sims = [r.similarity for r in response.results]
        assert sims == sorted(sims, reverse=True)
        # Closest match should be seed=10 (same as query)
        assert response.results[0].task_description == "seed-10"
