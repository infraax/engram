"""
ENGRAM Protocol — State Extractor Tests
Tests for all 3 EGR extraction modes (D3).
"""

from __future__ import annotations

import torch

from kvcos.core.cache_spec import LLAMA_3_1_8B, PHI_3_MINI
from kvcos.core.types import StateExtractionMode
from kvcos.core.state_extractor import MARStateExtractor
from tests.conftest import make_synthetic_kv


class TestMeanPool:
    """mean_pool: mean over layers, heads, context → [head_dim]."""

    def test_output_dim(self) -> None:
        keys, _ = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=64)
        ext = MARStateExtractor(mode=StateExtractionMode.MEAN_POOL)
        result = ext.extract(keys, LLAMA_3_1_8B)
        assert result.state_vec.shape == (128,)

    def test_output_dim_api(self) -> None:
        ext = MARStateExtractor(mode=StateExtractionMode.MEAN_POOL)
        assert ext.output_dim(LLAMA_3_1_8B) == 128

    def test_l2_norm_positive(self) -> None:
        keys, _ = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=64)
        ext = MARStateExtractor(mode=StateExtractionMode.MEAN_POOL)
        result = ext.extract(keys, LLAMA_3_1_8B)
        assert result.l2_norm > 0

    def test_deterministic(self) -> None:
        keys, _ = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=64)
        ext = MARStateExtractor(mode=StateExtractionMode.MEAN_POOL)
        r1 = ext.extract(keys, LLAMA_3_1_8B)
        r2 = ext.extract(keys, LLAMA_3_1_8B)
        assert torch.equal(r1.state_vec, r2.state_vec)


class TestSVDProject:
    """svd_project: truncated SVD, rank-160 → [rank]."""

    def test_output_dim(self) -> None:
        keys, _ = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=64)
        ext = MARStateExtractor(mode=StateExtractionMode.SVD_PROJECT, rank=160)
        result = ext.extract(keys, LLAMA_3_1_8B)
        assert result.state_vec.shape == (128,)  # clamped to head_dim

    def test_projection_stored(self) -> None:
        keys, _ = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=64)
        ext = MARStateExtractor(mode=StateExtractionMode.SVD_PROJECT, rank=160)
        ext.extract(keys, LLAMA_3_1_8B)
        proj = ext.last_projection
        assert proj is not None
        assert 0.0 < proj.explained_variance_ratio <= 1.0

    def test_n_layers_used(self) -> None:
        keys, _ = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=64)
        ext = MARStateExtractor(mode=StateExtractionMode.SVD_PROJECT)
        result = ext.extract(keys, LLAMA_3_1_8B)
        assert result.n_layers_used == 24  # layers 8-31


class TestXKVProject:
    """xkv_project: grouped cross-layer SVD."""

    def test_output_dim(self) -> None:
        keys, _ = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=64)
        ext = MARStateExtractor(mode=StateExtractionMode.XKV_PROJECT, rank=160)
        result = ext.extract(keys, LLAMA_3_1_8B)
        expected_dim = ext.output_dim(LLAMA_3_1_8B)
        assert result.state_vec.shape == (expected_dim,)

    def test_different_from_mean_pool(self) -> None:
        keys, _ = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=64)
        ext_mp = MARStateExtractor(mode=StateExtractionMode.MEAN_POOL)
        ext_xkv = MARStateExtractor(mode=StateExtractionMode.XKV_PROJECT)
        r_mp = ext_mp.extract(keys, LLAMA_3_1_8B)
        r_xkv = ext_xkv.extract(keys, LLAMA_3_1_8B)
        assert r_mp.state_vec.shape != r_xkv.state_vec.shape

    def test_phi3_works(self) -> None:
        keys, _ = make_synthetic_kv(PHI_3_MINI, ctx_len=64)
        ext = MARStateExtractor(mode=StateExtractionMode.XKV_PROJECT, rank=96)
        result = ext.extract(keys, PHI_3_MINI)
        assert result.state_vec.dim() == 1
        assert result.state_vec.shape[0] > 0
