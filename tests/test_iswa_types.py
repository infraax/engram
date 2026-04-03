"""
ENGRAM Protocol — ISWA Type System Tests
Tests for CacheSection, AttentionType, and ISWA-aware ModelCacheSpec.
"""

from __future__ import annotations

import pytest

from kvcos.core.types import AttentionType, CacheSection, ModelCacheSpec


class TestAttentionType:
    """AttentionType enum values."""

    def test_full_value(self) -> None:
        assert AttentionType.FULL == "full"

    def test_sliding_value(self) -> None:
        assert AttentionType.SLIDING == "sliding"

    def test_is_str(self) -> None:
        assert isinstance(AttentionType.FULL, str)


class TestCacheSection:
    """CacheSection frozen dataclass."""

    def test_global_section(self) -> None:
        sec = CacheSection(
            attention_type=AttentionType.FULL,
            n_layers=5,
            n_kv_heads=2,
            head_dim=512,
        )
        assert sec.n_layers == 5
        assert sec.n_kv_heads == 2
        assert sec.head_dim == 512
        assert sec.window_size is None

    def test_sliding_section(self) -> None:
        sec = CacheSection(
            attention_type=AttentionType.SLIDING,
            n_layers=25,
            n_kv_heads=8,
            head_dim=256,
            window_size=1024,
        )
        assert sec.attention_type == AttentionType.SLIDING
        assert sec.window_size == 1024

    def test_n_embd_kv(self) -> None:
        sec = CacheSection(
            attention_type=AttentionType.FULL,
            n_layers=5,
            n_kv_heads=2,
            head_dim=512,
        )
        assert sec.n_embd_kv == 1024  # 2 * 512

    def test_frozen(self) -> None:
        sec = CacheSection(
            attention_type=AttentionType.FULL,
            n_layers=5,
            n_kv_heads=2,
            head_dim=512,
        )
        with pytest.raises(AttributeError):
            sec.n_layers = 10  # type: ignore[misc]

    def test_equality(self) -> None:
        a = CacheSection(AttentionType.FULL, 5, 2, 512)
        b = CacheSection(AttentionType.FULL, 5, 2, 512)
        assert a == b

    def test_inequality(self) -> None:
        a = CacheSection(AttentionType.FULL, 5, 2, 512)
        b = CacheSection(AttentionType.SLIDING, 25, 8, 256, 1024)
        assert a != b


class TestModelCacheSpecISWA:
    """ModelCacheSpec with optional cache_sections."""

    def test_standard_spec_no_sections(self) -> None:
        spec = ModelCacheSpec(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            model_family="llama",
            n_layers=32,
            n_heads=32,
            n_kv_heads=8,
            head_dim=128,
            rope_enabled=True,
            extraction_layers=tuple(range(8, 32)),
        )
        assert "cache_sections" not in spec
        assert spec["n_kv_heads"] == 8

    def test_iswa_spec_with_sections(self) -> None:
        sections = (
            CacheSection(AttentionType.FULL, 5, 2, 512),
            CacheSection(AttentionType.SLIDING, 25, 8, 256, 1024),
        )
        spec = ModelCacheSpec(
            model_id="google/gemma-4-26b-a4b-it",
            model_family="gemma",
            n_layers=30,
            n_heads=32,
            n_kv_heads=8,
            head_dim=256,
            rope_enabled=True,
            extraction_layers=tuple(range(8, 30)),
            cache_sections=sections,
        )
        assert "cache_sections" in spec
        assert len(spec["cache_sections"]) == 2
        assert spec["cache_sections"][0].n_embd_kv == 1024
        assert spec["cache_sections"][1].n_embd_kv == 2048

    def test_iswa_total_layers_match(self) -> None:
        sections = (
            CacheSection(AttentionType.FULL, 5, 2, 512),
            CacheSection(AttentionType.SLIDING, 25, 8, 256, 1024),
        )
        total = sum(s.n_layers for s in sections)
        assert total == 30

    def test_backward_compat_existing_specs(self) -> None:
        """Existing specs without cache_sections still work."""
        from kvcos.core.cache_spec import LLAMA_3_1_8B
        assert LLAMA_3_1_8B["n_kv_heads"] == 8
        assert "cache_sections" not in LLAMA_3_1_8B
