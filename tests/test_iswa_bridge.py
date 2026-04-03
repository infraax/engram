"""
ENGRAM Protocol — ISWA Bridge Tests
Tests for multi-architecture metadata detection and ISWA cache extraction.
Does NOT require a real GGUF model — tests the metadata helpers and spec logic.
"""

from __future__ import annotations

import pytest

from integrations.llama_cpp_bridge import _meta_get
from kvcos.core.cache_spec import (
    GEMMA_4_26B_A4B,
    LLAMA_3_1_8B,
    get_model_spec,
    is_iswa_spec,
)


class TestMetaGet:
    """Metadata key fallback chain across architecture prefixes."""

    def test_llama_prefix(self) -> None:
        meta = {"llama.block_count": "32"}
        assert _meta_get(meta, "block_count") == "32"

    def test_gemma4_prefix(self) -> None:
        meta = {"gemma4.block_count": "30"}
        assert _meta_get(meta, "block_count") == "30"

    def test_gemma_prefix(self) -> None:
        meta = {"gemma.attention.head_count": "8"}
        assert _meta_get(meta, "attention.head_count") == "8"

    def test_general_fallback(self) -> None:
        meta = {"general.block_count": "28"}
        assert _meta_get(meta, "block_count") == "28"

    def test_default_when_missing(self) -> None:
        meta = {}
        assert _meta_get(meta, "block_count", "32") == "32"

    def test_llama_takes_priority(self) -> None:
        meta = {
            "llama.block_count": "32",
            "gemma4.block_count": "30",
            "general.block_count": "28",
        }
        assert _meta_get(meta, "block_count") == "32"

    def test_gemma4_before_general(self) -> None:
        meta = {
            "gemma4.embedding_length": "3072",
            "general.embedding_length": "4096",
        }
        assert _meta_get(meta, "embedding_length") == "3072"


class TestISWASpecDetection:
    """Registry and ISWA detection."""

    def test_gemma4_in_registry(self) -> None:
        spec = get_model_spec("google/gemma-4-26b-a4b-it")
        assert spec is not None
        assert spec["model_family"] == "gemma"

    def test_gemma4_is_iswa(self) -> None:
        assert is_iswa_spec(GEMMA_4_26B_A4B) is True

    def test_llama_not_iswa(self) -> None:
        assert is_iswa_spec(LLAMA_3_1_8B) is False

    def test_gemma4_sections_correct(self) -> None:
        sections = GEMMA_4_26B_A4B["cache_sections"]
        assert len(sections) == 2

        # Global section
        assert sections[0].n_layers == 5
        assert sections[0].n_kv_heads == 2
        assert sections[0].head_dim == 512
        assert sections[0].n_embd_kv == 1024

        # SWA section
        assert sections[1].n_layers == 25
        assert sections[1].n_kv_heads == 8
        assert sections[1].head_dim == 256
        assert sections[1].window_size == 1024

    def test_gemma4_total_layers(self) -> None:
        sections = GEMMA_4_26B_A4B["cache_sections"]
        total = sum(s.n_layers for s in sections)
        assert total == GEMMA_4_26B_A4B["n_layers"]
