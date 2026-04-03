"""
ENGRAM Protocol — ISWA Blob Parser Tests
Tests for multi-section KV cache parsing (Gemma 4 ISWA format).

Uses synthetic ISWA blobs from conftest.make_synthetic_iswa_blob().
"""

from __future__ import annotations

import pytest
import torch

from kvcos.core.blob_parser import (
    BlobParseError,
    ParsedKVCache,
    ParsedMultiSectionCache,
    parse_multi_section_blob,
    parse_state_blob,
)
from kvcos.core.types import AttentionType, CacheSection, ModelCacheSpec
from tests.conftest import (
    GEMMA4_GLOBAL_SECTION,
    GEMMA4_SECTIONS,
    GEMMA4_SWA_SECTION,
    make_synthetic_iswa_blob,
)


class TestParseMultiSectionBlob:
    """Parse ISWA blobs with multiple KV cache sections."""

    def test_parse_gemma4_shape(self) -> None:
        blob = make_synthetic_iswa_blob(GEMMA4_SECTIONS, n_cells=4)
        result = parse_multi_section_blob(blob, GEMMA4_SECTIONS)

        assert len(result.sections) == 2

        # Global section: [5, 2, 4, 512]
        s0 = result.sections[0]
        assert s0.keys.shape == (5, 2, 4, 512)
        assert s0.values.shape == (5, 2, 4, 512)

        # SWA section: [25, 8, 4, 256]
        s1 = result.sections[1]
        assert s1.keys.shape == (25, 8, 4, 256)
        assert s1.values.shape == (25, 8, 4, 256)

    def test_parse_metadata(self) -> None:
        blob = make_synthetic_iswa_blob(GEMMA4_SECTIONS, n_cells=4)
        result = parse_multi_section_blob(blob, GEMMA4_SECTIONS)

        assert result.arch == "gemma4"
        assert result.n_sections == 2
        assert result.total_layers == 30

        assert result.sections[0].n_layers == 5
        assert result.sections[0].arch == "gemma4"
        assert result.sections[1].n_layers == 25

    def test_parse_cells(self) -> None:
        blob = make_synthetic_iswa_blob(GEMMA4_SECTIONS, n_cells=4)
        result = parse_multi_section_blob(blob, GEMMA4_SECTIONS)

        for sec in result.sections:
            assert sec.n_cells == 4
            assert len(sec.cells) == 4
            assert sec.cells[0].pos == 0
            assert sec.cells[3].pos == 3

    def test_dtype_float16(self) -> None:
        blob = make_synthetic_iswa_blob(GEMMA4_SECTIONS, n_cells=2)
        result = parse_multi_section_blob(blob, GEMMA4_SECTIONS)

        for sec in result.sections:
            assert sec.keys.dtype == torch.float16
            assert sec.values.dtype == torch.float16

    def test_different_cell_counts(self) -> None:
        blob = make_synthetic_iswa_blob(GEMMA4_SECTIONS, n_cells=8)
        result = parse_multi_section_blob(blob, GEMMA4_SECTIONS)

        assert result.sections[0].n_cells == 8
        assert result.sections[1].n_cells == 8

    def test_non_transposed_v(self) -> None:
        blob = make_synthetic_iswa_blob(GEMMA4_SECTIONS, n_cells=2, v_trans=False)
        result = parse_multi_section_blob(blob, GEMMA4_SECTIONS)

        for sec in result.sections:
            assert sec.v_trans is False

    def test_single_section_works(self) -> None:
        """Single-section ISWA parse should work identically to standard."""
        single = (GEMMA4_GLOBAL_SECTION,)
        blob = make_synthetic_iswa_blob(single, n_cells=4)
        result = parse_multi_section_blob(blob, single)

        assert len(result.sections) == 1
        assert result.sections[0].keys.shape == (5, 2, 4, 512)


class TestParseMultiSectionErrors:
    """Error handling for ISWA blob parsing."""

    def test_section_mismatch_raises(self) -> None:
        """Blob has 2 sections but we pass specs for 3."""
        blob = make_synthetic_iswa_blob(GEMMA4_SECTIONS, n_cells=4)
        three_sections = GEMMA4_SECTIONS + (GEMMA4_GLOBAL_SECTION,)
        with pytest.raises(BlobParseError, match="Expected 3.*got 2"):
            parse_multi_section_blob(blob, three_sections)

    def test_truncated_blob_raises(self) -> None:
        blob = make_synthetic_iswa_blob(GEMMA4_SECTIONS, n_cells=4)
        with pytest.raises(BlobParseError):
            parse_multi_section_blob(blob[:100], GEMMA4_SECTIONS)

    def test_wrong_dimensions_raises(self) -> None:
        """Pass wrong KV head count for a section."""
        wrong_sections = (
            CacheSection(AttentionType.FULL, 5, 4, 512),  # wrong: 4 heads not 2
            GEMMA4_SWA_SECTION,
        )
        blob = make_synthetic_iswa_blob(GEMMA4_SECTIONS, n_cells=4)
        with pytest.raises(BlobParseError):
            parse_multi_section_blob(blob, wrong_sections)


class TestStandardBlobBackwardCompat:
    """Ensure parse_state_blob still works for single-stream blobs."""

    def test_single_stream_still_works(self) -> None:
        from tests.test_blob_parser import _make_blob

        blob = _make_blob(16, 32, 8, 128)
        result = parse_state_blob(blob, n_kv_heads=8, head_dim=128)
        assert result.keys.shape == (32, 8, 16, 128)
