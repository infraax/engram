"""
ENGRAM Protocol — ISWA Fingerprint Tests
Tests for per-section Fourier fingerprint computation and concatenation.
"""

from __future__ import annotations

import torch

from kvcos.core.blob_parser import ParsedKVCache, ParsedMultiSectionCache, parse_multi_section_blob
from kvcos.core.fingerprint import (
    compute_fourier_fingerprint_v2,
    compute_iswa_fingerprint,
)
from kvcos.core.types import AttentionType, CacheSection
from tests.conftest import GEMMA4_SECTIONS, make_synthetic_iswa_blob


class TestISWAFingerprint:
    """Per-section fingerprint computation for ISWA models."""

    def _make_parsed(self, n_cells: int = 4) -> ParsedMultiSectionCache:
        blob = make_synthetic_iswa_blob(GEMMA4_SECTIONS, n_cells=n_cells)
        return parse_multi_section_blob(blob, GEMMA4_SECTIONS)

    def test_fingerprint_shape(self) -> None:
        parsed = self._make_parsed()
        fp = compute_iswa_fingerprint(parsed, freqs=[0, 1])

        # Global: 2 * 512 * 2 = 2048
        # SWA:    8 * 256 * 2 = 4096
        # Total:  6144
        assert fp.shape == (6144,)

    def test_fingerprint_dtype(self) -> None:
        parsed = self._make_parsed()
        fp = compute_iswa_fingerprint(parsed)
        assert fp.dtype == torch.float32

    def test_fingerprint_normalized(self) -> None:
        """Each section's sub-FP is concat of per-freq L2-normalized vectors."""
        parsed = self._make_parsed()
        fp = compute_iswa_fingerprint(parsed, freqs=[0, 1])

        # Global section FP: first 2048 dims (1024 per freq, 2 freqs)
        global_fp = fp[:2048]
        # SWA section FP: next 4096 dims (2048 per freq, 2 freqs)
        swa_fp = fp[2048:]

        # Each sub-section is 2 concatenated unit vectors → norm = sqrt(2)
        import math
        expected_norm = math.sqrt(2)
        assert abs(global_fp.norm().item() - expected_norm) < 0.05
        assert abs(swa_fp.norm().item() - expected_norm) < 0.05

    def test_deterministic(self) -> None:
        parsed = self._make_parsed()
        fp1 = compute_iswa_fingerprint(parsed)
        fp2 = compute_iswa_fingerprint(parsed)
        assert torch.allclose(fp1, fp2)

    def test_different_inputs_differ(self) -> None:
        p1 = self._make_parsed(n_cells=4)
        blob2 = make_synthetic_iswa_blob(GEMMA4_SECTIONS, n_cells=4, seed=999)
        p2 = parse_multi_section_blob(blob2, GEMMA4_SECTIONS)

        fp1 = compute_iswa_fingerprint(p1)
        fp2 = compute_iswa_fingerprint(p2)
        cos = torch.nn.functional.cosine_similarity(fp1.unsqueeze(0), fp2.unsqueeze(0))
        assert cos.item() < 0.99  # different inputs → different FPs

    def test_single_section_matches_standard(self) -> None:
        """Single-section ISWA FP should match standard FP."""
        section = CacheSection(AttentionType.FULL, 5, 2, 512)
        blob = make_synthetic_iswa_blob((section,), n_cells=4)
        parsed = parse_multi_section_blob(blob, (section,))

        iswa_fp = compute_iswa_fingerprint(parsed, freqs=[0, 1])

        # Compare with standard FP on same data
        layer_keys = parsed.sections[0].keys.float().mean(dim=2)
        standard_fp = compute_fourier_fingerprint_v2(layer_keys, freqs=[0, 1])

        assert torch.allclose(iswa_fp, standard_fp, atol=1e-5)

    def test_custom_freqs(self) -> None:
        parsed = self._make_parsed()
        fp_f0 = compute_iswa_fingerprint(parsed, freqs=[0])
        fp_f01 = compute_iswa_fingerprint(parsed, freqs=[0, 1])

        # f0 only: Global(1024) + SWA(2048) = 3072
        assert fp_f0.shape == (3072,)
        # f0+f1: double
        assert fp_f01.shape == (6144,)
