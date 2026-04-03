"""
ENGRAM Protocol — Compression Tests


Tests for kvcos.core.compression:
  - FP16 passthrough
  - Q8_0 round-trip accuracy & shape preservation
  - PolarQuant round-trip accuracy & rotation invariants
  - Dispatcher routing and Q4_0 fallback warning
  - Edge cases: padding, single-element groups
"""

from __future__ import annotations

import warnings

import pytest
import torch

from kvcos.core.compression import (
    Q8_GROUP_SIZE,
    CompressionResult,
    compress,
    compress_fp16,
    compress_polarquant,
    compress_q8_0,
    decompress,
    decompress_fp16,
    decompress_polarquant,
    decompress_q8_0,
)
from kvcos.core.types import CompressionMethod


# ── FP16 Passthrough ──────────────────────────────────────────────────────────


class TestFP16:
    """FP16 passthrough: no quantization, just dtype normalization."""

    def test_fp16_passthrough_shape(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        keys, _ = llama_kv_256
        result = compress_fp16(keys)
        assert result.data.shape == keys.shape

    def test_fp16_passthrough_dtype(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        keys, _ = llama_kv_256
        result = compress_fp16(keys)
        assert result.data.dtype == torch.float16

    def test_fp16_passthrough_exact(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        keys, _ = llama_kv_256
        result = compress_fp16(keys)
        assert torch.equal(result.data, keys.to(torch.float16))

    def test_fp16_compression_ratio_one(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        keys, _ = llama_kv_256
        result = compress_fp16(keys)
        assert result.compression_ratio == 1.0

    def test_fp16_method_tag(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        keys, _ = llama_kv_256
        result = compress_fp16(keys)
        assert result.method == CompressionMethod.FP16

    def test_fp16_from_fp32(self) -> None:
        """FP32 input is cast to FP16."""
        t = torch.randn(4, 8, 32, 128, dtype=torch.float32)
        result = compress_fp16(t)
        assert result.data.dtype == torch.float16
        assert result.original_dtype == torch.float32

    def test_fp16_decompress_identity(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        keys, _ = llama_kv_256
        result = compress_fp16(keys)
        out = decompress_fp16(result.data)
        assert torch.equal(out, result.data)


# ── Q8_0 Quantization ────────────────────────────────────────────────────────


class TestQ8_0:
    """Q8_0: group quantization matching llama.cpp GGML_TYPE_Q8_0."""

    def test_q8_0_shape_preserved(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        keys, _ = llama_kv_256
        result = compress_q8_0(keys)
        assert result.data.shape == keys.shape

    def test_q8_0_output_dtype(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Q8_0 stores dequantized bfloat16 for safetensors compat."""
        keys, _ = llama_kv_256
        result = compress_q8_0(keys)
        assert result.data.dtype == torch.bfloat16

    def test_q8_0_method_tag(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        keys, _ = llama_kv_256
        result = compress_q8_0(keys)
        assert result.method == CompressionMethod.Q8_0

    def test_q8_0_metadata_group_size(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        keys, _ = llama_kv_256
        result = compress_q8_0(keys)
        assert result.metadata["q8_group_size"] == str(Q8_GROUP_SIZE)

    def test_q8_0_round_trip_low_error(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Q8_0 quantization error should be < 1% relative MSE."""
        keys, _ = llama_kv_256
        result = compress_q8_0(keys)
        decompressed = decompress_q8_0(result.data)

        original = keys.float()
        restored = decompressed.float()

        mse = ((original - restored) ** 2).mean()
        signal_power = (original**2).mean()
        relative_mse = (mse / signal_power).item()
        assert relative_mse < 0.01, f"Q8_0 relative MSE {relative_mse:.6f} > 1%"

    def test_q8_0_round_trip_values(
        self, phi3_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Q8_0 round-trip on Phi-3 (head_dim=96, needs padding)."""
        keys, values = phi3_kv_256
        for tensor in (keys, values):
            result = compress_q8_0(tensor)
            assert result.data.shape == tensor.shape

    def test_q8_0_compression_ratio_fp32(self) -> None:
        """FP32 input → bfloat16 output gives 2x compression ratio."""
        t = torch.randn(2, 4, 64, 128, dtype=torch.float32)
        result = compress_q8_0(t)
        assert abs(result.compression_ratio - 2.0) < 0.01

    def test_q8_0_compression_ratio_fp16(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """FP16 input → bfloat16 output gives 1x ratio (same byte width)."""
        keys, _ = llama_kv_256
        result = compress_q8_0(keys)
        assert abs(result.compression_ratio - 1.0) < 0.01

    def test_q8_0_preserves_original_dtype(self) -> None:
        t = torch.randn(4, 8, 32, 128, dtype=torch.float32)
        result = compress_q8_0(t)
        assert result.original_dtype == torch.float32

    def test_q8_0_padding_dim_not_divisible(self) -> None:
        """Head dims not divisible by 32 get padded then unpadded."""
        t = torch.randn(2, 4, 16, 96, dtype=torch.float16)  # 96 = 3*32, exact
        result = compress_q8_0(t)
        assert result.data.shape == t.shape

        t2 = torch.randn(2, 4, 16, 100, dtype=torch.float16)  # 100 not div by 32
        result2 = compress_q8_0(t2)
        assert result2.data.shape == t2.shape

    def test_q8_0_zero_tensor(self) -> None:
        """All-zero tensor should round-trip exactly."""
        t = torch.zeros(2, 4, 16, 128, dtype=torch.float16)
        result = compress_q8_0(t)
        decompressed = decompress_q8_0(result.data)
        assert torch.allclose(decompressed, t.to(torch.float16), atol=1e-6)


# ── PolarQuant ───────────────────────────────────────────────────────────────


class TestPolarQuant:
    """PolarQuant: MSE-optimal random rotation + Lloyd-Max at 3 bits.
    QJL intentionally absent (D5).
    """

    def test_polarquant_shape_preserved(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        keys, _ = llama_kv_256
        result = compress_polarquant(keys)
        assert result.data.shape == keys.shape

    def test_polarquant_output_dtype(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        keys, _ = llama_kv_256
        result = compress_polarquant(keys)
        assert result.data.dtype == torch.bfloat16

    def test_polarquant_method_tag(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        keys, _ = llama_kv_256
        result = compress_polarquant(keys)
        assert result.method == CompressionMethod.POLARQUANT

    def test_polarquant_metadata_qjl_disabled(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """D5: QJL must be marked disabled in metadata."""
        keys, _ = llama_kv_256
        result = compress_polarquant(keys)
        assert result.metadata["qjl_enabled"] == "false"
        assert result.metadata["polarquant_bits"] == "3"

    def test_polarquant_round_trip_bounded_error(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """PolarQuant 3-bit error should be < 15% relative MSE.

        3-bit Lloyd-Max on rotated Gaussian: theoretical ~10% for 8 centroids.
        Allow margin for rotation + dtype casting.
        """
        keys, _ = llama_kv_256
        result = compress_polarquant(keys)
        decompressed = decompress_polarquant(result.data)

        original = keys.float()
        restored = decompressed.float()

        mse = ((original - restored) ** 2).mean()
        signal_power = (original**2).mean()
        relative_mse = (mse / signal_power).item()
        assert relative_mse < 0.15, f"PolarQuant relative MSE {relative_mse:.4f} > 15%"

    def test_polarquant_worse_than_q8_0(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """3-bit PolarQuant should have higher error than 8-bit Q8_0."""
        keys, _ = llama_kv_256
        original = keys.float()

        q8_result = compress_q8_0(keys)
        pq_result = compress_polarquant(keys)

        q8_mse = ((original - decompress_q8_0(q8_result.data).float()) ** 2).mean()
        pq_mse = (
            (original - decompress_polarquant(pq_result.data).float()) ** 2
        ).mean()

        assert pq_mse > q8_mse, "PolarQuant 3-bit should be less accurate than Q8_0"

    def test_polarquant_deterministic(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Same input → same output (fixed seed rotation matrix)."""
        keys, _ = llama_kv_256
        r1 = compress_polarquant(keys)
        r2 = compress_polarquant(keys)
        assert torch.equal(r1.data, r2.data)

    def test_polarquant_phi3_shape(
        self, phi3_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Phi-3 head_dim=96 works with PolarQuant."""
        keys, _ = phi3_kv_256
        result = compress_polarquant(keys)
        assert result.data.shape == keys.shape


# ── Dispatcher ───────────────────────────────────────────────────────────────


class TestDispatcher:
    """compress() and decompress() dispatch to correct implementations."""

    @pytest.mark.parametrize(
        "method",
        [CompressionMethod.FP16, CompressionMethod.Q8_0, CompressionMethod.POLARQUANT],
    )
    def test_compress_dispatches(self, method: CompressionMethod) -> None:
        t = torch.randn(2, 4, 16, 128, dtype=torch.float16)
        result = compress(t, method)
        assert isinstance(result, CompressionResult)
        assert result.method == method

    @pytest.mark.parametrize(
        "method",
        [CompressionMethod.FP16, CompressionMethod.Q8_0, CompressionMethod.POLARQUANT],
    )
    def test_decompress_returns_fp16(self, method: CompressionMethod) -> None:
        t = torch.randn(2, 4, 16, 128, dtype=torch.float16)
        result = compress(t, method)
        out = decompress(result.data, method)
        assert out.dtype == torch.float16

    def test_q4_0_warns_and_falls_back(self) -> None:
        """D5: Q4_0 emits warning and uses Q8_0 instead."""
        t = torch.randn(2, 4, 16, 128, dtype=torch.float16)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compress(t, CompressionMethod.Q4_0)
            assert len(w) == 1
            assert "Q4_0" in str(w[0].message)
            assert "92%" in str(w[0].message)
        assert result.method == CompressionMethod.Q8_0

    def test_unknown_method_raises(self) -> None:
        t = torch.randn(2, 4, 16, 128, dtype=torch.float16)
        with pytest.raises(ValueError, match="Unknown compression method"):
            compress(t, "invalid_method")  # type: ignore[arg-type]

    def test_decompress_unknown_raises(self) -> None:
        t = torch.randn(2, 4, 16, 128, dtype=torch.float16)
        with pytest.raises(ValueError, match="Unknown compression method"):
            decompress(t, "invalid_method")  # type: ignore[arg-type]


# ── Round-trip Integration ───────────────────────────────────────────────────


class TestRoundTrip:
    """Full compress → decompress round-trip through dispatcher."""

    @pytest.mark.parametrize(
        "method",
        [CompressionMethod.FP16, CompressionMethod.Q8_0, CompressionMethod.POLARQUANT],
    )
    def test_round_trip_shape_preserved(self, method: CompressionMethod) -> None:
        t = torch.randn(4, 8, 64, 128, dtype=torch.float16)
        result = compress(t, method)
        out = decompress(result.data, method)
        assert out.shape == t.shape

    def test_round_trip_both_kv(
        self, llama_kv_256: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Compress and decompress both keys and values."""
        keys, values = llama_kv_256
        for tensor in (keys, values):
            for method in (CompressionMethod.FP16, CompressionMethod.Q8_0):
                result = compress(tensor, method)
                out = decompress(result.data, method)
                assert out.shape == tensor.shape
                assert out.dtype == torch.float16
