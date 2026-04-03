"""
ENGRAM Protocol — Blob Parser Tests
Tests for llama.cpp state blob → structured tensors (D1).
Uses synthetic blobs matching the real llama_state_get_data() format.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from kvcos.core.blob_parser import (
    GGML_TYPE_F16,
    BlobParseError,
    ParsedKVCache,
    parse_state_blob,
)


def _make_blob(
    n_cells: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    arch: str = "llama",
    v_trans: bool = True,
) -> bytes:
    """Build a synthetic blob matching llama_state_get_data() format."""
    parts: list[bytes] = []

    # 1. Architecture string header
    parts.append(struct.pack("<I", len(arch)))
    parts.append(arch.encode("ascii"))

    # 2. KV stream header
    parts.append(struct.pack("<I", 1))  # n_stream = 1
    parts.append(struct.pack("<I", n_cells))  # cell_count

    # 3. Cell metadata: (pos:i32, n_seq:u32, seq_id:i32) per cell
    for i in range(n_cells):
        parts.append(struct.pack("<i", i))  # pos
        parts.append(struct.pack("<I", 1))  # n_seq_id = 1
        parts.append(struct.pack("<i", 0))  # seq_id = 0

    # 4. Data section header
    parts.append(struct.pack("<I", 1 if v_trans else 0))  # v_trans
    parts.append(struct.pack("<I", n_layers))

    n_embd_kv = n_kv_heads * head_dim
    row_size = n_embd_kv * 2  # fp16

    # 5. K layers
    for _ in range(n_layers):
        parts.append(struct.pack("<i", GGML_TYPE_F16))  # type_k
        parts.append(struct.pack("<Q", row_size))  # row_size_k
        data = np.random.randn(n_cells * n_embd_kv).astype(np.float16)
        parts.append(data.tobytes())

    # 6. V layers
    for _ in range(n_layers):
        parts.append(struct.pack("<i", GGML_TYPE_F16))  # type_v
        if v_trans:
            parts.append(struct.pack("<I", 2))  # el_size (fp16)
            parts.append(struct.pack("<I", n_embd_kv))  # n_embd_v_gqa
        else:
            parts.append(struct.pack("<Q", row_size))  # row_size_v
        data = np.random.randn(n_cells * n_embd_kv).astype(np.float16)
        parts.append(data.tobytes())

    return b"".join(parts)


class TestBlobParser:
    """Parse synthetic blobs in real llama_state_get_data format."""

    def test_parse_shape(self) -> None:
        blob = _make_blob(16, 32, 8, 128)
        result = parse_state_blob(blob, n_kv_heads=8, head_dim=128)
        assert result.keys.shape == (32, 8, 16, 128)
        assert result.values.shape == (32, 8, 16, 128)

    def test_parse_metadata(self) -> None:
        blob = _make_blob(8, 32, 8, 128)
        result = parse_state_blob(blob, n_kv_heads=8, head_dim=128)
        assert result.n_cells == 8
        assert result.n_layers == 32
        assert result.arch == "llama"
        assert result.v_trans is True
        assert len(result.cells) == 8
        assert result.cells[0].pos == 0
        assert result.cells[7].pos == 7

    def test_dtype_float16(self) -> None:
        blob = _make_blob(4, 28, 8, 128)
        result = parse_state_blob(blob, n_kv_heads=8, head_dim=128)
        assert result.keys.dtype == torch.float16
        assert result.values.dtype == torch.float16

    def test_non_transposed_v(self) -> None:
        blob = _make_blob(4, 28, 8, 128, v_trans=False)
        result = parse_state_blob(blob, n_kv_heads=8, head_dim=128)
        assert result.values.shape == (28, 8, 4, 128)
        assert result.v_trans is False


class TestBlobParserErrors:
    """Edge cases."""

    def test_zero_cells_raises(self) -> None:
        blob = struct.pack("<I", 5) + b"llama" + struct.pack("<II", 1, 0) + b"\x00" * 20
        with pytest.raises(BlobParseError, match="0 cells"):
            parse_state_blob(blob, n_kv_heads=8, head_dim=128)

    def test_truncated_blob_raises(self) -> None:
        blob = _make_blob(4, 28, 8, 128)
        with pytest.raises(BlobParseError):
            parse_state_blob(blob[:100], n_kv_heads=8, head_dim=128)

    def test_bad_arch_length_raises(self) -> None:
        blob = struct.pack("<I", 999) + b"x" * 100
        with pytest.raises(BlobParseError, match="too large"):
            parse_state_blob(blob, n_kv_heads=8, head_dim=128)
