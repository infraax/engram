"""
ENGRAM Protocol — llama.cpp State Blob Parser


Parses the binary state blob from llama_state_get_data() (via save_state())
into structured PyTorch tensors of shape [n_layers, n_kv_heads, n_cells, head_dim].

D1: This is the critical extraction path. The blob format is defined by
llama.cpp's llama_kv_cache::state_write() and is version-dependent.

Validated against llama-cpp-python 0.3.19 (llama.cpp b5000+).

Binary format of llama_state_get_data() output:
  1. Architecture string: uint32 str_len + str_len bytes (e.g. "llama")
  2. KV cache section (from memory->state_write()):
     a. uint32 n_stream (always 1 for single-context)
     b. Per stream:
        - uint32 cell_count (= n_used_cells, NOT n_ctx)
        - Per cell: int32 pos, uint32 n_seq_id, int32[] seq_ids
        - uint32 v_trans (1 = values stored transposed)
        - uint32 n_layer
        - Per layer K: int32 type_k, uint64 row_size_k, bytes data[row_size_k * cell_count]
        - Per layer V (non-transposed): int32 type_v, uint64 row_size_v, bytes data[row_size_v * cell_count]
        - Per layer V (transposed): int32 type_v, uint32 el_size, uint32 n_embd_v_gqa,
                                    bytes data[el_size * n_embd_v_gqa * cell_count]

WARNING: This format is not stable across llama.cpp versions.
Pin llama-cpp-python version in pyproject.toml.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import torch

from kvcos.core.types import CacheSection


# ── GGML dtype constants ──────────────────────────────────────────────────────

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_0 = 2

GGML_TYPE_SIZE: dict[int, float] = {
    GGML_TYPE_F32: 4.0,
    GGML_TYPE_F16: 2.0,
    GGML_TYPE_Q8_0: 34.0 / 32.0,
    GGML_TYPE_Q4_0: 18.0 / 32.0,
}

GGML_BLOCK_SIZE: dict[int, int] = {
    GGML_TYPE_F32: 1,
    GGML_TYPE_F16: 1,
    GGML_TYPE_Q8_0: 32,
    GGML_TYPE_Q4_0: 32,
}


@dataclass
class CellMeta:
    """Metadata for a single KV cache cell."""

    pos: int
    seq_ids: list[int]


@dataclass
class ParsedKVCache:
    """Result of parsing a llama.cpp state blob into structured engram tensors."""

    keys: torch.Tensor  # [n_layers, n_kv_heads, n_cells, head_dim] float16
    values: torch.Tensor  # [n_layers, n_kv_heads, n_cells, head_dim] float16
    cells: list[CellMeta]
    n_cells: int
    n_layers: int
    v_trans: bool
    arch: str


@dataclass
class ParsedMultiSectionCache:
    """Result of parsing an ISWA state blob with multiple KV cache sections.

    Each section is a ParsedKVCache with its own tensor shapes.
    For Gemma 4: section[0] is Global (5 layers), section[1] is SWA (25 layers).
    """

    sections: list[ParsedKVCache]
    arch: str

    @property
    def n_sections(self) -> int:
        return len(self.sections)

    @property
    def total_layers(self) -> int:
        return sum(s.n_layers for s in self.sections)


class BlobParseError(Exception):
    """Raised when the state blob cannot be parsed."""


def _read_u32(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<I", data, offset)[0], offset + 4


def _read_i32(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<i", data, offset)[0], offset + 4


def _read_u64(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<Q", data, offset)[0], offset + 8


def _read_f16_block(
    data: bytes, offset: int, n_elements: int,
) -> tuple[torch.Tensor, int]:
    """Read n_elements of float16 data from bytes."""
    n_bytes = n_elements * 2
    if offset + n_bytes > len(data):
        raise BlobParseError(
            f"F16 read overflow: need {n_bytes}B at offset {offset}, blob is {len(data)}B"
        )
    arr = np.frombuffer(data, dtype=np.float16, count=n_elements, offset=offset)
    return torch.from_numpy(arr.copy()).to(torch.float16), offset + n_bytes


def parse_state_blob(
    blob: bytes,
    n_kv_heads: int,
    head_dim: int,
) -> ParsedKVCache:
    """Parse a llama.cpp full-context state blob into structured KV tensors.

    Parses output of llama_state_get_data() (via save_state()):
      1. Architecture string header
      2. KV stream: cell metadata + per-layer K and V tensor data

    The parser auto-detects n_layers, cell_count, and v_trans from the blob.

    Args:
        blob: Raw bytes from save_state().llama_state
        n_kv_heads: Number of KV heads (from model spec)
        head_dim: Head dimension (from model spec)

    Returns:
        ParsedKVCache with [n_layers, n_kv_heads, n_cells, head_dim] tensors.
    """
    if len(blob) < 20:
        raise BlobParseError(f"Blob too small: {len(blob)} bytes")

    offset = 0
    n_embd_kv = n_kv_heads * head_dim

    # ── 1. Architecture string ────────────────────────────────
    str_len, offset = _read_u32(blob, offset)
    if str_len > 100:
        raise BlobParseError(f"Arch string length {str_len} too large — format mismatch")
    arch = blob[offset : offset + str_len].decode("ascii", errors="replace")
    offset += str_len

    # ── 2. KV stream header ───────────────────────────────────
    n_stream, offset = _read_u32(blob, offset)
    if n_stream != 1:
        raise BlobParseError(f"Expected 1 KV stream, got {n_stream}")

    cell_count, offset = _read_u32(blob, offset)
    if cell_count == 0:
        raise BlobParseError("State blob has 0 cells")
    if cell_count > 200_000:
        raise BlobParseError(f"Suspiciously large cell_count: {cell_count}")

    # ── 3. Cell metadata ──────────────────────────────────────
    cells: list[CellMeta] = []
    for _ in range(cell_count):
        pos, offset = _read_i32(blob, offset)
        n_seq, offset = _read_u32(blob, offset)
        seq_ids: list[int] = []
        for _ in range(n_seq):
            sid, offset = _read_i32(blob, offset)
            seq_ids.append(sid)
        cells.append(CellMeta(pos=pos, seq_ids=seq_ids))

    # ── 4. Data section header ────────────────────────────────
    v_trans_u32, offset = _read_u32(blob, offset)
    v_trans = v_trans_u32 != 0

    n_layers, offset = _read_u32(blob, offset)
    if n_layers == 0 or n_layers > 200:
        raise BlobParseError(f"Invalid n_layers: {n_layers}")

    # ── 5. K tensor data (per layer) ──────────────────────────
    k_layers: list[torch.Tensor] = []
    for layer_idx in range(n_layers):
        type_k, offset = _read_i32(blob, offset)
        row_size_k, offset = _read_u64(blob, offset)

        if type_k != GGML_TYPE_F16:
            raise BlobParseError(
                f"Layer {layer_idx} K: unsupported type {type_k} (expected F16={GGML_TYPE_F16})"
            )

        data_bytes = row_size_k * cell_count
        n_elements = data_bytes // 2  # fp16

        if n_elements != n_embd_kv * cell_count:
            raise BlobParseError(
                f"Layer {layer_idx} K: expected {n_embd_kv * cell_count} elements, "
                f"got {n_elements} (row_size={row_size_k}, cells={cell_count})"
            )

        tensor, offset = _read_f16_block(blob, offset, n_elements)
        # Shape: [cell_count, n_kv_heads * head_dim] → [n_kv_heads, cell_count, head_dim]
        tensor = tensor.reshape(cell_count, n_kv_heads, head_dim)
        tensor = tensor.permute(1, 0, 2).contiguous()
        k_layers.append(tensor)

    # ── 6. V tensor data (per layer) ──────────────────────────
    v_layers: list[torch.Tensor] = []
    for layer_idx in range(n_layers):
        type_v, offset = _read_i32(blob, offset)

        if type_v != GGML_TYPE_F16:
            raise BlobParseError(
                f"Layer {layer_idx} V: unsupported type {type_v} (expected F16={GGML_TYPE_F16})"
            )

        if v_trans:
            el_size, offset = _read_u32(blob, offset)
            n_embd_v, offset = _read_u32(blob, offset)
            data_bytes = el_size * n_embd_v * cell_count
            n_elements = data_bytes // 2

            tensor, offset = _read_f16_block(blob, offset, n_elements)
            # V transposed: stored as [n_embd_v, cell_count] per layer
            # n_embd_v = n_kv_heads * head_dim
            tensor = tensor.reshape(n_embd_v // head_dim, head_dim, cell_count)
            # → [n_kv_heads, head_dim, cell_count] → [n_kv_heads, cell_count, head_dim]
            tensor = tensor.permute(0, 2, 1).contiguous()
        else:
            row_size_v, offset = _read_u64(blob, offset)
            data_bytes = row_size_v * cell_count
            n_elements = data_bytes // 2

            tensor, offset = _read_f16_block(blob, offset, n_elements)
            tensor = tensor.reshape(cell_count, n_kv_heads, head_dim)
            tensor = tensor.permute(1, 0, 2).contiguous()

        v_layers.append(tensor)

    # ── 7. Stack into [n_layers, n_kv_heads, n_cells, head_dim] ─
    keys = torch.stack(k_layers, dim=0)
    values = torch.stack(v_layers, dim=0)

    expected_shape = (n_layers, n_kv_heads, cell_count, head_dim)
    if keys.shape != expected_shape:
        raise BlobParseError(f"K shape {keys.shape} != expected {expected_shape}")
    if values.shape != expected_shape:
        raise BlobParseError(f"V shape {values.shape} != expected {expected_shape}")

    return ParsedKVCache(
        keys=keys,
        values=values,
        cells=cells,
        n_cells=cell_count,
        n_layers=n_layers,
        v_trans=v_trans,
        arch=arch,
    )


def _parse_single_stream(
    blob: bytes,
    offset: int,
    n_kv_heads: int,
    head_dim: int,
    arch: str,
) -> tuple[ParsedKVCache, int]:
    """Parse one KV cache stream from blob at given offset.

    Returns (ParsedKVCache, new_offset) so caller can continue
    parsing subsequent streams for ISWA blobs.
    """
    n_embd_kv = n_kv_heads * head_dim

    # Cell count
    cell_count, offset = _read_u32(blob, offset)
    if cell_count == 0:
        raise BlobParseError("Stream has 0 cells")
    if cell_count > 200_000:
        raise BlobParseError(f"Suspiciously large cell_count: {cell_count}")

    # Cell metadata
    cells: list[CellMeta] = []
    for _ in range(cell_count):
        pos, offset = _read_i32(blob, offset)
        n_seq, offset = _read_u32(blob, offset)
        seq_ids: list[int] = []
        for _ in range(n_seq):
            sid, offset = _read_i32(blob, offset)
            seq_ids.append(sid)
        cells.append(CellMeta(pos=pos, seq_ids=seq_ids))

    # Data section header
    v_trans_u32, offset = _read_u32(blob, offset)
    v_trans = v_trans_u32 != 0

    n_layers, offset = _read_u32(blob, offset)
    if n_layers == 0 or n_layers > 200:
        raise BlobParseError(f"Invalid n_layers: {n_layers}")

    # K layers
    k_layers: list[torch.Tensor] = []
    for layer_idx in range(n_layers):
        type_k, offset = _read_i32(blob, offset)
        row_size_k, offset = _read_u64(blob, offset)

        if type_k != GGML_TYPE_F16:
            raise BlobParseError(
                f"Layer {layer_idx} K: unsupported type {type_k} (expected F16={GGML_TYPE_F16})"
            )

        data_bytes = row_size_k * cell_count
        n_elements = data_bytes // 2

        if n_elements != n_embd_kv * cell_count:
            raise BlobParseError(
                f"Layer {layer_idx} K: expected {n_embd_kv * cell_count} elements, "
                f"got {n_elements} (row_size={row_size_k}, cells={cell_count})"
            )

        tensor, offset = _read_f16_block(blob, offset, n_elements)
        tensor = tensor.reshape(cell_count, n_kv_heads, head_dim)
        tensor = tensor.permute(1, 0, 2).contiguous()
        k_layers.append(tensor)

    # V layers
    v_layers: list[torch.Tensor] = []
    for layer_idx in range(n_layers):
        type_v, offset = _read_i32(blob, offset)

        if type_v != GGML_TYPE_F16:
            raise BlobParseError(
                f"Layer {layer_idx} V: unsupported type {type_v} (expected F16={GGML_TYPE_F16})"
            )

        if v_trans:
            el_size, offset = _read_u32(blob, offset)
            n_embd_v, offset = _read_u32(blob, offset)
            data_bytes = el_size * n_embd_v * cell_count
            n_elements = data_bytes // 2

            tensor, offset = _read_f16_block(blob, offset, n_elements)
            tensor = tensor.reshape(n_embd_v // head_dim, head_dim, cell_count)
            tensor = tensor.permute(0, 2, 1).contiguous()
        else:
            row_size_v, offset = _read_u64(blob, offset)
            data_bytes = row_size_v * cell_count
            n_elements = data_bytes // 2

            tensor, offset = _read_f16_block(blob, offset, n_elements)
            tensor = tensor.reshape(cell_count, n_kv_heads, head_dim)
            tensor = tensor.permute(1, 0, 2).contiguous()

        v_layers.append(tensor)

    keys = torch.stack(k_layers, dim=0)
    values = torch.stack(v_layers, dim=0)

    expected_shape = (n_layers, n_kv_heads, cell_count, head_dim)
    if keys.shape != expected_shape:
        raise BlobParseError(f"K shape {keys.shape} != expected {expected_shape}")
    if values.shape != expected_shape:
        raise BlobParseError(f"V shape {values.shape} != expected {expected_shape}")

    parsed = ParsedKVCache(
        keys=keys,
        values=values,
        cells=cells,
        n_cells=cell_count,
        n_layers=n_layers,
        v_trans=v_trans,
        arch=arch,
    )
    return parsed, offset


def parse_multi_section_blob(
    blob: bytes,
    sections: tuple[CacheSection, ...],
) -> ParsedMultiSectionCache:
    """Parse an ISWA state blob with multiple sequential KV cache sections.

    ISWA models (e.g., Gemma 4) serialize multiple cache sections in a single
    blob. Each section has its own cell metadata, layer count, and KV dimensions.
    The n_stream field in the blob header equals the number of sections.

    Args:
        blob: Raw bytes from save_state().llama_state
        sections: Cache section specifications (order must match blob layout)

    Returns:
        ParsedMultiSectionCache with one ParsedKVCache per section.
    """
    if len(blob) < 20:
        raise BlobParseError(f"Blob too small: {len(blob)} bytes")

    offset = 0

    # Architecture string
    str_len, offset = _read_u32(blob, offset)
    if str_len > 100:
        raise BlobParseError(f"Arch string length {str_len} too large")
    arch = blob[offset : offset + str_len].decode("ascii", errors="replace")
    offset += str_len

    # Stream count
    n_stream, offset = _read_u32(blob, offset)
    if n_stream != len(sections):
        raise BlobParseError(
            f"Expected {len(sections)} streams, got {n_stream}"
        )

    # Parse each stream
    parsed_sections: list[ParsedKVCache] = []
    for section in sections:
        parsed, offset = _parse_single_stream(
            blob, offset,
            n_kv_heads=section.n_kv_heads,
            head_dim=section.head_dim,
            arch=arch,
        )
        parsed_sections.append(parsed)

    return ParsedMultiSectionCache(sections=parsed_sections, arch=arch)


# ── Legacy compat wrapper ────────────────────────────────────────────────────


def parse_seq_state_blob(
    blob: bytes,
    spec: dict,
    kv_dtype: int = GGML_TYPE_F16,
) -> ParsedKVCache:
    """Legacy wrapper — delegates to parse_state_blob.

    Kept for backward compatibility with existing tests.
    """
    return parse_state_blob(
        blob=blob,
        n_kv_heads=spec["n_kv_heads"],
        head_dim=spec["head_dim"],
    )


def estimate_blob_size(
    n_kv_heads: int,
    head_dim: int,
    n_layers: int,
    n_cells: int,
    v_trans: bool = True,
) -> int:
    """Estimate expected blob size for validation."""
    header = 4 + 5 + 4 + 4  # str_len + "llama" + n_stream + cell_count
    cell_meta = n_cells * 12  # pos(4) + n_seq(4) + seq_id(4) typical
    data_header = 4 + 4  # v_trans + n_layer

    n_embd_kv = n_kv_heads * head_dim
    k_per_layer = 4 + 8 + (n_embd_kv * 2 * n_cells)  # type + row_size + data
    if v_trans:
        v_per_layer = 4 + 4 + 4 + (n_embd_kv * 2 * n_cells)  # type + el_size + n_embd + data
    else:
        v_per_layer = 4 + 8 + (n_embd_kv * 2 * n_cells)

    return header + cell_meta + data_header + n_layers * (k_per_layer + v_per_layer)
