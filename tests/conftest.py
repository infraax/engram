"""
ENGRAM Protocol — Test Fixtures


Shared pytest fixtures for all test modules.
Provides synthetic KV cache tensors at correct shapes,
temp directories, and model specs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kvcos.core.cache_spec import GEMMA_4_26B_A4B, LLAMA_3_1_8B, PHI_3_MINI
from kvcos.core.types import AttentionType, CacheSection, ModelCacheSpec


@pytest.fixture
def llama_spec() -> ModelCacheSpec:
    """Llama 3.1 8B model spec."""
    return LLAMA_3_1_8B


@pytest.fixture
def phi3_spec() -> ModelCacheSpec:
    """Phi-3-Mini model spec."""
    return PHI_3_MINI


@pytest.fixture
def gemma4_spec() -> ModelCacheSpec:
    """Gemma 4 26B-A4B ISWA model spec."""
    return GEMMA_4_26B_A4B


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary data directory for storage tests."""
    data_dir = tmp_path / "engram_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def tmp_index_dir(tmp_path: Path) -> Path:
    """Temporary directory for FAISS index persistence tests."""
    index_dir = tmp_path / "engram_index"
    index_dir.mkdir()
    return index_dir


def make_synthetic_kv(
    spec: ModelCacheSpec,
    ctx_len: int = 256,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic KV cache tensors with correct shapes.

    Returns (keys, values) each [n_layers, n_kv_heads, ctx_len, head_dim].
    Values are random but reproducible via seed.
    """
    torch.manual_seed(seed)
    shape = (spec["n_layers"], spec["n_kv_heads"], ctx_len, spec["head_dim"])
    keys = torch.randn(shape, dtype=torch.float16)
    values = torch.randn(shape, dtype=torch.float16)
    return keys, values


@pytest.fixture
def llama_kv_256(llama_spec: ModelCacheSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic Llama 3.1 8B KV cache, 256 tokens.

    Shape: [32, 8, 256, 128] for both keys and values.
    """
    return make_synthetic_kv(llama_spec, ctx_len=256)


@pytest.fixture
def llama_kv_1024(llama_spec: ModelCacheSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic Llama 3.1 8B KV cache, 1024 tokens."""
    return make_synthetic_kv(llama_spec, ctx_len=1024, seed=123)


@pytest.fixture
def phi3_kv_256(phi3_spec: ModelCacheSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic Phi-3-Mini KV cache, 256 tokens.

    Shape: [32, 32, 256, 96] for both keys and values.
    """
    return make_synthetic_kv(phi3_spec, ctx_len=256, seed=99)


# ── ISWA Fixtures ────────────────────────────────────────────────────────────


def make_synthetic_iswa_blob(
    sections: tuple[CacheSection, ...],
    n_cells: int = 4,
    arch: str = "gemma4",
    v_trans: bool = True,
    seed: int = 42,
) -> bytes:
    """Build a synthetic ISWA blob with multiple KV cache sections.

    Matches llama.cpp state blob format for ISWA models:
      1. Architecture string header
      2. n_stream = len(sections)
      3. Per stream: cell metadata + K/V data per layer

    Args:
        sections: Cache sections (e.g., global + SWA for Gemma 4).
        n_cells: Number of KV cells per section.
        arch: Architecture string in blob header.
        v_trans: Whether V tensors are stored transposed.
        seed: Random seed for reproducible data.
    """
    import struct

    import numpy as np

    from kvcos.core.blob_parser import GGML_TYPE_F16

    rng = np.random.RandomState(seed)
    parts: list[bytes] = []

    # 1. Architecture string header
    parts.append(struct.pack("<I", len(arch)))
    parts.append(arch.encode("ascii"))

    # 2. Stream count = number of cache sections
    parts.append(struct.pack("<I", len(sections)))

    # 3. Per-stream data
    for section in sections:
        n_embd_kv = section.n_kv_heads * section.head_dim
        row_size = n_embd_kv * 2  # fp16

        # Cell metadata
        parts.append(struct.pack("<I", n_cells))
        for i in range(n_cells):
            parts.append(struct.pack("<i", i))    # pos
            parts.append(struct.pack("<I", 1))    # n_seq_id = 1
            parts.append(struct.pack("<i", 0))    # seq_id = 0

        # Data section header
        parts.append(struct.pack("<I", 1 if v_trans else 0))
        parts.append(struct.pack("<I", section.n_layers))

        # K layers
        for _ in range(section.n_layers):
            parts.append(struct.pack("<i", GGML_TYPE_F16))
            parts.append(struct.pack("<Q", row_size))
            data = rng.randn(n_cells * n_embd_kv).astype(np.float16)
            parts.append(data.tobytes())

        # V layers
        for _ in range(section.n_layers):
            parts.append(struct.pack("<i", GGML_TYPE_F16))
            if v_trans:
                parts.append(struct.pack("<I", 2))         # el_size (fp16)
                parts.append(struct.pack("<I", n_embd_kv)) # n_embd_v_gqa
            else:
                parts.append(struct.pack("<Q", row_size))
            data = rng.randn(n_cells * n_embd_kv).astype(np.float16)
            parts.append(data.tobytes())

    return b"".join(parts)


# Gemma 4 ISWA section constants (reverse-engineered)
GEMMA4_GLOBAL_SECTION = CacheSection(
    attention_type=AttentionType.FULL,
    n_layers=5,
    n_kv_heads=2,
    head_dim=512,
)

GEMMA4_SWA_SECTION = CacheSection(
    attention_type=AttentionType.SLIDING,
    n_layers=25,
    n_kv_heads=8,
    head_dim=256,
    window_size=1024,
)

GEMMA4_SECTIONS = (GEMMA4_GLOBAL_SECTION, GEMMA4_SWA_SECTION)


@pytest.fixture
def gemma4_iswa_blob() -> bytes:
    """Synthetic Gemma 4 ISWA blob with 2 sections, 4 cells."""
    return make_synthetic_iswa_blob(GEMMA4_SECTIONS, n_cells=4)


@pytest.fixture
def gemma4_iswa_blob_8cells() -> bytes:
    """Synthetic Gemma 4 ISWA blob with 2 sections, 8 cells."""
    return make_synthetic_iswa_blob(GEMMA4_SECTIONS, n_cells=8, seed=99)
