"""
ENGRAM Protocol — Core Type Definitions


All enums, TypedDicts, constants, and type aliases live here.
Every downstream module imports from this file. No circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypedDict

# ── Constants ─────────────────────────────────────────────────────────────────

ENGRAM_VERSION = "0.1.0"
ENG_FILE_EXTENSION = ".eng"  # ENGRAM file format extension
BLOCK_SIZE_TOKENS = 256  # 256-token blocks per arXiv:2603.04428
DEFAULT_SVD_RANK = 160  # ShadowKV default for 8B models
DEFAULT_LATENT_DIM = 512  # MLA: full KV info recoverable from 512-dim
MAX_CONTEXT_TOKENS = 131072  # 128K max supported context


# ── Enums ─────────────────────────────────────────────────────────────────────


class CompressionMethod(StrEnum):
    """Supported KV cache compression methods.

    Phase 1: Q8_0, FP16
    Phase 2: POLARQUANT (TurboQuant without QJL — QJL removed per D5)
    """

    FP16 = "fp16"
    Q8_0 = "q8_0"  # llama.cpp GGML_TYPE_Q8_0: ~2x compression, <5% speed hit
    Q4_0 = "q4_0"  # NOT recommended at 64K+ (92% dequant slowdown)
    POLARQUANT = "polarquant_3bit"  # PolarQuant only, no QJL
    INT8 = "int8"  # Phase 2: true int8 + per-row scale, 2x on-disk compression
    LAYER_DELTA = "layer_delta"  # Phase 2: fp16 baseline + int8 inter-layer deltas


class StorageBackend(StrEnum):
    """Supported storage backends."""

    LOCAL = "local"
    REDIS = "redis"  # Phase 2
    S3 = "s3"  # Phase 2


class StateExtractionMode(StrEnum):
    """EGR (Engrammatic Geometry Retrieval) state vector extraction modes.

    mean_pool:   Fast baseline. Mean over heads + context of key matrices.
    svd_project: Truncated SVD on pre-RoPE keys, layers 8-31, rank-160.
                 Validated by ShadowKV (ICML 2025) on Llama-3.1-8B.
    xkv_project: Grouped cross-layer SVD, 4-layer groups, K:V rank 1:1.5.
                 From xKV (arXiv:2503.18893). 6.8x compression.

    REMOVED: sals_project — last-layer-only extraction invalidated by
    Layer-Condensed KV Cache (ACL 2024). See D3.
    """

    MEAN_POOL = "mean_pool"
    SVD_PROJECT = "svd_project"
    XKV_PROJECT = "xkv_project"


class IndexBackend(StrEnum):
    """EGR manifold index backends."""

    FAISS_FLAT_IP = "faiss_flat_ip"  # Phase 1: exact MIPS
    FAISS_IVF_IP = "faiss_ivf_ip"  # Phase 2: approximate MIPS for >100K vectors
    QDRANT_DOT = "qdrant_dot"  # Phase 2: production persistent index


class AttentionType(StrEnum):
    """KV cache attention mechanism per layer group.

    Standard models use FULL for all layers.
    ISWA models (Gemma 4) interleave FULL (global) and SLIDING (SWA) sections.
    """

    FULL = "full"        # Full-context attention (standard)
    SLIDING = "sliding"  # Sliding window attention (SWA)


# ── Data Classes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CacheSection:
    """One section of a multi-section KV cache.

    Standard models have a single implicit section covering all layers.
    ISWA models serialize multiple sections sequentially in the state blob,
    each with its own n_layers, n_kv_heads, and head_dim.

    Reverse-engineered from Gemma 4 26B-A4B (llama.cpp b5200+):
      Section 0 (Global): 5 layers, 2 KV heads, head_dim=512
      Section 1 (SWA):   25 layers, 8 KV heads, head_dim=256
    """

    attention_type: AttentionType
    n_layers: int
    n_kv_heads: int
    head_dim: int
    window_size: int | None = None  # SWA window size in tokens (None for full)

    @property
    def n_embd_kv(self) -> int:
        """Total KV embedding dimension for this section."""
        return self.n_kv_heads * self.head_dim


# ── TypedDicts ────────────────────────────────────────────────────────────────


class _ModelCacheSpecRequired(TypedDict):
    """Required fields for ModelCacheSpec (internal base)."""

    model_id: str  # e.g. "meta-llama/Llama-3.1-8B-Instruct"
    model_family: str  # e.g. "llama"
    n_layers: int  # total transformer layers
    n_heads: int  # query heads (may differ from KV heads in GQA)
    n_kv_heads: int  # key/value heads (GQA-aware)
    head_dim: int  # dimension per head
    rope_enabled: bool  # whether model uses RoPE
    extraction_layers: tuple[int, ...]  # layers for EGR state extraction (D3)


class ModelCacheSpec(_ModelCacheSpecRequired, total=False):
    """Architecture-agnostic specification of a model's KV cache layout.

    Used to validate .eng files and ensure correct tensor shapes.

    For standard models (Llama, Phi, Qwen, Mistral):
        n_kv_heads and head_dim describe the single uniform KV cache.
        cache_sections is absent.

    For ISWA models (Gemma 4):
        cache_sections lists per-section dimensions. Each section has its
        own n_layers, n_kv_heads, and head_dim. The top-level n_kv_heads
        and head_dim reflect the dominant (largest) section.
        The state blob contains multiple sequential KV streams.
    """

    cache_sections: tuple[CacheSection, ...]


class EngramMetadata(TypedDict, total=False):
    """Metadata stored in .eng file header (safetensors __metadata__).

    All values are strings per safetensors spec (D7).
    Optional fields use total=False.
    """

    # Required
    engram_version: str
    cache_id: str
    compression: str  # CompressionMethod value
    model_id: str
    model_family: str
    n_layers: str  # stringified int
    n_heads: str
    n_kv_heads: str
    head_dim: str
    context_len: str
    agent_id: str
    task_description: str
    created_at: str  # ISO 8601

    # Optional
    expires_at: str
    parent_cache_id: str
    delta_from: str
    token_hash: str  # SHA-256 of input tokens
    state_vec_norm: str  # L2 norm of state vector (D4: stored as metadata)
    extraction_mode: str  # StateExtractionMode value
    block_index: str  # block position within a multi-block cache
    total_blocks: str


class CacheSearchResult(TypedDict):
    """Result from EGR manifold search over cached engram states."""

    cache_id: str
    similarity: float  # raw inner product score (not normalized — D4)
    task_description: str
    model_id: str
    created_at: str
    context_len: int


class CacheStats(TypedDict):
    """Aggregate statistics for the engram store."""

    total_entries: int
    total_size_bytes: int
    avg_compression_ratio: float
    model_breakdown: dict[str, int]  # model_family → count
