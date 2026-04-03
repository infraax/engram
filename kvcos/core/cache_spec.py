"""
ENGRAM Protocol — Model Architecture Registry


Contains ModelCacheSpec definitions for known models and utilities
to look up specs by model_id or infer model family from string.

D3: extraction_layers set to middle-to-deep (8-31 for 32-layer models)
per ShadowKV validation. Early layers (0-7) and final layer preserved.
"""

from __future__ import annotations

from kvcos.core.types import AttentionType, CacheSection, ModelCacheSpec

# ── Pre-registered Model Specs ────────────────────────────────────────────────

# Llama 3.1 8B — Primary Phase 1 target (D1, D6)
# GQA: 32 query heads, 8 KV heads, head_dim 128
LLAMA_3_1_8B = ModelCacheSpec(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    model_family="llama",
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    rope_enabled=True,
    extraction_layers=tuple(range(8, 32)),  # layers 8-31 (D3)
)

# Llama 3.1 8B base (non-instruct)
LLAMA_3_1_8B_BASE = ModelCacheSpec(
    model_id="meta-llama/Llama-3.1-8B",
    model_family="llama",
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    rope_enabled=True,
    extraction_layers=tuple(range(8, 32)),
)

# Phi-3-Mini-128K — Secondary Phase 1 target
# ShadowKV validated SVD on this model (D3)
# MHA: 32 query heads, 32 KV heads (no GQA), head_dim 96
PHI_3_MINI = ModelCacheSpec(
    model_id="microsoft/Phi-3-mini-128k-instruct",
    model_family="phi",
    n_layers=32,
    n_heads=32,
    n_kv_heads=32,  # Phi-3-Mini uses MHA, not GQA
    head_dim=96,
    rope_enabled=True,
    extraction_layers=tuple(range(8, 32)),
)

# Gemma 2 2B — NOTE: QK-Norm model, SVD behavior may differ (T3 caveat)
GEMMA_2_2B = ModelCacheSpec(
    model_id="google/gemma-2-2b-it",
    model_family="gemma",
    n_layers=26,
    n_heads=8,
    n_kv_heads=4,
    head_dim=256,
    rope_enabled=True,
    extraction_layers=tuple(range(6, 26)),
)

# Qwen 2.5 7B
QWEN_2_5_7B = ModelCacheSpec(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    model_family="qwen",
    n_layers=28,
    n_heads=28,
    n_kv_heads=4,
    head_dim=128,
    rope_enabled=True,
    extraction_layers=tuple(range(7, 28)),
)

# Mistral 7B v0.3
MISTRAL_7B = ModelCacheSpec(
    model_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_family="mistral",
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    rope_enabled=True,
    extraction_layers=tuple(range(8, 32)),
)


# Gemma 4 26B-A4B — ISWA model (Interleaved Sliding Window Attention)
# Dual KV cache: Global (full context) + SWA (sliding window 1024 tokens)
# MoE: 128 experts, 8 active — does NOT affect KV cache (FFN-only)
# Reverse-engineered from llama.cpp b5200+ state blob format.
GEMMA_4_26B_A4B = ModelCacheSpec(
    model_id="google/gemma-4-26b-a4b-it",
    model_family="gemma",
    n_layers=30,   # total: 5 global + 25 SWA
    n_heads=32,
    n_kv_heads=8,  # dominant section (SWA)
    head_dim=256,  # dominant section (SWA)
    rope_enabled=True,
    extraction_layers=tuple(range(8, 30)),
    cache_sections=(
        CacheSection(
            attention_type=AttentionType.FULL,
            n_layers=5,
            n_kv_heads=2,
            head_dim=512,
        ),
        CacheSection(
            attention_type=AttentionType.SLIDING,
            n_layers=25,
            n_kv_heads=8,
            head_dim=256,
            window_size=1024,
        ),
    ),
)


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, ModelCacheSpec] = {
    spec["model_id"]: spec
    for spec in [
        LLAMA_3_1_8B,
        LLAMA_3_1_8B_BASE,
        PHI_3_MINI,
        GEMMA_2_2B,
        GEMMA_4_26B_A4B,
        QWEN_2_5_7B,
        MISTRAL_7B,
    ]
}

_FAMILY_MAP: dict[str, str] = {
    "llama": "llama",
    "meta-llama": "llama",
    "phi": "phi",
    "microsoft/phi": "phi",
    "gemma": "gemma",
    "google/gemma": "gemma",
    "qwen": "qwen",
    "mistral": "mistral",
    "deepseek": "deepseek",
}


def get_model_spec(model_id: str) -> ModelCacheSpec | None:
    """Look up a ModelCacheSpec by exact model_id."""
    return _REGISTRY.get(model_id)


def register_model_spec(spec: ModelCacheSpec) -> None:
    """Register a new model spec in the runtime registry."""
    _REGISTRY[spec["model_id"]] = spec


def infer_model_family(model_id: str) -> str:
    """Infer model family from a model_id string."""
    model_id_lower = model_id.lower()
    for prefix, family in _FAMILY_MAP.items():
        if prefix in model_id_lower:
            return family
    return "unknown"


def make_spec_from_metadata(
    model_id: str,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rope_enabled: bool = True,
) -> ModelCacheSpec:
    """Create a ModelCacheSpec from raw parameters.

    Automatically sets extraction_layers to middle-to-deep range (D3).
    """
    skip_layers = max(1, n_layers // 4)
    extraction_layers = tuple(range(skip_layers, n_layers))

    return ModelCacheSpec(
        model_id=model_id,
        model_family=infer_model_family(model_id),
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        rope_enabled=rope_enabled,
        extraction_layers=extraction_layers,
    )


def is_iswa_spec(spec: ModelCacheSpec) -> bool:
    """Check if a model spec describes an ISWA (multi-section) cache."""
    return "cache_sections" in spec


def validate_kv_shape(
    spec: ModelCacheSpec,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
) -> bool:
    """Validate that KV tensor dimensions match the model spec."""
    return (
        spec["n_layers"] == n_layers
        and spec["n_kv_heads"] == n_kv_heads
        and spec["head_dim"] == head_dim
    )
