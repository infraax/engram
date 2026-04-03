# ENGRAM Multi-Architecture Support

How ENGRAM handles different model architectures for KV cache fingerprinting.

## Architecture Types

### Standard (Single KV Cache)

Models with one uniform KV cache across all layers.

| Model | Layers | KV Heads | Head Dim | FP Dim |
|-------|--------|----------|----------|--------|
| Llama 3.1 8B | 32 | 8 | 128 | 2048 |
| Phi-3-Mini 128K | 32 | 32 | 96 | 6144 |
| Gemma 2 2B | 26 | 4 | 256 | 2048 |
| Qwen 2.5 7B | 28 | 4 | 128 | 1024 |
| Mistral 7B | 32 | 8 | 128 | 2048 |

Fingerprint dimension: `n_kv_heads * head_dim * len(freqs)` (default freqs=[0,1]).

### ISWA (Interleaved Sliding Window Attention)

Models with multiple KV cache sections, each with different dimensions.
Used by Gemma 4 and similar architectures.

| Model | Section | Type | Layers | KV Heads | Head Dim | Section FP |
|-------|---------|------|--------|----------|----------|------------|
| Gemma 4 26B-A4B | 0 (Global) | Full | 5 | 2 | 512 | 2048 |
| Gemma 4 26B-A4B | 1 (SWA) | Sliding | 25 | 8 | 256 | 4096 |
| **Total** | | | **30** | | | **6144** |

ISWA total FP = sum of per-section fingerprints.

### MoE (Mixture of Experts)

MoE routing only affects FFN layers, **not** the attention mechanism.
KV cache structure is identical to standard models — no special handling needed.
Gemma 4's 128/8 MoE configuration is transparent to ENGRAM.

## Blob Format

### Standard (n_stream = 1)

```
┌──────────────────────────┐
│ u32 arch_len             │
│ str arch (e.g. "llama")  │
│ u32 n_stream = 1         │
├──────────────────────────┤
│ u32 cell_count           │
│ cell metadata × N        │
│ u32 v_trans              │
│ u32 n_layers             │
│ K layers × n_layers      │
│ V layers × n_layers      │
└──────────────────────────┘
```

### ISWA (n_stream > 1)

```
┌──────────────────────────┐
│ u32 arch_len             │
│ str arch (e.g. "gemma4") │
│ u32 n_stream = 2         │
├──── Stream 0 (Global) ───┤
│ u32 cell_count           │
│ cell metadata × N        │
│ u32 v_trans              │
│ u32 n_layers = 5         │
│ K layers × 5             │
│ V layers × 5             │
├──── Stream 1 (SWA) ──────┤
│ u32 cell_count           │
│ cell metadata × N        │
│ u32 v_trans              │
│ u32 n_layers = 25        │
│ K layers × 25            │
│ V layers × 25            │
└──────────────────────────┘
```

Each stream has identical internal structure.
The `n_stream` field distinguishes standard from ISWA blobs.

## Type System

```
CacheSection (frozen dataclass)
├── attention_type: AttentionType (FULL | SLIDING)
├── n_layers: int
├── n_kv_heads: int
├── head_dim: int
├── window_size: int | None
└── n_embd_kv: property (n_kv_heads * head_dim)

ModelCacheSpec (TypedDict)
├── model_id, model_family, n_layers, ...  (required)
└── cache_sections: tuple[CacheSection, ...]  (optional, ISWA only)
```

Standard models: `cache_sections` absent.
ISWA models: `cache_sections` lists per-section specs.

## Fingerprint Strategy

### Standard Pipeline

```
text → LlamaCppBridge → KV cache → parse_state_blob()
     → keys[layers, heads, ctx, dim] → mean(ctx)
     → Fourier DFT (f0 + f1) → L2-normalize → 2048-dim
```

### ISWA Pipeline (Strategy A: Per-Section Concatenation)

```
text → LlamaCppBridge → KV cache → parse_multi_section_blob()
     → [ParsedKVCache_0, ParsedKVCache_1]
     ├── Section 0: keys → mean → Fourier → normalize → 2048-dim
     └── Section 1: keys → mean → Fourier → normalize → 4096-dim
     → concatenate → 6144-dim
```

Each section's sub-fingerprint is independently L2-normalized,
preserving the relative geometry within each attention type.

## Adding a New Architecture

1. **types.py**: If the model has a non-standard KV cache (e.g., multi-section),
   define the `CacheSection` entries.

2. **cache_spec.py**: Add a `ModelCacheSpec` to the registry.
   For standard models, just set the required fields.
   For ISWA models, include `cache_sections`.

3. **llama_cpp_bridge.py**: The `_meta_get()` helper automatically tries
   architecture-specific metadata prefixes. Add new prefixes to
   `_METADATA_PREFIXES` if needed.

4. **blob_parser.py**: Standard models work automatically.
   Multi-section models use `parse_multi_section_blob()`.

5. **Tests**: Use `make_synthetic_iswa_blob()` from conftest.py to create
   test blobs without needing a real GGUF model.

## Key Functions

| Function | File | Purpose |
|----------|------|---------|
| `is_iswa_spec()` | cache_spec.py | Check if spec has cache_sections |
| `parse_multi_section_blob()` | blob_parser.py | Parse ISWA blobs |
| `compute_iswa_fingerprint()` | state_extractor.py | Per-section FP concat |
| `_meta_get()` | llama_cpp_bridge.py | Multi-arch metadata lookup |
| `make_synthetic_iswa_blob()` | conftest.py | Test blob builder |

## Reverse Engineering Notes

Gemma 4 26B-A4B format reverse-engineered from:
- llama-cpp-python 0.3.20 (llama.cpp b5200+)
- Model: bartowski/google_gemma-4-26B-A4B-it-IQ4_XS.gguf
- Metadata prefix: `gemma4.*`
- Architecture string: `"gemma4"`
- V transposed: yes (all sections)
- All KV data: F16

Key finding: MoE (128 experts, 8 active) is FFN-only and transparent to KV cache.
The ISWA dual-cache with heterogeneous KV dimensions is the only compatibility issue.
