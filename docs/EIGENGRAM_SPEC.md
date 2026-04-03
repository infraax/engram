# EIGENGRAM Format Specification
## EGR1 Version 1.0

An EIGENGRAM (`.eng`) file is a self-contained semantic certificate
for a KV-cache document. It encodes the geometric fingerprint of
a text document as processed by a local LLM, enabling retrieval
without re-running the model and without a vector database.

---

## Binary Layout

All multi-byte integers are **little-endian**. Floats are IEEE 754.

### Fixed Header (99 bytes)

| Offset | Size | Type | Field | Notes |
|---|---|---|---|---|
| 0 | 4 | bytes | `magic` | Always `b"EGR1"` |
| 4 | 1 | uint8 | `version` | Currently `1` |
| 5 | 32 | ASCII | `corpus_hash` | SHA256[:32] of basis file |
| 37 | 20 | ASCII | `created_at` | `YYYY-MM-DDTHH:MM:SS` UTC |
| 57 | 16 | ASCII | `model_id` | Null-padded |
| 73 | 2 | uint16 | `basis_rank` | R (116) |
| 75 | 2 | uint16 | `n_corpus` | Documents in basis (200) |
| 77 | 2 | int8x2 | `layer_range` | `[8, 24]` |
| 79 | 4 | uint32 | `context_len` | KV rows processed |
| 83 | 4 | float32 | `l2_norm` | L2 norm of mean key vector |
| 87 | 4 | float32 | `scs` | Semantic Coverage Score 0-1 |
| 91 | 4 | float32 | `margin_proof` | Retrieval margin at write time |
| 95 | 2 | uint16 | `td_len` | Byte length of task_description |
| 97 | 2 | uint16 | `ci_len` | Byte length of cache_id |

### Variable Section

| Offset | Size | Type | Field |
|---|---|---|---|
| 99 | R x 2 | float16[] | `vec_perdoc` |
| 99+Rx2 | R x 2 | float16[] | `vec_fcdb` |
| 99+2Rx2 | 256 | float16[] | `joint_center` (128 x fp16) |
| +256 | td_len | UTF-8 | `task_description` |
| +td_len | ci_len | UTF-8 | `cache_id` |

**Total size for R=116:** ~856 bytes

---

## Dual Fingerprints

### `vec_perdoc` -- per-doc SVD fingerprint
Best for same-model retrieval. Margin ~0.37.
Model-specific: queries must use the same model family.

### `vec_fcdb` -- FCDB fingerprint
Best for cross-model retrieval. Margin ~0.013.
Model-agnostic: any model sharing the basis can query.

### `joint_center`
Embedded in the file so readers can fold-in new queries
without loading the full basis file.

---

## Fingerprint Selection Guide

| Scenario | Use | Expected margin |
|---|---|---|
| Same model wrote and queries | `perdoc` | ~0.37 |
| Different models (e.g. 3B wrote, 8B queries) | `fcdb` | ~0.013 |
| Unknown source model | `fcdb` | ~0.013 |

---

## SCS Interpretation

| SCS range | Interpretation |
|---|---|
| > 0.50 | Well-represented by corpus |
| 0.15-0.50 | Partially represented |
| < 0.15 | Out-of-corpus |

---

## Compatibility

- Readers MUST reject `magic != "EGR1"`
- Readers MUST reject `version > EIGENGRAM_VERSION`
- `corpus_hash` mismatch is advisory, not blocking

## Basis

FCDB v2 basis trained on Llama-3.2-3B + Llama-3.1-8B,
200 documents, 10 domains, layers 8-23. Stability 0.999.

## Reference Implementation

- `kvcos/engram/format.py` -- EigramEncoder / EigramDecoder
- `kvcos/engram/writer.py` -- write_eigengram()
- `kvcos/engram/reader.py` -- read_eigengram(), load_eigengram_index()
- `kvcos/engram/__main__.py` -- CLI
- `tests/test_eigengram.py` -- 20 tests

## Proof of Concept (April 2026)

| Metric | Value |
|---|---|
| File size (R=116) | ~856 bytes |
| Cross-model (3B->8B) | PASS, margin +0.013 |
| Same-model (8B->8B) | PASS, margin +0.373 |
| Basis stability (N=200) | 0.997-0.999 |
| Format tests | 20/20 pass |
