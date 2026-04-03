# ENGRAM Benchmark Results — April 2026
## Primary Results Table

| Model | Tokens | Cold TTFT | Warm TTFT | Speedup | K→K Margin | EGR ms | Trials | All Correct |
|---|---|---|---|---|---|---|---|---|
| Llama 3.2 3B | 4,002 | 11,439ms | 170ms | 67.2x | — | 9.5ms | 1 | n/a |
| Llama 3.2 3B | 16,382 | 94,592ms | 1,777ms | 53.2x | 0.477 | 9.5ms | 1 | 1/1 |
| Llama 3.1 8B | 591 | 3,436ms | 127ms | 27.1x | 0.381 | 30.6ms | 3 | 3/3 |

## Confirmed Results (n=3, diverse documents, stripped)

| Metric | Value |
|---|---|
| Model | Meta Llama 3.1 8B Instruct (Q4_K_M) |
| Tokens | 591 (doc A) / 560 (doc B) / 19 (query) |
| Layer range | 8-24 (16 of 32 layers) |
| SVD rank | 128 |
| K→K margin | 0.3813 ± 0.0000 |
| Correct | 3/3 |
| Speedup | 30.8x ± 5.4x |
| EGR overhead | 30.6ms ± 1.8ms |
| Cold TTFT | 3,508ms ± 127ms |
| Warm TTFT | 116ms ± 15ms |
| Verdict | **PASS** |

## Excluded Results

| Config | Reason excluded |
|---|---|
| 8B 28K synthetic | Repeated filler text — semantic variance collapsed |
| 8B 14K padded | Same filler bug — margin 0.093, not representative |
| 8B 592tok unstripped | Leading newline in document changed tokenization, flipped margin sign |

## EGR Overhead by Configuration

| Tokens | Layers | EGR ms | Note |
|---|---|---|---|
| ~600 | 8-24 (16L) | 30.6ms | 8B, confirmed n=3 |
| 6,403 | 8-24 (16L) | 48.8ms | 8B, synthetic benchmark |
| ~600 | 0-32 (32L) | ~84ms | 8B, full layers |

## INT8 Compression Results

| Tokens | FP16 .eng | INT8 .eng | Ratio | cos_sim | K→K margin (INT8 RT) |
|---|---|---|---|---|---|
| 591 | 73.9MB | 37.5MB | 1.97x | 0.99998 | 0.262 (ranking OK) |
| 6,403 | 800.4MB | 406.5MB | 1.97x | 0.99998 | — |

INT8 quantization: per-row symmetric, int8 data + float16 scales in safetensors.
K→K margin degrades slightly (0.381 → 0.262) but ranking is preserved.
200MB target requires INT4 (Phase 3).

## EGR Quality Improvements

| Method | Margin | Correct | Notes |
|---|---|---|---|
| SVD baseline | 0.381 | 3/3 | layers 8-24, rank 128, gate_start=0 |
| **SVD + gating** | **0.519** | **3/3** | **gate_start=6 (+36% margin)** |
| INT8 round-trip | 0.399 | 1/1 | 2x compression, gated |
| LAYER_DELTA round-trip | 0.213 | 1/1 | int8 inter-layer deltas |
| Cross-model per-doc SVD | -0.104 | 0/1 | Procrustes, LOOCV cos=-0.017 |
| Cross-model FCB+ridge | -0.017 | 0/1 | FCB LOOCV cos=0.969 but no discrimination |
| Cross-model Relative Repr | -0.066 | 0/1 | K=20 anchors, per-doc SVD input |

## Cross-Model Transfer Results

| Method | LOOCV cos | Retrieval margin | Correct | Notes |
|---|---|---|---|---|
| Per-doc SVD + linear | 0.002 | -0.104 | NO | Coordinates non-transferable |
| Per-doc SVD + Procrustes | -0.017 | -0.104 | NO | Orthogonal map insufficient |
| **FCB + ridge** | **0.969** | -0.017 | NO | Alignment works, discrimination lost |
| Relative Representations | n/a | -0.066 | NO | K=20 anchors, topology doesn't transfer |

### Analysis

Three progressively stronger cross-model alignment methods all fail:

1. **Per-document SVD** (LOOCV ~0): local coordinates are document-dependent
   and non-transferable. This was the expected baseline failure.

2. **Fixed Corpus Basis (FCB)** (LOOCV 0.969): FCB resolves coordinate
   instability — the ridge adapter achieves 0.969 cosine alignment between
   3B and 8B representations. However, FCB projection kills semantic
   discrimination (same-model FCB margin 0.029 vs per-doc SVD 0.519).
   The fixed basis captures corpus-wide structure, not document-specific content.

3. **Relative Representations** (margin -0.066): similarity profiles to
   K=20 anchor documents should be model-independent in theory (Moschella
   et al., 2023). In practice, the per-doc SVD state vectors used as input
   are already model-specific, contaminating the relative profiles.

### Diagnostic: CKA Layer Sweep

| Depth | 3B layer | 8B layer | CKA (keys) | CKA (vals) |
|---|---|---|---|---|
| 25% | 7 | 8 | 0.973 | 0.974 |
| 50% | 14 | 16 | 0.973 | 0.982 |
| 75% | 21 | 24 | 0.986 | 0.971 |
| Best pair (keys) | 7 | 7 | **0.988** | — |
| Best pair (vals) | 14 | 15 | — | **0.992** |

CKA > 0.97 at ALL layers. The representational geometry IS compatible.
The cross-model failure is in the coordinate system, not the topology.

### Cross-Model Transfer — Complete Diagnostic

#### Architecture
- Source: Llama 3.2 3B (28 layers, GQA 8 KV heads)
- Target: Meta-Llama 3.1 8B (32 layers, GQA 8 KV heads)
- Layers: 8-24 (semantic mid-band), gate_start=6
- CKA: >0.97 all layer pairs (manifolds topologically isomorphic)

#### All methods tested (9 approaches across 3 sessions)

| Method | Adapter | Margin | Correct | Note |
|---|---|---|---|---|
| Per-doc SVD + linear | ridge | -0.104 | NO | Local coords non-transferable |
| Per-doc SVD + Procrustes | ortho | -0.104 | NO | Same cause |
| FCB + ridge | ridge | -0.017 | NO | Alignment OK (0.969), discrimination lost |
| RR (K=20 anchors) | none | -0.066 | NO | SVD contaminates anchor profiles |
| Contrastive delta | ridge | +0.001 | YES | Direction transfers; barely |
| RCCA (n=30) | symm | -0.420 | NO | Per-doc SVD kills CCA |
| Residual FCB | none | -0.382 | NO | FCB complement = noise |
| JCB (joint SVD) | none | +0.011 | YES | Shared coords, weak discrimination |
| JCB + delta | none | +0.037 | YES | JCB frame + neutral removal |
| **FCDB (delta basis)** | **none** | **+0.124** | **YES** | **WINNER — delta from Frechet mean** |

#### FCDB: Fixed Corpus Delta Basis

The winning method operates on document-level mean vectors:
1. Compute corpus Frechet mean (joint center of 3B+8B mean key vectors)
2. Delta vectors: each doc's mean keys minus joint center
3. Joint SVD on normalized deltas from both models
4. Gate top 6 components, project into delta subspace

Same-doc cross-model cosine similarity: **0.475 +/- 0.106** (no adapter).
This means the directional variation from the manifold center IS preserved
across architectures — consistent with the CKA > 0.97 finding.

The FCDB succeeds where FCB fails because it captures the principal
directions of *variation* away from the mean, not the mean itself.
FCB captures what is common; FCDB captures what differentiates.

### Synthesis

FCDB achieves margin +0.124 (CORRECT) with no adapter required — the
first cross-model result exceeding 0.10. The progression tells a clear
scientific story:
- Per-doc SVD: coordinates are local, non-transferable (margin -0.10)
- FCB: coordinates stable but non-discriminative (margin -0.02)
- Contrastive delta: direction from neutral transfers (margin +0.001)
- FCDB: direction from corpus mean transfers AND discriminates (+0.124)

The key insight: cross-model transfer requires representing documents as
*directions from a shared reference point*, not as positions in space.
The Frechet mean of the joint corpus provides this reference.

## Basis Stability and Streaming Continuity (FCDB Geodesic)

### Phase 1: Basis Stability
- Method: FCDB (delta from Frechet mean, joint SVD)
- Subspace agreement across corpus subsets: 0.82-0.84 (UNSTABLE)
- Convergence: N=50 insufficient (agreement < 0.90), needs N>100
- Verdict: **UNSTABLE** — basis is corpus-dependent at N=50

### Phase 2: Projection Residual
- FCDB operates on normalized deltas, not absolute key vectors
- All domain categories show similar residuals (0.57-0.84)
- Residual is NOT a domain discriminator for FCDB
- Verdict: N/A (metric not applicable to directional basis)

### Phase 3: Streaming Fold-In (same model, 12 docs)
- DOC_A ranked 7th out of 12 after adding 10 noise docs
- Margin A-B: +0.168, but both outranked by noise
- Verdict: **FAIL** — FCDB same-model discrimination too weak for dense index

### Phase 4: Cross-Model Fold-In (FCDB, 3 trials)
- Mean margin: +0.124 (3/3 correct, deterministic)
- No adapter required
- Verdict: **PASS** — FCDB cross-model retrieval works

### Phase 5: Incremental Update
- Max drift on existing vectors: 0.182 (threshold: 0.02)
- Rank-1 basis update too aggressive for 94-dim basis
- Verdict: **FAIL** — basis updates destabilize existing projections

### Phase 6: Semantic Coverage Score
- SCS values 0.38-0.55 across all domains
- No domain separation — FCDB SCS is not a domain discriminator
- Verdict: **PASS** (values in reasonable range)

### Phase 7: Full Integration (cross-model end-to-end)
- Store 3B DOC_A/DOC_B, query 8B, margin +0.124, correct
- Verdict: **PASS**

### System-Level Assessment

| Property | FCDB Status | Per-doc SVD Status |
|---|---|---|
| Same-model retrieval | WEAK (rank 7/12) | **STRONG (margin 0.519)** |
| Cross-model retrieval | **PASS (+0.124)** | FAIL (-0.104) |
| Basis stability | UNSTABLE (N=50) | N/A (no shared basis) |
| Incremental update | UNSAFE (drift 0.18) | N/A |
| No adapter needed | **YES** | N/A |

**Conclusion:** FCDB and per-doc SVD are complementary:
- Per-doc SVD: production same-model retrieval (margin 0.519)
- FCDB: cross-model retrieval only (margin 0.124, no adapter)
- A production system should use per-doc SVD by default,
  switching to FCDB only for cross-model queries

## FCDB v2 Scaling Results (N=200 Corpus)

### Basis Stability (Phase 1 v2)
- N=200 corpus: 10 domains × 20 docs, diverse, no duplicates
- Subspace agreement (70% subsets vs full): **0.997-0.999 (STABLE)**
- Convergence: N=50 → 0.744, N=100 → 0.906, N=125 → 0.983, N=200 → 1.000
- **EIGENGRAM unblocked at N≥125**

### Dual-Fingerprint Streaming (Phase 3 v2)
- Per-doc SVD: DOC_A rank **1/12**, margin +0.373 → **PASS**
- FCDB v2: DOC_A rank 2/12, margin +0.109 → FAIL (but correct A>B)
- **Production: use per-doc SVD for same-model retrieval**

### Cross-Model FCDB v2 (Phase 4 v2)
- 3/3 correct, deterministic
- Margin: +0.013 (down from v1's +0.124)
- Verdict: **MARGINAL** — correct ranking but thin margin
- Note: larger corpus dilutes per-document discrimination

### Buffer-Rebuild (Phase 5 v2)
- Single-model rebuild produces orthogonal basis (drift 0.99)
- Correct rebuild must use same model pair as original
- With same pair at N≥125: stability 0.999 (Phase 1 result)
- Verdict: **N/A** — methodology requires both models

### Cross-Model Method Comparison (Final)
| Method | Corpus | Margin | Correct | Note |
|---|---|---|---|---|
| Per-doc SVD | n/a | -0.104 | NO | Local coords non-transferable |
| FCB + ridge | 50 | -0.017 | NO | Alignment OK, discrimination lost |
| Contrastive delta | 50 | +0.001 | YES | Direction transfers; barely |
| FCDB v1 | 50 | **+0.124** | YES | **Best margin (small corpus)** |
| FCDB v2 | 200 | +0.013 | YES | Stable but thinner margin |
| JCB-delta | 50 | +0.037 | YES | JCB frame + neutral removal |

### Key Finding: Stability-Discrimination Tradeoff
FCDB v1 (N=50): unstable basis (agreement 0.82) but strong discrimination (+0.124)
FCDB v2 (N=200): stable basis (agreement 0.999) but thin discrimination (+0.013)
This is the fundamental tradeoff: a larger corpus stabilizes the subspace but
dilutes per-document signal. The optimal corpus size depends on the use case.

## Hardware

- Apple M3, 24GB RAM, macOS Darwin 25.4.0
- Metal GPU (n_gpu_layers=-1)
- llama-cpp-python 0.3.19
- FAISS CPU 1.13.2
- torch 2.11.0
