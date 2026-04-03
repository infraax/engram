# ENGRAM Protocol — Technical Disclosure
# For Attorney Review — Provisional Patent Application

## Filing metadata
Date: 2026-04-01
Inventors: [FILL — legal names]
Title: Engrammatic Geometry Retrieval: Persistent Cross-Session
       AI Cognitive State Storage via Pre-RoPE Key Manifold Similarity

## Three independent claims

### Claim 1 — The .eng file format
A portable serialization format for transformer KV cache state.
Novel elements:
  - Safetensors tensor storage indexed by layer
  - JSON metadata header with agent_id, task_description,
    context_len, l2_norm, cache_id (SHA-256 of token sequence)
  - Block-addressed layout: one file per 256-token context block
  - Delta chain pointers for incremental state extension
  - INT8 quantized storage with per-row scales for 2x compression

### Claim 2 — The EGR retrieval method
A method for semantic retrieval of stored KV cache states using
pre-RoPE key manifold similarity. Novel elements:
  - Extracts pre-RoPE key tensors from middle layers (8-24 of 32)
  - Computes state vector via truncated SVD projection (rank 128)
    with row subsampling (8192 of N rows) for O(1) extraction time
  - Indexes state vectors via MIPS (FAISS IndexFlatIP)
  - Retrieves by K→K inner product (not Q→K, not text embedding)
  - Injects retrieved state via llama_state_get_data() equivalent
  - Discriminates topically distinct cached states with margin 0.38
    using only the geometric structure of key matrices

### Claim 3 — The combined ENGRAM system
A system combining claims 1 and 2 with agent-native API,
cross-session persistence, and cross-model transfer capability.

## Measured results (hardware: Apple M3, 24GB, Metal GPU)
- TTFT speedup: 27–99x across 591–14K token contexts
- K→K discrimination margin: 0.381 ± 0.000 (3/3 trials, 8B model)
- EGR overhead: 30.6ms ± 1.8ms (layers 8-24, FAISS search)
- INT8 compression: 2x on-disk reduction, cos_sim=0.99998
- INT8 round-trip margin: 0.262 (ranking preserved)
- State blob: 800MB at 14K fp16, 407MB INT8 (target: <200MB with INT4)

## Prior art distinctions
- LMCache: token-hash prefix only, no semantic retrieval
- agent-memory (arXiv:2603.04428): no semantic retrieval,
  no cross-model transfer, no open format
- MemArt (ICLR 2026): latent-space concept only, no implementation,
  no cross-session persistence, no .eng format
- TurboRAG/FusionRAG: document precompute, not session state

Novel combination: open format + K→K manifold retrieval +
agent-native API + cross-session persistence.
No prior art covers all four elements in combination.

## Source evidence
Repository: github.com/infraax/ENGRAM (private)
Proof script: scripts/egr_semantic_proof.py
Results: results/egr_semantic_proof_8B_14K.json
INT8 proof: results/egr_semantic_proof_8B_INT8.json
