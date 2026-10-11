# OpenAI Math Release × ENGRAM-A — Cross-Domain Map v0.2 (Agent / KV-memory lens)
### Re-run of v0.1 against the original ENGRAM idea: KV-cache fingerprinting for persistent cross-session LLM memory, upgraded with .eng v2 / GFR v2
### 2026-10-11 · Dex × Claude · status: abstract-level screen; deep-reads pending · companion to `05_AGENT_ENGRAM/ENGRAM_AGENT_EVOLUTION_PLAN_v0.1.md`

---

## 0. What changed versus v0.1

- **Lens.** v0.1 mapped against the embodied creature (RCL field, soma, dock cycle). This pass maps against **ENGRAM-A**:
  - K-geometry fingerprints of an LLM's KV cache (per-layer mean keys → rFFT over the layer axis → f0+f1; per-doc SVD with `gate_start`)
  - `.eng` certificates
  - GFR-A retrieval
  - restoring KV state into the engine (llama.cpp / MLX / LMCache)
  - multi-agent, session-to-session memory
- **Same corpus:** 372 families / 719 papers. Same index: `04_MATH_CROSSMAP/openai_math_index.tsv`. All 372 summaries re-screened; 15 extra families pulled for abstracts.
- **Net effect:**
  - **Up:** online caching/admission (110, 111), sequence/edit geometry (099, 121, 128), clustering limits (117, 125), summarisation chains (229), cross-model geometry (322, 324, 374).
  - **Down:** everything tied to the continuous field (325, 149, 261, 143, 213/214).

**Tags:**
- **Trust:** `LEAN` (formally verified) · `CERT` (computer certificate) · `—` (unverified).
- **Verdict:** `PROVES` · `CONSTRAINS` · `IMPROVES` · `DISPROVES` · `NULL`.

---

## 1. Tier A — direct contact with ENGRAM-A

| # | Result | Trust | ENGRAM-A touchpoint | What it gives us | Verdict |
|---|---|---|---|---|---|
| **222 + 234** | Perceptron / spin-glass free energies for **(bi-)orthogonally invariant** disorder with arbitrary spectrum, **"no outliers"** | — | K→K index capacity; `gate_start=6`; FCDB centering | LLM key matrices are famously **not** outlier-free: massive-activation channels plus a large shared mean. `gate_start` (skip the top 6 singular vectors) and FCDB (subtract the Fréchet mean) are both, in effect, **outlier removal**. These theorems say why that was necessary: after outlier removal the spectrum is in a class where capacity is *computable*. | **CONSTRAINS + IMPROVES** — gives the theory behind two empirical v1 tricks, and a route to a predicted capacity per model. |
| **219** (+ external BBP / Marchenko–Pastur) | GOE bulk universality, random regular graphs | — | Choosing `gate_start` | Random-matrix bulk vs outlier ("spike") separation gives a **principled gate**: skip exactly the singular values above the noise-bulk edge, per model and per layer band. This replaces the hand-tuned constant 6 (tuned on Llama-8B only). | **IMPROVES** — per-model gate without tuning. |
| **110** | Randomized **k-server** O(log² k), every metric | — | KV tiering: GPU ↔ CPU ↔ disk ↔ remote | Each KV state is 74–800 MB, so tier placement is the dominant runtime cost. k-server is paging with movement costs in a metric. With a semantic metric (fingerprint distance) plus a tier-cost metric, you get a **competitive guarantee for prefetch/evict**. LMCache uses recency/frequency-style policies. Use the structure; the computable constants are huge. | **IMPROVES** — a semantic eviction/prefetch policy with a theory anchor. |
| **111** | One-sample **matroid prophet inequality** | — | Admission: which states get certified **and** stored | Far more important for agents than for the robot: storing a 14k-token state costs ~400–800 MB. Fixed storage budget = matroid constraint. One past sample per slot is enough for a constant-competitive online accept/reject. | **IMPROVES** — the admission gate (S1 / write budget) gets a baseline policy. |
| **229** | Exact Kesten–Stigum thresholds + capacity criterion on trees | CERT | Recursive summarisation / compaction (summary of summaries); DISTORTION FULL→SUMMARY→GIST; session-chain certificates | Information from the leaves survives a tree of noisy copies **iff branching × fidelity² > 1**. Recursive summarisation (MemGPT-style compaction, session-of-sessions) is exactly such a tree. | **PROVES (by analogy) a design rule:** each compaction layer needs fan-in × (retained correlation)² > 1, or old sessions become provably unrecoverable at the top. A testable kill criterion. |
| **099 + 121** | Edit distance → ℓ1 needs distortion exp(Θ(√(log d·log log d))); almost-linear (1+ε) edit-distance approximation | — | W2 ("leading newline flips margin"), restore-hit rate, CacheBlend-style recompute set | (a) **No fixed-dimension fingerprint can faithfully preserve token-sequence edit similarity**, so a fingerprint alone will always be fragile to small edits. (b) Near-linear edit-distance alignment between a new prompt and a cached block's token sequence is cheap. It says *which* tokens differ, i.e. the selective-recompute set for restoring a near-match. | **CONSTRAINS + IMPROVES** — keep a token-level path (hash + alignment) beside the fingerprint. Fingerprint for *finding*, alignment for *reusing*. |
| **125** | Metric **k-median** threshold 1+2/e, plus recovery from an anchor solution | one **LEAN** | Consolidation: pick k prototype states (warm set, merged sessions); incremental re-clustering | Best possible polynomial approximation is 1+2/e, and it's achieved. "Recovery from an anchor solution" fits incremental consolidation: repair yesterday's prototypes instead of re-clustering from scratch. | **PROVES + CONSTRAINS** — the optimum for prototype selection, and don't chase better. |
| **128** | Poly-time **2-approximation for shortest common superstring** | **LEAN** | Assembling several retrieved chunks into **one** restore sequence | Ordering retrieved cached chunks to maximise token overlap (shared prefixes → reusable KV) is a superstring problem. Provable factor 2. | **IMPROVES** — principled multi-chunk restore ordering. |
| **117** | Uniform sparsest cut: NP-hard to approximate within any constant; ~√log n SDP gap | one **LEAN** | Topic partitioning of the memory/link graph; "auto-folders" | Optimal global topic partitioning of a memory graph is intractable even approximately. | **CONSTRAINS** — don't build features that need optimal clustering; use local/greedy structure (facets, Mode B links). |
| **140** | Memory–sample lower bounds for streaming estimation | **LEAN ×5** | Certificate size; "what can 856 bytes encode about a state" | A learner keeping < d² bits needs Ω(d·log 1/ε) observations to pin a direction. Per head (d=128) d² = 16 kbit ≈ 2 KB, and an EGR1 certificate (~856 B) sits **below** it. A certificate is a *pointer + coarse direction*, never a substitute for the KV payload. | **CONSTRAINS** — formal backing for "certificate locates, payload restores". Sizing rule for fingerprint dims. |
| **322 + 324 + 374 / 360** | Tingley (sphere isometries extend to linear isometries); Lipschitz-equivalent ≠ linearly isomorphic; Brenier-map stability exponent 1/3 (sharp); MTW-regular OT on manifolds | 322 **LEAN**, 324 **LEAN**, 360 **LEAN** | W4 cross-model transfer (9 methods, best +0.013 at N=200) | (a) A map preserving all cosine distances between two models' normalised fingerprint spheres **must** be linear (orthogonal). So Procrustes failing ⇒ the spheres are **not isometric** ⇒ no distance-preserving map exists. (b) CKA > 0.97 with linear adapters failing is consistent with "geometrically equivalent, not linearly mappable" (324 as analogy: infinite-dimensional statement). (c) OT-based alignment is only W₂^(1/3)-stable, so noise gets amplified. | **DISPROVES** the hope for a cheap linear cross-model map; **CONSTRAINS** OT alternatives. → Park cross-model; separate spaces per model (`space_key`). |
| **076 + 179** (+ external SRHT) | Ultraflat ±1 polynomials; circulant Hadamard only at orders 1, 4 | 179 **LEAN**, 076 one **LEAN** | Projecting 2048–6144-d fingerprints; binary prefilter codes | ±1 sequences with near-flat spectrum exist at every length; exactly flat ±1 circulants don't. This is the math behind **subsampled randomised Hadamard transforms** (O(d log d) JL projections) and good binary codes. | **IMPROVES** — fast structured projection and binary codes for the index. |
| **119** | Courtade–Kumar: under noise, a single coordinate is the most informative bit | **LEAN** | Binary/1-bit prefilter (SimHash-style) | Per-bit, a sign-of-one-projection bit maximises mutual information under bit-flip noise. Supports SimHash-style codes over clever learned bit-mixtures. | **IMPROVES (minor)** — Hamming prefilter design. |

---

## 2. Tier B — strong analogies (need mapping work)

| # | Result | Trust | Hypothetically possible mapping |
|---|---|---|---|
| **227** | SK dynamics: fast mixing below β=1, critical slowing at β=1, glassy above | one **LEAN** | GFR L3 softmax retrieval (Hopfield = attention) has a temperature transition. β too high → retrieval gets stuck on spurious attractors; too low → everything blends. Calibrate β against the empirical score spread per space. |
| **089** | Bounded-treewidth/planar → O(1) multicommodity flow–cut gap | **LEAN** | GFR-A Physarum aggregation over many agents' queries: keep the link/flow graph low-treewidth (local links) to inherit a guarantee. |
| **149** | Permanence of weakly reversible mass-action systems | — | GFR-A L2 conductance as a weakly reversible network → no edge decays to 0, nothing runs away, by theorem. |
| **174 + 131** | Strong thin trees; switch-chain uniform sampler with fixed degrees | 174 **LEAN** | Multi-agent sync: share a thin backbone of the link graph instead of everything. Null models (same degree sequence) to test whether learned conductance is real structure. |
| **127** | Gotsman–Linial: degree-d PTF average sensitivity ≤ 8d√n | **LEAN** | The accept gate (cos ≥ τ on quantised fingerprints) is a low-degree threshold function. Bound how many decisions flip under INT8/INT4/binary quantisation noise; useful for compression choices (the v1 INT8 margin went 0.381 → 0.262). |
| **093** (+ external EVT) | Dimension-free log-Sobolev for subgaussian log-concave measures | — | Concentration → the max impostor cosine among N stored states grows like √(2 log N / d_eff). Predicts margin decay **with a finite collapse point** N\*. Contradicts the v1 abstract's "no collapse point" (E-06). |
| **098** | A doubling subset of Hilbert space with no bi-Lipschitz embedding in any finite-dimensional normed space | **LEAN** | Fundamental limit: no fixed-dimension fingerprint can preserve *all* distance structure of a rich state space with bounded distortion. Fingerprints are lossy *indices*, never isometric copies. Same conclusion as 099 from geometry. |
| **265** | 2D gapped area law → polynomial PEPS (tensor-network) compression | — | KV caches are empirically low-rank (ShadowKV, xKV). "Locally correlated state ⇒ compressible by a tensor network" suggests tensor-network compression of the payload. Hypothesis only. |
| **137** | One-tape time T simulable in T^(2/5) space | **LEAN** | The KV cache *is* a time-vs-space trade (store vs recompute). TCS says space can be traded sublinearly against time. Framing for the "store / recompute / partially recompute" policy; no direct algorithm. |
| **332** | ℓ1 metric Markov cotype 2 → Lipschitz extension from Hilbert subsets into ℓ1 | — | Anchor-based adapters (relative representations used K=20 anchors): maps defined on anchors extend to the whole space with bounded loss into ℓ1. A possible cross-model route if cross-model is ever un-parked. |
| **097** | Euclidean Steinitz: signing keeps prefix sums ≤ C√d | **LEAN** | The fingerprint is a running mean of keys over tokens. Prefix-sum control is relevant to incremental fingerprints as the context grows (stability of the running mean). |
| **090 / 092** | Fourier certificates for packing; covering density Θ(n log n) | CERT / one **LEAN** | Spherical-code style bounds on how many states can sit at a given separation → index capacity at a target margin. |
| **120** | Almost-linear exact matching | — | Assign query segments ↔ cached blocks when restoring several blocks into one prompt. |
| **266** | MUB theory | CERT | The two senses (K-geometry vs text-embedding) as "mutually unbiased" spaces: weak. |

---

## 3. Tier C — NULL for ENGRAM-A

| # | Why |
|---|---|
| **075** L log L | The fingerprint keeps f0+f1 of a 16–32-point DFT over layers; asymptotic partial-sum convergence says nothing at that size. |
| **077 / 074** restriction, Kakeya | No operational handle. |
| **130** FFT below n log n · **107** ω ≤ 9/4 | Zero practical effect (savings like (log n)^(10⁻¹³)); attention cost is memory-bound, not ω-bound. |
| **325** Crouzeix · **261** Anderson · **143** Hilbert-16 · **213/214** percolation | Tied to the continuous creature field; no field in ENGRAM-A v2. |
| **328 · 245 · 103** | Finite-dim trivial / type-system specific / determinism already by seeding. |
| **238** Thorp/partial permutations · **145/154** mixing | No agent mechanism they constrain. |
| **221 · 217 · 281 · 239 · 101 · 096 · 115 · 139 · 094 · 186 · 113** | Out of scope for this design. |

---

## 4. Hypothetically possible — cross-domain ideas from outside the repo (agent/KV-specific)

1. **Attention sinks & massive activations** (StreamingLLM; "Massive Activations in LLMs", 2024). The first token(s) and a few channels carry huge-norm keys. A per-layer **mean over tokens is dominated by the sink**, which plausibly explains (a) the high cross-document correlation, (b) why `gate_start=6` helps, (c) part of the newline sensitivity. *Cheap test:* drop the first k tokens and the outlier channels before pooling.
2. **Inverse RoPE / position-independent caching** (EPIC, Block-Attention, CacheBlend). Un-rotate cached keys for fingerprinting (position-free geometry). Re-rotate to new positions on restore → non-prefix reuse.
3. **"All-but-the-top"** (Mu & Viswanath 2018): remove the mean and the top principal components to make embeddings isotropic. This is literally FCDB + `gate_start` under a published name. A precedent to cite and a baseline to compare against.
4. **Spiked covariance / BBP transition / Marchenko–Pastur edge** → number of components to gate, derived per model (with 219).
5. **Extreme-value theory for margins** → predicted margin(N) = c − a·√(log N) with collapse N\*; fit and compare with the v1 power law N^-0.207.
6. **Multi-vector fingerprints (ColBERT-style late interaction)**: keep top-k token keys per layer band instead of one mean, score with MaxSim. Mean-pooling throws away exactly what distinguishes long contexts. RetrievalAttention showed Q→K needs custom indexes; K→K multi-vector stays in-distribution.
7. **Content-defined chunking** (rolling hash, LBFS/rsync) for KV block boundaries instead of fixed 256-token blocks → reuse survives insertions (block-level version of the newline problem).
8. **RadixAttention prefix trees** (SGLang) as the exact-reuse layer underneath the semantic layer: exact prefix first, semantic fallback second.
9. **Conformal prediction / conformal risk control** → the false-confidence rate per result with a **finite-sample, distribution-free guarantee**. Strictly better than the heuristic CalTab in GFR v2 L6.
10. **SimHash / SRHT / Matryoshka-style nested dims** → coarse-to-fine retrieval: binary prefilter → first 256 dims → full fingerprint.
11. **Rate-distortion agent memory** (arXiv 2607.08032, 2605.10870) + **Guo–Li ICML 2026** (space-optimal memory must sometimes be confidently wrong) → DISTORTION classes and the fcr budget, from the v4 capstone.
12. **CRDT G-Set + Lamport/vector clocks** → certificate exchange between agents as set union (already in the .eng v2 spec; carry over).

---

## 5. Error register — ENGRAM-A additions (continues E-01…E-04 from v0.1)

| ID | Where | Issue | Status |
|---|---|---|---|
| **E-02** | Fourier fingerprint (`|rFFT|` ≥ 0) — **also in v1 KV path** | All-positive offset inflates cosines (baseline ≈ 0.64). | **Confirmed on vector_engram's own harness**: centering → mean\|cos\| 0.673→0.053; capacity 12→16 (raw), 12→32 (gdf), 32→48+ (projected). Synthetic data; KV check in P0. |
| **E-05** | `llama_cpp_bridge.py`, `blob_parser.py`, docs, patent text | Keys taken from llama.cpp are **post-RoPE**; no inverse rotation; docs claim pre-RoPE. | Open → P0 fix (inverse RoPE). |
| **E-06** | Paper abstract: "graceful decay with no collapse point" | A power-law fit over a limited N range can't support "no collapse"; EVT predicts a finite collapse N\*. | Open → P0 measurement at N = 10³…10⁵. |
| **E-07** | Paper/README: "architecture-invariant" | Cross-model margin +0.013 (N=200); 322 implies no isometric map exists. | Claim to be withdrawn in v2. |
| **E-08** | Margin claims | n=3, 2 documents; no embedding baseline. | Open → P0 benchmark. |

---

## 6. What this hands to the P0 test plan — hypotheses

| ID | Hypothesis | Source | Kill / pass criterion |
|---|---|---|---|
| **H1** | Inverse RoPE removes position fragility and improves long-context margin | E-05, §4.2 | Shifted-prefix and leading-whitespace tests: same doc ±1–8 tokens of offset keeps cos ≥ 0.95; margin non-decreasing from 1k → 16k |
| **H2** | Sink-token exclusion + centering reproduces most of the `gate_start` gain | §4.1, §4.3, 222/234 | Δmargin(sink-drop + center) ≥ 70 % of Δmargin(gate_start=6) |
| **H3** | BBP-derived gate ≈ empirically optimal gate, and transfers across models | 219, §4.4 | Derived gate within ±2 of the sweep optimum on ≥ 3 model families |
| **H4** | Margin(N) follows EVT √(log N), with finite collapse N\* | 093, §4.5, E-06 | Fit at N = 10³, 10⁴, 10⁵; EVT model beats power law on held-out N |
| **H5** | K-fingerprint ≥ text-embedding baseline on retrieval | E-08 (null hypothesis) | R@1/R@5 on own corpus + BEIR subset + LongMemEval retrieval split vs MiniLM / BGE-small / nomic |
| **H6** | Multi-vector (top-k token keys, MaxSim) beats the mean-pooled fingerprint | §4.6, 098, 099 | ≥ +5 pt R@1 at N = 10⁴ at ≤ 4× index size |
| **H7** | Fingerprint-find + edit-alignment-reuse beats fingerprint-only restore on edited prompts | 099 + 121, §4.2, §4.7 | Restore-hit rate and answer quality on a perturbed-prompt suite |
| **H8** | Conformal fcr achieves nominal coverage | §4.9, 111 | Empirical false-accept ≤ target α on held-out queries |
| **H9** | Compaction chains obey the 229 rule | 229 | Leaf recoverability vs depth tracks fan-in·fidelity² across the threshold |
| **H10** | k-server-style semantic prefetch beats LRU on tiered KV | 110 | Hit rate / TTFT on a multi-session agent trace |

**Ordering:** H5 + H1 + H2 decide the product shape (gate G0); H3, H4, H6 refine the fingerprint; H7–H10 belong to P2–P4.
