# ENGRAM v2 — Development Plan
### Generic agent/model memory over LLM KV state · certificate + algorithm + benchmark programme
### 2026-10-11 · Dex × Claude · branch `v2` · status: plan v1.0 (supersedes the phase sketch in `ENGRAM_AGENT_EVOLUTION_PLAN_v0.1.md` §4)

Inputs: `05_AGENT_ENGRAM/ENGRAM_AGENT_EVOLUTION_PLAN_v0.1.md` · `04_MATH_CROSSMAP/OPENAI_MATH_CROSSMAP_v0.2_AGENT.md` (H1–H10, E-02…E-08) · `01_SPEC_v2/GFR v2 ALGORITHM SPEC.md` · `01_SPEC_v2/ENG CERTIFICATE v2 SPEC.md` · `00_CURRENT/EGR3 CERTIFICATE SPEC.md` · repo `infraax/engram` @ d45273e.

---

## 0. Non-negotiables (carried from GFR v2 / EGR3, adapted)

1. **Measure before claiming.** Every number in README/paper comes from `engram-bench` with a recorded config hash + git SHA. v1 claims E-05…E-08 are not repeated until re-measured.
2. **Training-free core.** No gradients in fingerprint, write, retrieve or repair paths. Anything learned (embedder, LLM) is an external, versioned input identified by `space_key`.
3. **Spaces are types.** Fingerprints are compared only within one `space_key`. Cross-space comparison = type error.
4. **Certificates locate, payloads restore.** A certificate never pretends to contain the state (formal floor: math 140). The heavy KV lives in the payload plane, referenced by content hash.
5. **Immutable body, mutable envelope, ledger outside.** Unchanged from the .eng v2 spec. The store is a grow-only set (CRDT).
6. **Honest confidence.** Each result carries a calibrated false-confidence rate, or `UNCALIBRATED`. Never a made-up number.
7. **One core, two front-ends.** `engram_core` is shared by the agent front-end (this plan) and `vector_engram` (creature).

---

## 1. Branch, layout, machines

### 1.1 Git
- Worktree: `~/IDEAS/ENGRAM/engram-v2` on branch **`v2`** (created from `origin/main` @ d45273e). The `main` checkout at `~/IDEAS/ENGRAM/ENGRAM` and its untracked `results/`, `data/` are untouched. `data/corpus_200.jsonl` is tracked on `v2` (needed to reproduce M0).
- TUF clone: `~/dev/engram` on `v2`, tracks the same remote.
- Commit convention: `v2(<area>): …`, where area ∈ {core, cert, gfr, payload, bench, mcp, docs}. Push only on explicit OK.
- `kvcos/` stays **frozen** as the v1 reference (golden vectors are generated from it). It is deleted only once v2 reaches parity (M6).

### 1.2 Package tree (target)
```
engram_core/                 # shared, pure numpy (+ torch optional)
  space.py                   # SpaceDescriptor, space_key, registry
  transforms/                # fingerprint transform stack (§2)
    capture.py  derope.py  sinks.py  pool.py  layer_axis.py
    center.py   gate.py    project.py  pipeline.py
  cert/                      # EGR2 certificate (§3)
    schema.fbs  canonical.py  codec.py  sections.py  envelope.py  migrate_egr1.py
  gfr/                       # retrieval engine (§4)
    prefilter.py  skeleton.py  conductance.py  resonance.py
    negative.py  links.py  calibrate.py  plan.py  write.py  repair.py  ledger.py
  store/                     # CAS (BLAKE3), index partitions, bloom
engram_agent/                # agent front-end
  payload/                   # PayloadStore adapters (§5): hf.py llamacpp.py mlx.py lmcache.py
  reuse/                     # alignment + selective recompute planner
  mcp/                       # MCP v2 server
  sdk/                       # Python client; TS reader/writer lives in npm-package/
engram_bench/                # benchmark harness (§6)
  datasets/  models.py  baselines.py  runners/  metrics.py  registry.py  report.py
kvcos/                       # FROZEN v1 reference
docs/v2/                     # this plan, specs, decision log, results
```

### 1.3 Machines
| Machine | Role | Runs |
|---|---|---|
| **MacBook Air M3, 24 GB** | Reference + interactive | v1 parity (llama.cpp Metal), 3B/8B confirmation, H1/H2 short loops, cert/GFR dev, golden vectors |
| **TUF (`tuf-ts`), Ryzen 4800H, 15 GB RAM, RTX 2060 6 GB** | Batch worker (overnight) | Embedding baselines, index scaling N=10³–10⁵, gate/centering sweeps on 0.5–1.5B models (fp16 CUDA), 3B in 4-bit |
| Rented GPU (later, optional) | P3 only | vLLM + LMCache connector tests |

**Model set** (ungated first, so no HF token is needed to start):
- Iteration: Qwen2.5-0.5B / 1.5B, SmolLM2-360M / 1.7B.
- Mid: Qwen2.5-3B, Llama-3.2-3B (gated on HF → token, or GGUF via bartowski).
- Confirm: Llama-3.1-8B (v1 parity, GGUF on M3).
- Families: ≥ 3 (Qwen, Llama, SmolLM / Gemma) for every transfer claim.

---

## 2. Upgrade A — Fingerprint pipeline v2 (the transform stack)

v1 = one fixed function. v2 = an **explicit, versioned pipeline** where every stage is a registry ID inside `SpaceDescriptor`. Every experiment is a different pipeline; every pipeline is a different space.

| # | Stage | v1 behaviour | v2 options (registry IDs) | Driven by |
|---|---|---|---|---|
| S0 | **capture** | llama.cpp blob → K (post-RoPE) | `hf_past_kv` (K + position ids), `hf_prerope_hook` (k_proj output, true pre-RoPE), `llamacpp_blob` (v1 parity), `mlx_cache` | E-05 |
| S1 | **derope** | none | `none`, `inverse_rope` (exact un-rotation using rope θ / scaling config + positions) | H1, E-05 |
| S2 | **sinks / outlier channels** | none | `none`, `drop_first_k(k)`, `drop_massive_channels(z>τ)` | H2, §4.1 cross-map |
| S3 | **token pooling** | mean over all tokens | `mean`, `mean_excl_sinks`, `blockwise_mean(256)`, `topk_multivector(k)` (MaxSim) | H6, math 099/098 |
| S4 | **layer axis** | rFFT over layers, \|f0\|,\|f1\| | `rfft_amp(f0,f1)`, `rfft_amp+gdf`, `dct(k)`, `concat_layers` (no transform), `svd_perdoc(rank, gate)` (v1 MAR) | ablation |
| S5 | **center** | none (FCDB = cross-model only) | `none`, `frozen_mean(space)` (mean from a calibration corpus, hash stored) | E-02, H2 |
| S6 | **gate** | `gate_start=6` (SVD only) | `none`, `fixed(g)`, `bbp_auto` (count singular values above the Marchenko–Pastur edge) | H3, math 219/222/234 |
| S7 | **project** | none | `none`, `gaussian(D, seed)`, `srht(D, seed)` | capacity / speed |
| S8 | **normalize** | L2 per block | `l2`, `l2_per_block`, `none` (MIPS, v1 D4) | — |

**Default candidates** (to be decided by G0, not by taste):
- `FP-A` = hf_prerope → mean_excl_sinks → rfft_amp(f0,f1) → frozen_mean → l2
- `FP-B` = hf_past_kv → inverse_rope → drop_first_k → svd_perdoc(128, bbp_auto) → l2
- `FP-MV` = hf_prerope → topk_multivector(k=32) → frozen_mean → l2 (MaxSim)
- `EMB-*` = text embedding baselines (MiniLM, BGE-small, nomic-embed) — the MEANING sense

**Two senses, re-mapped:**
- **REFLEX = K-geometry space** (open models only).
- **MEANING = text-embedding space** (any agent, incl. closed APIs).
- Never cross-compared; fused at the result level (§4.6).

---

## 3. Upgrade B — Certificate `.eng` v2 (EGR2)

Base: `ENG CERTIFICATE v2 SPEC.md` (canonical TLV + BLAKE3, FlatBuffers container, immutable body / mutable envelope, typed optional sections). Agent-specific deltas below.

### 3.1 SpaceDescriptor (replaces EGR1's `corpus_hash` / `model_id` / `layer_range` header fields)
```
SpaceDescriptor {
  sense            : REFLEX_KV | MEANING_TEXT | (future)
  model_id         : string     // e.g. "Qwen2.5-1.5B@<weights-sha>"
  model_cfg_hash   : u64        // n_layers, n_kv_heads, head_dim, rope cfg
  pipeline         : [ {stage_id: u16, params: bytes} ]   // S0..S8, exact
  center_hash      : u64?       // hash of the frozen mean vector (stored in registry)
  proj_seed        : u64?
  dims             : u16
  embedder_id      : string?    // MEANING spaces
}
space_key = BLAKE3(canonical(SpaceDescriptor))[0:8]
```
Changing **any** stage parameter ⇒ new space ⇒ new index partition ⇒ explicit migration. This is what makes the experiment matrix and the product the same mechanism.

### 3.2 CORE (mandatory)
```
format_ver   u16 = 0x0200        space_key   u64        role  CORE|MARGIN
kind         STATE|SESSION|SUMMARY|TOOL_RESULT|GAP|MAINTENANCE|PROPOSAL
timestamp_us u64 (metadata only) agent_id    u32        session_id u64
dims u16 · precision F16|I8 · fp [bytes, 16-aligned]
scs f32 · self_margin f32 · crc32 u32
```

### 3.3 New and changed sections (agent)
| id | Section | Purpose |
|---|---|---|
| **0x30** | **PAYLOAD_REF** | `{kind: KV_HF\|KV_LLAMACPP\|KV_MLX\|KV_LMCACHE\|TEXT, content_hash, token_hash, n_tokens, block_size, block_hashes[], quant, bytes, location_hint}` — the certificate points at the state, never contains it |
| **0x31** | **TOKEN_SPAN** | Token-hash chain per block (content-defined boundaries), for exact prefix match + edit alignment (math 099/121) |
| **0x32** | **TEXT_DIGEST** | Optional short text (≤ 1 KB) or pointer for RECALL when the KV can't be restored (closed agents, model mismatch) |
| **0x33** | **MULTIVECTOR** | Optional top-k token fingerprints (FP-MV) |
| 0x10 | PARENTS | Previous-session / derived-from chain → session causality (replaces the flat "last session") |
| 0x11 | LINKS | CONTINUATION, DERIVED_FROM, CONTRADICTS, SUPERSEDES, ANSWERS, … |
| 0x12 | ROLE_MARGIN | Corrections, consolidation, redaction |
| 0x13 | FACETS | actor(agent) · action(tool/task type) · object(topic hash) · context(project/session) · time bucket |
| 0x14 | CLOCK | Lamport + vector clock per agent |
| 0x15 | PROOFREAD | Write-time self-retrieval audit |
| 0x17 | DISTORTION | FULL_FIDELITY (KV) · SUMMARY · GIST, + compaction depth (math 229 rule) |
| 0x18 | REGIME | LOCAL/AP/CP per operation class |
| 0x1B | ERASURE_PROOF | Payload deleted, provenance kept |
| **C11** | `derived` flag in provenance | LLM-written summaries are marked permanently and can never be relabelled as first-hand |

### 3.4 Envelope / ledger / registry
- **Envelope:** flags (ORPHAN, TOMBSTONED, PAYLOAD_EVICTED, PAYLOAD_ERASED, QUARANTINED, PROVISIONAL), payload tier location.
- **Ledger:** conductance, hits/wins, confusion pairs — droppable.
- **Registry:** spaces, frozen means, projection seeds, vocab tables.

### 3.5 Migration
- `migrate_egr1.py`: EGR1 → EGR2 CORE in space `legacy_egr1_perdoc` / `legacy_egr1_fcdb`.
- No payload ref (v1 certs never had one); flagged `PAYLOAD_EVICTED`.

### 3.6 Size target
- CORE + typical sections ≈ 1.0–1.4 KB.
- With MULTIVECTOR (k=32, 256-d, F16) ≈ +16 KB.
- Payload separate (MBs).

---

## 4. Upgrade C — GFR-A retrieval and write engine

Base: `GFR v2 ALGORITHM SPEC.md` (entry points Q/W/R/L; L0–L6; Physarum ledger; Mode B; G-Set CRDT). Agent deltas:

### 4.1 Query pipeline
```
QCtx → pre-filter (space_key, facets, envelope flags)
     → L0  transform stack for the query (same space)
     → L0.5 exact layer: RadixAttention-style prefix match on TOKEN_SPAN (free hit if exact)
     → L1  binary prefilter (SimHash/SRHT bits, Hamming) → HNSW/USearch candidates (K1=50)
     → L2  Physarum conductance re-weight (utility-driven)            [X1]
     → L3  resonance + cleanup (Hopfield, β calibrated per space)     [X3]
     → L4  negative constraints (v1 IndexC ported)
     → L5  Mode B typed-link traversal (session chain, derivations)
     → L6  calibrated output: conformal fcr per result (math cross-map §4.9)
     → ACTION PLAN: RESTORE | RECALL | CITE   (§4.3)
```

### 4.2 Write pipeline (pipe-composed, stage order load-bearing)
1. **Admission:** budget-aware accept/reject.
   - Prophet-inequality baseline: one past sample per slot (math 111).
   - Surprise/novelty optional.
2. **Dedup:**
   - Exact: token-hash.
   - Near: bloom + cosine ≥ τ_dup → MARGIN `RE_OBSERVATION`, no new payload.
3. **Transform + project** (space pipeline).
4. **Proofread:** self-query L1–L4. Collision → confusion pair + CONTRADICTS.
5. **Facets, clock, links** (training-free rules only).
6. **DISTORTION class + compaction depth check** (math 229: refuse a summary layer whose fan-in × fidelity² ≤ 1).
7. **Hash + commit:** BLAKE3 → CAS → index → bloom → ledger init. The payload is written by the PayloadStore and referenced by hash.

### 4.3 Action planner (new)
| Situation | Action | Mechanism |
|---|---|---|
| Same model, exact prefix | RESTORE (exact) | Load KV blocks, continue |
| Same model, near match (edited) | RESTORE (partial) | Edit alignment (math 121) → reuse aligned blocks, selective recompute of the diff (CacheBlend-style), re-rotate positions |
| Same model, several relevant states | RESTORE (composed) | Order chunks by superstring overlap (math 128, 2-approx), recompute boundaries |
| Different model / closed agent | RECALL | TEXT_DIGEST or retrieve source text; no KV |
| Low confidence (fcr > class threshold) | CITE only | Return pointers + fcr, no injection |

### 4.4 Ledger and tiers
- Physarum conductance as in GFR v2, with utility u from agent events: `accepted`, `tool_success`, `restored_and_used`, explicit feedback.
- Tier placement (GPU/CPU/disk/remote) as a k-server-style policy over a semantic + tier-cost metric (math 110), with LRU as the baseline.

### 4.5 Repair and consolidation
- Repair rules R1–R8 as in GFR v2.
- Consolidation at session close / idle:
  - merge near-duplicates
  - k-median prototype selection (math 125, 1+2/e is the target, no better)
  - compaction with the 229 check
  - conformal recalibration
  - maintenance + GAP certificates

### 4.6 Fusion of senses
- Result = `{reflex: […], meaning: […], links: […], overlap: […]}`.
- Labelled, never silently merged. RESTORE is only possible from REFLEX results whose model matches the live engine.

---

## 5. Payload plane adapters

```
class PayloadStore(Protocol):
    def capture(self, engine, span) -> PayloadRef            # extract KV blocks + token hashes
    def restore(self, engine, ref, plan) -> RestoreReport    # inject (exact / partial / composed)
    def evict(self, ref, tier) / locate(ref) -> Tier
```
| Adapter | Priority | Notes |
|---|---|---|
| `hf` (transformers DynamicCache) | **P0** | Research + bench; gives K tensors + positions directly |
| `llamacpp` | P0 (parity) / P3 | v1 bridge ported; Metal on M3 |
| `mlx` | P3 | Apple Silicon production path; compete with agent-memory niche |
| `lmcache` | P3+ | vLLM/SGLang; rented GPU |

---

## 6. `engram-bench` — the harness

### 6.1 Datasets
| ID | Source | Use |
|---|---|---|
| `c200` | v1 `data/corpus_200.jsonl` (10 domains × 20) | v1 parity, quick loops |
| `beir-small` | BEIR SciFact, NFCorpus, FiQA (doc-level subsets) | Public retrieval benchmark vs embedders (H5) |
| `lme` | LongMemEval retrieval split (sessions → evidence) | Agent-session memory (H5, H9) |
| `scale` | c200 + BEIR docs + synthetic distractors up to 10⁵ | Margin(N), collapse N\* (H4) |
| `perturb` | Each doc × {leading/trailing whitespace, ±1–8 token prefix shift, typo, sentence insert/delete, reorder} | Position fragility (H1), edit reuse (H7) |
| `long` | Same docs at 1k/4k/16k/32k context | Long-context collapse (H1, H6) |
| `trace` | Recorded multi-session agent traces (Claude Code sessions via MCP v1 logs + synthetic) | Tiering (H10), admission, consolidation |

### 6.2 Metrics
- **Retrieval:** R@1, R@5, MRR, nDCG@10, margin (correct − best impostor), margin(N) curve.
- **Calibration:** empirical false-accept rate vs nominal (conformal coverage).
- **Restore:** TTFT cold/warm, restore-hit rate, **answer quality after restore** (exact-match / LLM-judge vs cold baseline).
- **Cost:** fingerprint ms, index ms, cert bytes, payload bytes, RAM/VRAM peak.
- **Stability:** cos(fp(doc), fp(perturbed doc)) distribution.

### 6.3 Run registry
- Every run writes `results/v2/<date>/<run_id>.json` with: config (pipeline, model, dataset, N), config hash, git SHA, machine, seeds, metrics, timings.
- `report.py` builds the tables.
- No number goes into docs without a run_id.
- Golden vectors: `tests/golden/` generated from `kvcos/` (v1) and from each v2 pipeline. Cross-language parity rel-L2 ≤ 1e-5; certificate bytes bit-exact.

---

## 7. Experiment matrix (P0 → G0 and beyond)

| ID | Question | Machine | Models | Datasets | Pass / kill |
|---|---|---|---|---|---|
| **E0** | v1 parity reproduces | M3 | Llama-3.2-3B / 3.1-8B GGUF | c200 | R@1 within ±2 pt of v1 (98 %); else document the drift |
| **H5** | REFLEX ≥ MEANING baseline? (**null hypothesis**) | TUF + M3 | Qwen-1.5B, SmolLM2-1.7B, Llama-3B | c200, beir-small, lme | Decides G0 (below) |
| **H1** | inverse RoPE fixes fragility + long-context | M3 | Qwen-1.5B, Llama-3B | perturb, long | stability cos ≥ 0.95 under shifts; margin non-decreasing 1k→16k |
| **H2** | sink-drop + centering ≈ gate gain | TUF | 0.5–1.5B ×3 families | c200, beir-small | ≥ 70 % of the `gate_start=6` gain |
| **H3** | BBP gate ≈ swept optimum, transfers | TUF | ×3 families | c200 | Within ±2 of the optimum on ≥ 3 families |
| **H4** | margin(N) is EVT-shaped, finite N\* | TUF | best pipeline | scale | EVT fit beats power law on held-out N |
| **H6** | multi-vector beats mean-pool | TUF | 1.5B | long, beir-small | ≥ +5 pt R@1 at N=10⁴, ≤ 4× index |
| **H7** | find + align + selective recompute beats find-only restore | M3 | 1.5B/3B | perturb | Higher restore-hit + equal answer quality |
| **H8** | conformal fcr valid | TUF | best | all | Empirical false-accept ≤ α |
| **H9** | compaction obeys the 229 rule | TUF | 1.5B | lme | Leaf recoverability tracks fan-in·fidelity² |
| **H10** | semantic prefetch beats LRU | M3 | 1.5B | trace | Hit rate / TTFT ↑ |
| **X1/X3** | Physarum vs static; resonance vs kNN | TUF | best | trace, lme | GFR v2 kill criteria |

**Gate G0** (after E0, H5, H1, H2):
- **G0-A:** REFLEX ≥ best embedder on R@k (any dataset family) → KV-native semantics is the headline; both senses ship.
- **G0-B:** REFLEX < embedder but restore beats cold on TTFT with equal answer quality → REFLEX = restore key; MEANING does semantics.
- **G0-C:** neither → drop the KV fingerprint as a retrieval signal; v2 = certificate + GFR over embeddings + restore by exact prefix only.

---

## 8. Iteration loop

1. **Hypothesis card** (`docs/v2/cards/Hx.md`): claim, pipeline(s), dataset, metric, pass/kill, expected direction.
2. **Run** through `engram_bench.runners` (local or `ssh tuf-ts` batch; long sweeps nohup + log on TUF).
3. **Record:** run JSON + one-line entry in `docs/v2/DECISION_LOG.md` (accept / adjust / kill + why).
4. **Promote:** only pipelines that pass a card get a registry ID marked `candidate`. Defaults change only at gates.
5. **Sweeps** (gate, k, β, τ): autoresearch-style loop on TUF overnight. A fixed eval set and a held-out set are both required.

---

## 9. Milestones

| M | Deliverable | Duration | Exit evidence |
|---|---|---|---|
| **M0 Fundament** | Envs on M3 + TUF; `engram_core/space.py`; transform stack S0–S8 (hf capture first); bench skeleton; c200 + perturb loaders; run registry; golden vectors from kvcos | ~1 wk | Both machines run `bench smoke`; v1 golden vectors reproduce |
| **M1 P0 / G0** | E0, H5, H1, H2 (+ H3 if time); G0 decision recorded | ~1–2 wk | G0 entry in the decision log with run_ids |
| **M2 Certificate** | EGR2 schema + canonical codec + sections + envelope + EGR1 migration; TS reader | ~1 wk (parallel with M1) | Round-trip bit-exact Py⇄TS; 30+ tests |
| **M3 GFR-A** | L0–L6, write path, ledger, conformal calibration, repair R1–R8 | ~2 wk | lme + beir-small numbers; X1/X3 outcomes logged |
| **M4 Payload & restore** | hf + llamacpp adapters; action planner; partial/composed restore | ~2 wk | H7 result; TTFT + answer quality table |
| **M5 Agent surfaces** | MCP v2 tools; session chain certs; multi-agent G-Set sync | ~1–2 wk | Claude Code uses v2 for its own session memory |
| **M6 Publish** | Paper v2 (corrected claims), README, kvcos removal | ~1–2 wk | All claims traceable to run_ids |

Total ≈ 9–12 weeks solo. M2 runs in parallel with M1.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| G0-C outcome (KV fingerprint loses) | Plan already survives it: certificate + GFR + exact restore still ship |
| RTX 2060 VRAM (6 GB) limits model size | Iterate on ≤ 1.5B; confirm on M3/8B; rented GPU only for P3 |
| Gated Llama weights | Start with Qwen/SmolLM; HF token or GGUF for Llama parity |
| M3 thermal throttling | Long jobs on TUF |
| Scope creep (field, COMPOSE, cross-model) | Explicitly parked; reopening needs a decision-log entry |
| Public patent disclosure | Owner decision D4 (plan v0.1) — outside this plan |

---

## 11. First actions (now)

1. Env setup: M3 worktree + venv (py3.12, torch-MPS, transformers, faiss, llama-cpp Metal, mlx); TUF venv via uv (torch cu126). *(running)*
2. `engram_core/space.py` + `transforms/` skeleton with S0 `hf_past_kv` / `hf_prerope_hook`, S1 `inverse_rope`, S3 pools, S4 `rfft_amp`, S5 `frozen_mean`.
3. `engram_bench`: c200 loader, perturb generator, embedder baselines, R@k/margin metrics, run registry.
4. First runs: **H1 smoke** (Qwen2.5-0.5B, c200 + perturb) on M3; **H5 baselines** (MiniLM/BGE/nomic on c200) on TUF.
