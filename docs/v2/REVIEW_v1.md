# ENGRAM-A — Evolving the original ENGRAM into a generic Agent / Model memory
### Review of `infraax/engram` (v1, Apr 2026) + `chimera-brain/vector_engram` (Jun 2026) → upgrade plan using .eng v2 / GFR v2 / ENGRAM-R research
### 2026-10-11 · Dex × Claude · status: plan v0.1 (pre-Voorkeursbeslissing)

---

## 0. One-line verdict

**The idea is real, but what's actually proven is narrower than the README says.** The proven value is **semantic retrieval keyed on the model's own K-geometry, combined with KV restore (27–67× TTFT)**. "Architecture-invariant, corpus-independent memory" is not proven: cross-model margin is +0.013 at N=200. Build v2 around what is proven, and put the certificate + GFR as the semantic layer *above* existing KV storage (LMCache / llama.cpp / MLX). Don't compete on storage.

---

## 1. What v1 actually is (as read from the code)

| Component | What it does | Status |
|---|---|---|
| `core/fingerprint.py` | Mean key per layer (mean over **all tokens**) → rFFT over the **layer axis** → amplitude f0 + f1, each L2-normed → 2·dim vector | Works same-model: 98 % R@1 at N=200 (own corpus) |
| `core/state_extractor.py` | MAR state vector: per-doc SVD on keys (layers 8–24, rank 128, `gate_start=6`), mean-pooled projection | Best same-model discriminator: margin 0.38 → 0.52 with gating (n=3, 2 docs) |
| FCDB | Delta from a joint Fréchet mean + joint SVD → cross-model basis | Margin +0.124 at N=50 (unstable basis), **+0.013 at N=200** (stable but marginal) |
| `engram/format.py` EGR1 | 99-byte header + fp16 `vec_perdoc` + `vec_fcdb` + 128-d `joint_center`, ~856 B | Works, 20 tests |
| `engram/retrieval.py` | 4-stage geodesic: IndexC prior → HNSW → trajectory correction → negative constraints → metadata tiebreak | Takes 98 % → 100 % on own corpus |
| `integrations/llama_cpp_bridge.py` | Extract / **inject** llama.cpp state blob; cold vs warm TTFT | **Strongest result:** 27–67× TTFT on 0.6k–16k tokens |
| `engram/embedder.py` + `mcp/engram_memory.py` | Text → fingerprint via a local llama.cpp model → sbert → hash fallback; 7 MCP tools for Claude Code session memory | Alpha |
| Results doc | Honest negative results (9 cross-model methods, excluded runs) | Good scientific hygiene — keep this culture |

`vector_engram` changed the *input*, not the mechanism: a `StateVector` seam ([T, D] time window instead of the layer axis), GDF phase, two senses, HRR/COMPOSE, resonance, streaming writer, hot/cold, IndexC-lite.

---

## 2. Findings — blunt

### 2.1 What is genuinely strong
- **Retrieve-then-restore loop.** Retrieval keyed on K-geometry chooses *which* cached state to inject. LMCache does prefix/hash reuse and CacheBlend does non-prefix reuse, but neither selects states by semantic K-similarity. This is still the differentiator.
- **Certificate concept.** Small cert (~856 B) separate from the heavy payload (74 MB fp16 at 591 tokens; 800 MB at 14k), with confidence baked in (SCS, margin).
- **Negative-results discipline** in `results/`.

### 2.2 What is weak or wrong

| # | Issue | Evidence | Consequence |
|---|---|---|---|
| W1 | **Thin evidence base.** Margin claims rest on n=3 trials with 2 documents. Scaling claims rest on one self-built 200-doc corpus. No public benchmark. **No baseline against a plain text-embedding model.** | `PAPER_TABLE.md` | The null hypothesis "MiniLM/BGE on the same text retrieves as well or better" has never been tested. This is the #1 risk to the whole thesis. |
| W2 | **Keys are post-RoPE, not pre-RoPE.** llama.cpp writes K into the cache *after* RoPE. The bridge has no un-rotation step, yet docs and patent text claim pre-RoPE. | `blob_parser.py`, `llama_cpp_bridge.py` (no inverse RoPE anywhere) | Explains the "leading newline flips the margin sign" result (one token shift rotates every key). That was framed as "high resolution"; it's position fragility. Mean-pooling post-RoPE keys over long contexts also averages the high-frequency RoPE dimensions toward zero — a plausible cause of long-context collapse (the 28k/14k failures blamed on filler text). |
| W3 | **All-positive amplitude fingerprint (same as E-02).** `|rFFT|` → every entry ≥ 0 → unrelated vectors already sit around cos ≈ 0.64. | Same math as the Vector capacity finding | "Cross-model invariant, cos ≈ 0.89" is partly this offset. FCDB *works because it centers* (delta from the Fréchet mean) — you found the fix in April and again in October. |
| W4 | **Cross-model is not working.** +0.013 margin. | `PAPER_TABLE.md` | Drop "architecture-invariant" from the claims. Treat each model as its own `space_key`. Cross-model = re-encode/migrate, not compare. |
| W5 | **Closed-model agents (Claude, GPT) expose no KV.** The MCP path re-encodes text through a *local* model. | `embedder.py` | For closed agents, ENGRAM's fingerprint is just "a small LLM used as an embedding model". It must beat dedicated embedders or yield to them. The certificate + GFR is still the value there. |
| W6 | **Payload size.** 74 MB at 591 tok, 800 MB at 14k (fp16), ~2× with INT8. | `PAPER_TABLE.md` | Don't build storage tiering. LMCache already does (CPU/disk/remote, 256-token blocks). |

### 2.3 Verified today
**E-02 on vector_engram's own capacity harness** (frozen mean from an independent corpus, N=64):

| Condition | mean \|cos\| | capacity@0.9 |
|---|---|---|
| raw @230 | 0.673 | 12 |
| **raw centered @230** | **0.053** | **16** (= random ceiling) |
| gdf @460 | 0.374 | 12 |
| **gdf centered @460** | **0.040** | **32** |
| projected raw @2048 | 0.679 | 32 |
| **projected centered @2048** | **0.056** | **48+** (= random ceiling) |

Caveat: the harness data is `synth.py` (synthetic situations), not lived data. The mechanism is confirmed; the magnitude on real KV states is still to be measured (P0).

### 2.4 Landscape check (Oct 2026)
- **LMCache** has reframed itself as more than a KV cache ("something much bigger", Apr 2026). It owns storage, tiering and engine integration (vLLM/SGLang).
- **CacheBlend** (EuroSys '25 best paper): non-prefix KV reuse with selective recompute → the mechanism ENGRAM needs to restore a *retrieved* non-prefix state correctly.
- **agent-memory** (arXiv 2603.04428, yshk-mxim): persistent Q4 KV cache for multi-agent on Apple Silicon. Same hardware niche, no semantic retrieval.
- → The gap is still **semantic selection + certificate/provenance + calibrated confidence**. The window is narrowing.

---

## 3. The reframe: ENGRAM-A = three planes, one shared core

```
┌────────────────────────────────────────────────────────────────────┐
│ AGENT SURFACES   MCP v2 · Python/TS SDK · REST · multi-agent sync   │
├────────────────────────────────────────────────────────────────────┤
│ RETRIEVAL PLANE  GFR-A (GFR v2 ported): pre-filter → HNSW → Physarum│
│                  conductance (utility) → Hopfield cleanup → negative│
│                  constraints → Mode B typed links → calibrated fcr  │
│                  Actions: RECALL (text) · RESTORE (inject KV) · CITE│
├────────────────────────────────────────────────────────────────────┤
│ CERTIFICATE PLANE  .eng v2-A (EGR2): canonical TLV + BLAKE3, G-Set  │
│                  CRDT, space_key, payload_ref, PARENTS/LINKS/CLOCK, │
│                  FACETS, DISTORTION, REGIME, PROOFREAD, ERASURE     │
├────────────────────────────────────────────────────────────────────┤
│ PAYLOAD PLANE    NOT OURS: LMCache (vLLM/SGLang) · llama.cpp state  │
│                  · MLX · plain text. Referenced by content hash.    │
└────────────────────────────────────────────────────────────────────┘
          shared core package: engram-core (transforms, codec, GFR)
          front-ends: engram-agent (this plan) · vector_engram (creature)
```

### 3.1 Two senses, re-mapped for agents (from `sense.py`)
| Vector | Agent equivalent | Space |
|---|---|---|
| REFLEX (raw signal → FFT) | **K-geometry fingerprint** of the model's own KV (open models only) | `space_key = H(model_id, layers, transform, rope_mode, center_hash, proj_seed)` |
| MEANING (learned embeddings → FFT) | **Text-embedding fingerprint** (any model, incl. closed agents) | `space_key = H(embedder_id, …)` |

Never compared across spaces (type error, as in GFR v2). Fusion happens only at the result level.

### 3.2 What carries over from each research line

| From | Into ENGRAM-A | Status |
|---|---|---|
| E-02 centering | Frozen per-space mean subtracted before projection; FCDB generalised | Verified (synthetic); P0 on real KV |
| `.eng` v2 / EGR3 | Certificate plane almost 1:1. `payload_ref` replaces `fp_primary`-only certs. TRAJECTORY → **session windows** (block sequence). GAP certs → failed tool calls / truncations | Spec exists |
| GFR v2 | Retrieval plane; utility `u` = task success / answer accepted / restored-and-used | Spec exists |
| GDF (order) | Order over **token blocks** within a session (trajectory of a conversation), not over layers | Hypothesis |
| Salience gate | Write gating: certify a block only on surprise/novelty/explicit marker → controls storage | Hypothesis |
| Dock cycle / RCL-Cons | Consolidation at **session close / idle**: merge near-dups, decay conductance, rebuild calibration, project-identity basis | Hypothesis |
| Ψ identity + drift | Per-agent / per-project principal basis; drift = behavioural drift monitor | Optional, later |
| C11 "derived" provenance | LLM-written summaries permanently marked `derived`; never laundered into first-hand records | Adopt |
| Math crossmap | k-server (HotSet = warm KV set on GPU/CPU), prophet inequality (write budget), thin trees/sparsifier (shared multi-agent sync subset), 140 memory–sample floor, 229 summarisation rule | Per item |

### 3.3 What does NOT carry over
- The **RCL living field** (continuous 30 Hz activation). Agents are event-driven; a persistent field is optional later (warm-set modelling), not v2.
- **Soma / embodiment**, HRR COMPOSE (erasure problem; already rejected in ENGRAM-R).
- E-01 (field-norm bug) only matters if a field is ever added.

---

## 4. Phased plan (MIRT-style gates)

### P0 — Honesty baseline (≈1–2 weeks) · *the gate everything depends on*
1. Reproduce v1 numbers on the M3 (golden vectors frozen as the v1 reference).
2. **Fix W2:** inverse-RoPE the cached keys (positions are known; exact, training-free) → test the leading-newline case and margin vs context length (1k / 4k / 16k / 32k).
3. **Fix W3:** frozen-mean centering for the Fourier fingerprint; re-measure mean |cos|, R@1 and margin.
4. **Baseline (W1):** same documents through MiniLM / BGE-small / nomic-embed. Also run a public retrieval benchmark subset (BEIR subset + LongMemEval/LoCoMo retrieval split) at N = 1k / 10k.
5. Measure the restore path end-to-end: retrieve → inject → answer quality (not just TTFT).

**Gate G0 (decides the product shape):**
- K-fingerprint ≥ embedding baseline on R@k → **KV-native semantics is the headline**.
- K-fingerprint < baseline, but restore wins on TTFT → **KV fingerprint = restore key; semantics from embeddings** (two-sense design still holds; claims change).
- Both fail → the KV fingerprint is dropped. ENGRAM-A = certificate + GFR over embeddings (still novel through provenance, calibration and CRDT federation, but a different paper).

### P1 — Certificate plane `.eng` v2-A (≈1 week)
- Fork `ENG_CERTIFICATE_v2_SPEC.md` → `EGR2-A`. CORE gains `payload_ref {kind: KV_LMCACHE|KV_LLAMACPP|KV_MLX|TEXT, content_hash, token_hash, n_tokens, block_size}`.
- Canonical TLV + BLAKE3, FlatBuffers container, golden vectors Python ⇄ TS (npm package already exists) ⇄ Go later.
- EGR1 → EGR2 migration tool (v1 certs become CORE-only certs in a `legacy` space).

### P2 — Retrieval plane GFR-A (≈2 weeks)
- Port GFR v2 L0–L6. L4 = existing IndexC + negative constraints (already in v1).
- Utility signal from agent events: `accepted`, `tool_success`, `restored_and_used`, explicit feedback.
- Calibration table (fcr per margin bin × store size), built by a probe sweep. Results report `fcr` or `UNCALIBRATED`, never a made-up number.
- X1 (Physarum vs static) and X3 (relaxation vs kNN) re-run on agent workloads.

### P3 — Payload integration (≈2 weeks)
- Connectors: **LMCache** (vLLM/SGLang, block-hash addressing), llama.cpp state (existing bridge, inverse RoPE), MLX (Apple Silicon, competes directly with agent-memory).
- **Non-prefix restore** via CacheBlend-style selective recompute → RESTORE works for retrieved states that aren't a prefix of the current prompt.
- Storage stays the payload plane's job. ENGRAM keeps certs and `payload_ref`s only.

### P4 — Agent surfaces (≈1–2 weeks)
- MCP v2 tools: `remember`, `recall`, `restore`, `link`, `forget` (erasure with proof), `explain` (Mode B path + fcr), `session_open/close`.
- Session protocol v2: keep VALIDATED / CURRENT / NEXT / OPEN, but written as a cert with PARENTS = previous session → causal chain instead of a flat list.
- Multi-agent: certificate exchange as G-Set union; `CLOCK` vector clocks per agent; REGIME for erasure (CP) vs everything else (AP).

### P5 — Consolidation (≈1 week, after real usage data)
- Session-close / idle job: merge near-duplicates, decay conductance, rebuild calibration, write maintenance + GAP certs.
- Optional: per-project identity basis + drift metric.

### P6 — Publish (≈1–2 weeks)
- Paper v2: corrected claims (W1–W4), public benchmark numbers, honest negative results kept.
- Repo restructure (§5), README rewrite, tests.

**Total ≈ 9–12 weeks solo**, gated at G0. P1 can start in parallel with P0 (the format doesn't depend on the G0 outcome).

---

## 5. Repository strategy (recommendation)

- **Keep `infraax/engram`, create `v2` branch → `main` at P6.** The name, history and paper lineage are assets.
- **New `engram_core/` package:** transforms (Fourier, centering, projection, GDF), certificate codec (EGR2), GFR. Pure numpy/torch-optional.
- **`kvcos/` frozen as the v1 reference + golden vectors.** No new features; deleted once v2 has parity.
- **`chimera-brain/vector_engram` depends on `engram_core`** instead of carrying copies → one core, two front-ends (agent / creature). Fixes drift between the two codebases (`format.py` already diverged: EGR1 vs EGRV).
- **npm package** becomes the TS reader/writer for EGR2 (agent frameworks are largely TS).

---

## 6. Decisions needed (Owner)

| # | Decision | Recommendation |
|---|---|---|
| D1 | Run P0 before anything else, and accept that G0 may demote the KV fingerprint | **Yes** — every downstream claim depends on it |
| D2 | Primary payload target: LMCache/vLLM (server/GPU) vs MLX/llama.cpp (Apple Silicon, local) | **llama.cpp + MLX first** (your hardware, existing bridge, the agent-memory niche), LMCache second |
| D3 | Same repo `v2` branch + shared `engram_core` | **Yes** |
| D5 | Benchmark set for G0 | Own 200-doc corpus + a BEIR subset + LongMemEval retrieval split |

---

## 7. Sources
- [LMCache — "Stop Calling It KV Cache: It's Something Much Bigger" (Apr 2026)](https://blog.lmcache.ai/en/2026/04/28/stop-calling-it-kv-cache-its-something-much-bigger/)
- [LMCache GitHub](https://github.com/lmcache/lmcache) · [LMCache tech report](https://lmcache.ai/tech_report.pdf)
- [CacheBlend (arXiv 2405.16444)](https://arxiv.org/pdf/2405.16444) · [EuroSys'25 best paper post](https://blog.lmcache.ai/2025-03-31-eurosys)
- [agent-memory: Persistent Q4 KV cache for multi-agent on Apple Silicon (GitHub)](https://github.com/yshk-mxim/agent-memory) · [arXiv 2603.04428](https://arxiv.org/abs/2603.04428v1)
- Internal: `infraax/engram` @ d45273e (2026-04-04), `chimera-brain/vector_engram` @ 417ddcc (2026-06-26), `00_CURRENT/*`, `01_SPEC_v2/*`, `04_MATH_CROSSMAP/*`
