# GFR v2.0 — Geodesic Flow Retrieval
## Cornerstone A: the ENGRAM memory algorithm — full engineering specification
### Companion to `ENG_CERTIFICATE_v2_SPEC.md` (Cornerstone B). Designed as one coupled pair.
### Status: engineering spec, 2026-08-18 · supersedes the sketch in `engram-foundation-v2-algorithm-and-certificate.md` §1

---

## 0. Scope, axioms, non-negotiables

GFR is **one engine with four entry points** that share the same machinery:

| Entry point | Purpose | Consumes | Produces |
|---|---|---|---|
| **GFR-Q** | Query (retrieval) | QCtx | ranked, calibrated result sets (Mode A + Mode B) |
| **GFR-W** | Write (commit pipeline) | StateVector / content | committed `.eng` v2 certificate + index/bloom/ledger entries |
| **GFR-R** | Repair (self-stabilization daemon) | store shards | convergence back to legal state L; calibration tables |
| **GFR-L** | Ledger (adaptivity service) | resolution events | conductance / fitness / confusion updates |

Theoretical identity (v4 capstone T7): ENGRAM is a **rate-distortion-constrained associative vector memory implemented as a CAP-constrained distributed store**. GFR is the read/write realization of that identity.

**Non-negotiables (violating any of these is a spec bug):**

1. **Training-free core.** No gradients anywhere in Q/W/R paths. All adaptive parameters are analytic formulas or bounded counters. (Titans-style learned re-ranking = deferred auxiliary slot, outside GFR.)
2. **Single utility signal.** Every adaptive mechanism (conductance, confusion lists, lineage fitness) is driven by one signal `u` = confirmed retrieval utility. No mechanism gets its own private heuristic score. (Foundation §1.1 discipline.)
3. **Two-senses isolation.** Fingerprints are only ever compared within an identical `space_key` (see Cert spec §2). Cross-space cosine = type error, rejected at the pre-filter, not merely discouraged.
4. **Edge state lives outside certificates.** Certificates are immutable nodes; all mutable retrieval state (conductance, fitness, confusion) lives in the Ledger.
5. **Crash-droppable adaptivity.** Losing the entire Ledger loses performance, never correctness. The CAS + index are the only load-bearing stores; the index is rebuildable from the CAS.
6. **Determinism tiers.** Robot profile: bit-deterministic given (inputs, seeds, hot-set snapshot). Box profile: deterministic given (inputs, seeds, ledger state); ledger evolution is order-dependent and accepted as such.
7. **Honesty tiers carried through:** every mechanism below is tagged **[PROVEN]** (measured in ENGRAM), **[PROVEN ELSEWHERE]** (formal result in source domain, unvalidated for ENGRAM), or **[HYPOTHESIS]** (design bet, has a kill criterion in §12).

**The two-graph model (load-bearing distinction):**

- **Assertion graph** — immutable, declarative. Nodes = certificates; edges = `PARENTS` (Merkle derivation, acyclic) and `LINKS` (typed lateral assertions, may cycle). Written once at commit. Mode B traverses this.
- **Flow graph** — mutable, behavioral. Nodes = certificates; edges = HNSW neighbor edges ∪ assertion edges, each carrying a Ledger conductance. Mode A's L2 flows over this.

The certificate defines the assertion graph; the Ledger annotates the flow graph. Never conflate them.

---

## 1. Data structures

```
SpaceDescriptor   — see Cert spec §2. space_key = H(desc)[0:8].

QCtx {
  space_key   : u64          // hard partition selector
  z_q         : f32[P]       // projected query vector, P = proj_dim (2048 default)
  fp_q        : f16[dims]    // raw fingerprint (kept for audit / cleanup)
  facet_filter: FacetPred?   // optional 5-facet predicate (bitmap intersection)
  regime      : LOCAL|AP|CP  // read-consistency demand
  mode        : A|B|A+B
  k           : u16          // requested results (default 5)
  authority   : AuthCtx      // for SEALED / erased visibility
}

EdgeRec {                     // Edge Ledger, hash-map keyed
  key    : u128 = H(min(a16,b16) ‖ max(a16,b16) ‖ space_key)
  D      : f32   // conductance ∈ [0, D_max]
  t_last : u32   // seconds since store epoch, for lazy decay
}

NodeRec {                     // Node Ledger
  cert     : hash16
  hits     : u32   // times returned in top-k
  wins     : u32   // times confirmed useful (u ≥ 0.5)
  confusion: [(hash16, f16)]  // ≤ 8 negative-constraint partners + penalties
  t_last   : u32
}
// lineage fitness(v) := (wins + 1) / (hits + 2)   — Laplace-smoothed, computed, not stored

CalTab[space_key] : table[margin_bin × log2(N)_bin] → fcr ∈ [0,1]
  // empirical P(returned cert was wrong | margin, store size); version-stamped;
  // built ONLY by GFR-R probe sweeps (§4 R8). Absent table ⇒ fcr = UNCALIBRATED.

WriteBloom[shard] : bloom filter over content_hash[0:8]
  // sized for FPR ≤ 1% at shard capacity; rebuildable (repair R7)

ResolutionEvent { query_id, path_edges: [u128], returned: [hash16], accepted: hash16?, u: f32 }
```

**Utility signal `u` (the single signal):** precedence order —
explicit caller feedback (`u∈[0,1]`) > task-success proxy (tool call succeeded / answer accepted: `u=1`) > result consumed downstream without complaint (`u=0.5`) > nothing (`u=0`, no ledger touch). Confirmed-wrong feedback (`u=0` explicit + wrong flag) additionally seeds confusion pairs (§2, L4).

---

## 2. GFR-Q — the query pipeline (Mode A: L0–L4, L6; Mode B: L5)

```
QCtx → [pre-filters] → L0 → L1 → L2 → L3 → L4 → L6   (Mode A)
                              seeds ─────→ L5 → L6   (Mode B)
```

### Pre-filters (cheapest checks first, before any ANN work)

Hard exclusions, in order: (1) `space_key` mismatch → type error; (2) facet bitmap intersection (5 independent facet indexes, AND of postings); (3) tombstoned / erased certs (unless `authority.audit`); (4) SEALED-FULL (dark) certs unless authority presents ≥ k shares; (5) `provisional` variants excluded from Mode A unless query opts in (`include_provisional`).

### L0 — Representation gate

**[PROVEN]** `fp = T(state)`, default `transform_id=1` (`rfft.raw.v1`, 230-d, per-frequency-block L2-normed) or `=2` (`rfft+gdf.v1`, 460-d). GDF is an **order-sensitivity** option, not a discriminability upgrade (CAPACITY_RESULTS Part 1): default REFLEX spaces = amplitude-only; spaces needing temporal-order discrimination opt into GDF.

**[PROVEN]** Projection: `z = normalize(P_s · fp)`, `P_s ∈ ℝ^{2048×dims}`, entries `N(0, 1/2048)`, generated deterministically from `proj_seed`. This is the measured capacity fix (8 → 32 @ 0.9) and is the load-bearing default. `z` is cached in the index; it is *derivable*, never stored in the certificate (one matvec to recompute — keeps certs at v1-class sizes).

**Swappable-transform discipline (capstone T5):** everything above L0 assumes only "a fixed-length projected vector in a declared space exists". Changing transform ⇒ new `space_key` ⇒ new index partition ⇒ migration tool. No layer may special-case FFT.

**[HYPOTHESIS — X6]** Domain-of-validity gate (capstone T6): spectral flatness `SFM = geomean(|rFFT|)/mean(|rFFT|)`; suitability `s = 1 − SFM`. If `s < τ_dom (0.15)` the state is too unstructured for spectral fingerprinting → set `domain_warning` on the result (Q path) or route to alternate transform / flag (W path). Kill criterion: if `s` fails to correlate with retrieval error on graph-like/discrete states, drop the gate.

### L1 — Static skeleton [PROVEN]

- **Box:** HNSW (USearch), one index per `space_key`, M=32, efSearch=64. Emit candidate set `C1`, |C1| = K1 = 50. Measured 51–120 µs/query; this layer's only job is a bounded-latency candidate set.
- **Robot:** no ANN. Exact NEON dot-product scan over the hot ring (`etl::vector`, N ≤ 1024), 10–50 µs. Robot profile skips L2/L3 entirely (§7).

### L2 — Physarum conductance flow [PROVEN ELSEWHERE mechanism / HYPOTHESIS aggregation — X1]

The single plasticity mechanism. Absorbs and replaces v2-V0 (STC tags) and v5-T2 (stigmergy): both are special cases of flow-reinforced conductance without Physarum's convergence proof (Tero et al. dynamics; Bonifaci–Mehlhorn–Varma proof of shortest-path convergence for the single-commodity case).

**Subgraph construction:** `G_c` = C1 ∪ { j : ∃ i∈C1, edge (i,j) in flow graph with `D̂ ≥ D_min (0.05)` }, capped at |G_c| ≤ K2 = 64 (drop lowest-D̂ expansions first).

**Conductance semantics:**

```
lazy decay (applied on every read of an EdgeRec):
    D ← D · 2^(−(now − t_last)/T_half)          T_half = 7 days

soft conservation (read-time normalization, no write amplification):
    D̂(i→j) = D(i,j) / Σ_{k ∈ out(i)} D(i,k)     (:= 0 when Σ = 0 → cold start)

edge weight:
    w(i,j) = α·max(0, cos(z_i, z_j)) + (1−α)·D̂(i→j)      α = 0.7, hard floor α ≥ 0.3
```

The `α ≥ 0.3` floor guarantees similarity never vanishes ⇒ **no permanent ruts** even with mistuned decay (structural fix for v5-T2's calcification risk).

**Flow score:** connect a virtual query node `q` to each seed `i∈C1` with `w(q,i) = cos(z_q, z_i)`. With `W` the (sparse) weight matrix over `G_c`:

```
φ = w_q + Wᵀ·w_q          // exact 1- and 2-hop path mass from q; one sparse matvec
r2_i = μ·cos(z_q, z_i) + (1−μ)·φ_i / max(φ)      μ = 0.7
```

**Update rule (executed by GFR-L, async, post-resolution):** for every edge on the path to the accepted result, with utility `u`:

```
D ← (1−η)·D + η·u          η = 0.1     // per-query EMA sample of Tero's dD/dt = f(Q) − D, f = id
t_last ← now
```

Continuous form and proof cover single-source/single-sink; per-query application preserves the local rule exactly. **The additive multi-query aggregate is the unproven part** — flagged, benchmarked in X1 (Physarum vs static HNSW vs stigmergy-only). Cold start `D=0` ⇒ L2 degenerates to pure similarity (safe).

**Robot adaptivity without robot mutability:** the robot never runs L2. The box computes a **HotSet** every sync interval (60 s): top-N certs by `conductance-weighted degree + recency`, shipped into the robot's ring buffer. Physarum reaches the robot only as a snapshot ⇒ robot stays bit-deterministic.

### L3 — Resonance and cleanup [PROVEN ELSEWHERE — Hopfield ≡ attention]

Over the K2-candidate memory matrix `M = [z_i]ᵢ` (≤ 64 × 2048 f32 ≈ 0.5 MB, trivial):

```
x⁰ = z_q
xᵗ⁺¹ = normalize( Mᵀ · softmax(β · M · xᵗ) )      β = 2.0, steps S = 1 (clean cue) … 3 (fragmentary cue)
```

**Honesty correction to Foundation §1.5:** for unit-normalized vectors, `⟨q,x⟩` and `−‖q−x‖²` are affine-equivalent — the proposed "multi-footprint kernel" collapses. The crosstalk suppression the foundation wanted comes from **iterative relaxation + cleanup**, not kernel shape. L3 is therefore specified as fixed-β modern-Hopfield relaxation, nothing more. (Adaptive/learned kernels — U-Hop/A-Hop — stay out: they would break non-negotiable #1. β-modulation by context is a reserved slot: `β = β₀·g(ctx)`, `g ≡ 1` in v2.0.)

**Cleanup [PROVEN — mandatory per CAPACITY_RESULTS]:** snap `x*` to `argmaxᵢ cos(x*, zᵢ)` if ≥ `τ_snap = 0.85`; otherwise return the completed pattern flagged `SYNTHETIC` — a synthesized pattern is **never** presented as a stored memory.

**Ambiguity detector (lineage hook, v5-T0):** if `cos(top1) ∈ [τ_rec=0.70, τ_conf=0.90)` and `cos(top1) − cos(top2) < τ_amb = 0.05` → attach `spawn_recommendation {parent: top1, ops: [perturbation op ids]}` to the result. GFR-Q only *recommends*; spawning happens in GFR-W under the population bound (§3).

### L4 — Negative constraint [PROVEN — unchanged]

`r4_i = r3_i − Σ_{p ∈ NodeRec(i).confusion} pen_p · 1[p relevant to QCtx]`. Confusion lists (≤8 partners, f16 penalties) live in the Node Ledger, populated from (a) GFR-W proofread collisions, (b) confirmed-wrong feedback. The existing 4-stage apophatic logic ports unchanged.

### L5 — Mode B: assertion-graph traversal [HYPOTHESIS — X7b, cheap]

Answers *"what did I (or another agent) explicitly assert is connected, and why"* — structurally separate from similarity. This is the direct answer to the "no causally-grounded retrieval" gap flagged in every geodesic.

- **Input:** seeds = Mode A top-k ∪ explicit hashes; `link_type mask`; direction ∈ {out, in, both}; `depth ≤ 3`; `limit 32`.
- **Traversal:** BFS with visited set (PARENTS acyclic by construction; LINKS may cycle). Each result carries its **path explanation** `[(link_type, hash16), …]` — the "why" is the chain itself.
- **Ordering:** (Lamport clock, depth). Lamport gives causal order for free (v3-V9).
- **Effective-view resolver:** for a core hash → gather ROLE margins referencing it → apply precedence `REDACTION > CORRECTION > CONSOLIDATION > base`; `SUPERSEDES` links redirect; `CONTRADICTS` pairs are **both surfaced, flagged** — never silently resolved. Regime-aware: LOCAL/CP → only epoch-acked non-monotone effects apply; AP → optimistic view + `staleness_note`.

**Fusion:** the response is `{mode_a: […], mode_b: […], overlap: […]}` — labeled, never silently merged. A caller can always distinguish "similar" from "asserted-connected".

### L6 — Distortion-aware calibrated output [PROVEN ELSEWHERE — hallucination floor]

Per result:

```
Result { hash16, score, scs, margin, fcr, flags{synthetic, domain_warning, staleness, provisional},
         distortion_class, path_explanation? }
```

- `fcr` = `CalTab[space_key][margin_bin][⌊log₂N⌋]` — the **empirically measured false-confidence rate** at this operating point. Guo & Li (ICML 2026) prove space-optimal memories must sometimes be confidently wrong; GFR does not pretend otherwise — it **quantifies and surfaces** the rate. No table ⇒ `fcr = UNCALIBRATED`, never fabricated.
- Recommended act-thresholds per `distortion_class`: FULL_FIDELITY act if fcr ≤ 0.02, SUMMARY ≤ 0.05, GIST ≤ 0.15 (policy-overridable).
- GFR never silently abstains; ranked results + fcr are returned, the caller decides.

---

## 3. GFR-W — the write pipeline

Pipe-composed (v3-V3, Adopt): every stage is `Record → Record`, independently swappable, introspectable mid-stream. Stage order is load-bearing — **proofread precedes hashing**, so the proofread record is *inside* the immutable certificate.

```
S1  sample-gate        [hook]      PCM knob #1: WHEN to write. Default = pass-through.
                                   (surprise/ESN gate = deferred policy, slot defined here)
S2  transform+project  [PROVEN]    L0. Assign/verify SpaceDescriptor.
S3  bloom pre-check    [adopt]     WriteBloom hit → exact CAS lookup → true dup ⇒ return existing
                                   hash (+ optional MARGIN "re-observed"). ~0-cost dedup path.
S4  proofread (DNA)    [HYP — X5]  virtual self-query via L1–L4 against existing store:
      c1 = cos to nearest existing cert in space
      (a) c1 ≥ τ_dup  (0.985) ∧ facet/claims compatible   ⇒ DEDUP → margin, no new core
      (b) c1 ≥ τ_coll (0.95)  ∧ incompatible              ⇒ FLAGGED: seed confusion pair
                                                             + auto-LINK CONTRADICTS
      (c) fp degenerate (‖fp‖ anomaly, SFM out of band)   ⇒ REGENERATED: re-window/re-summarize
                                                             hook, one retry, else reject
      (d) else                                            ⇒ PASSED
      record → PROOFREAD section {status, self_retrieval_rank, self_margin, collision_ref}
S5  facet tagging      [adopt]     5 orthogonal facets (ACTOR/ACTION/OBJECT/CONTEXT/TIME —
                                   Ranganathan PMEST mapping, Cert §4.3). Controlled vocab ids.
S6  clock              [adopt]     lamport = 1 + max(lamport of PARENTS ∪ causal LINKS targets
                                   ∪ session predecessor, 0). Vector-clock bump if multi-agent.
S7  link proposal      [training-free rules ONLY]
                                   session-predecessor ⇒ CONTINUATION; declared tool derivation
                                   ⇒ DERIVED_FROM; S4(b) ⇒ CONTRADICTS; caller-supplied links
                                   pass through. Semantic link inference = external margin-writing
                                   process later — honest boundary, not smuggled into the core.
S8  distortion_class   [hook]      PCM knob #2: HOW MUCH detail. Default FULL_FIDELITY.
                                   (Rate-distortion policy = deferred; class field is the seam.)
S9  hash + commit      [PROVEN ELSEWHERE]
                                   canonical-serialize → BLAKE3-256 → CAS append (fsync) →
                                   index insert(z) → bloom add → NodeRec init.
                                   Crash between CAS and index is legal: repair R1 heals it.
```

**Lineage spawning (v5-T0, [HYPOTHESIS — X7]):** consumed here, never in Q. On a `spawn_recommendation`: create ≤ k variants via registered perturbation operators `{op_id, seed}` (deterministic, reproducible — Cert §4.6), each a full cert with LINEAGE section, `provisional` flag set. **Population bound:** live variants per `lineage_root ≤ P_max = 4` (enforced by repair R4 using ledger fitness). Promotion (clear `provisional`) and retirement are repair actions, expressed as MARGIN certs — cores are never edited.

---

## 4. GFR-R — self-stabilizing repair (Dijkstra, v3-V0)

The store is modeled as a distributed program; GFR-R is the set of **local, uniform, idempotent correction rules** that provably converge to the legal predicate **L** from *any* starting state — including adversarial ones (prompt-injected certs are just certs; they get quarantined by L2/L3 below, not special-cased).

**Legal state predicate L over Σ = (CAS, Index, Ledger, Bloom):**

| Clause | Statement |
|---|---|
| **L1 bijection** | every index entry → live, unerased cert; every live non-dark cert → indexed in its space |
| **L2 provenance** | every cert's PARENTS-closure reaches a TrustRoot, else cert carries ORPHAN flag (envelope) |
| **L3 contradiction hygiene** | certs asserting the same *monitored key* (facet-equal + claim-hash-equal scope) are pairwise compatible ∨ CONTRADICTS-linked ∨ margined. Semantic contradiction beyond monitored keys = out of scope (honest boundary) |
| **L4 lineage bound** | live variants per lineage ≤ P_max |
| **L5 ledger sanity** | D ∈ [0, D_max]; every NodeRec/EdgeRec references existing certs |
| **L6 erasure** | `payload_erased` ⇒ payload bytes zeroed in CAS and replicas |
| **L7 bloom soundness** | bloom ⊇ committed hashes (over-approx allowed; under-approx ⇒ rebuild) |

**Local rules (guard → action), round-robin over shards, fair scheduling, safe from any state:**

| Rule | Guard → action | Targets | Variant function (strictly ↓) |
|---|---|---|---|
| R1 | dangling index entry → drop; unindexed live cert → insert | L1 | # L1 mismatches |
| R2 | unrooted parent-closure → set ORPHAN (envelope flag; monotone) | L2 | # unflagged orphans |
| R3 | monitored-key conflict, unlinked → write CONTRADICTS margin pair | L3 | # unflagged conflicts |
| R4 | lineage live-count > P_max → retire lowest fitness (margin) | L4 | Σ max(0, live−P_max) |
| R5 | D ∉ range → clamp; orphan ledger rec → GC | L5 | # ledger violations |
| R6 | erased flag ∧ payload nonzero → zero payload | L6 | # unenforced erasures |
| R7 | shard checksum mismatch → rebuild bloom from CAS scan | L7 | # unsound blooms |
| R8 | (scheduled, not L-driven) probe-sweep → refresh CalTab | feeds Q-L6 | — |

**Convergence argument (sketch, per v3-V0's "moderate proof effort" stance):** each rule strictly decreases a non-negative integer violation count for its clause; a pairwise interference table (maintained with the code) shows no rule increases another clause's count — R2/R3/R4 only *add* flags/margins (monotone actions), R1/R5/R6/R7 only remove/normalize derived state. Closure: when Σ ⊨ L, all guards are false ⇒ fixpoint. Full mechanized proof (TLA+/model checking on a reduced model) = backlog item, not claimed.

---

## 5. GFR-L — the ledger service

Async consumer of `ResolutionEvent`s (bounded queue; drop-oldest under pressure — adaptivity is best-effort by design).

- For each edge on the accepted path: lazy-decay → EMA update (`§2 L2`) → `t_last = now`.
- NodeRec: `hits += 1` for all returned; `wins += 1` if `u ≥ 0.5` for accepted.
- Confirmed-wrong (`u = 0` + wrong flag): add/strengthen confusion pair `(returned_wrong, actually_right)` with penalty `pen = min(pen + 0.1, 1.0)`.
- Edge ledger capacity: LRU-capped (default 4 M edges ≈ 100 MB); eviction of a cold edge = D reset, harmless.
- **Invariant restated:** deleting the entire ledger returns the system to cold-start retrieval quality and touches nothing else.

---

## 6. Consistency model — the CRDT move [PROVEN ELSEWHERE: CRDT + CALM]

**The store is a grow-only set (G-Set) of immutable, content-addressed certificates.** This is a textbook state-based CRDT: replication and multi-agent federation reduce to **set union** — commutative, associative, idempotent, coordination-free, eventually consistent. Certificate exchange between agents/robots is literally "gossip a G-Set". (This dissolves most of v2-V8's federation problem at the storage layer; identity-shape/namespace reconciliation remains a deferred policy above it.)

**All mutation is reinterpretation.** Supersession, correction, redaction, retirement are *margin certificates* (appends = monotone). Per CALM: monotone ops need no coordination. The only non-monotone thing left is **effectiveness** of a retraction/uniqueness claim — handled by interpretation epochs:

| Regime | Non-monotone effects (supersede/redact/unique) | Reads |
|---|---|---|
| LOCAL (single node) | immediate (trivial epoch) | exact |
| AP (default robot↔box) | applied optimistically | + `staleness_note`, bounded by last sync |
| CP (compliance-critical, e.g. erasure) | applied only after epoch ack from quorum | may refuse under partition |

CAP declaration per deployment is therefore explicit and per-*operation-class*, not hand-waved: appends are always available; retraction-effectiveness picks CP or AP deliberately (Cert §4.11 `REGIME` section).

---

## 7. Profiles and budgets

| | **GFR-R(obot)** — APQ8009, C++/NEON | **GFR-B(ox)** — Go + C++ cores |
|---|---|---|
| Layers | pre-filters → L1(exact) → L4 → L6(snapshot CalTab) | full L0–L6 + W/R/L services |
| Adaptivity | none at runtime; HotSet snapshot sync (60 s) | full ledger |
| Determinism | bit-deterministic per snapshot | deterministic per ledger state |
| L0 | ~5 ms fast path (PFFFT, DC-first) | full precision, ~50–100 ms tolerable |
| L1 | 10–50 µs exact scan, N ≤ 1024 | 51–120 µs HNSW |
| L2 | — | ≤ 64-node subgraph, 1 sparse matvec, ~20 µs |
| L3 | — | S ≤ 3 × (64 × 2048) matmuls ≈ 0.8 MFLOP, ~50 µs |
| L5 | — | BFS ≤ 32 nodes, ~10 µs |
| End-to-end | < 1 frame budget (33 ms) incl. L0 | ≤ 2 ms p50 excl. I/O; ledger async |
| Memory | ring ≤ 1024 × cert ≈ 1 MB + code | index + ledger caps configured |

Language mapping unchanged from ENGRAM 1.0 report: Python = executable reference spec + golden vectors; C++ = robot + compute cores; Go = box service. Cross-language parity: fp/z at rel-L2 ≤ 1e-5; canonical serialization and hashes **bit-exact**.

---

## 8. Failure modes → degradation ladder

| Loss | Effect | Recovery |
|---|---|---|
| Edge/Node Ledger | cold-start retrieval quality | rebuilds from traffic; nothing else touched |
| HNSW index | slow path | rebuild from CAS (streaming re-insert); exact-scan fallback meanwhile |
| WriteBloom | dedup misses (extra exact lookups) | R7 rebuild |
| CalTab | `fcr = UNCALIBRATED` on results | R8 probe sweep |
| CAS shard | data loss — the only real loss | replication (G-Set union) / SEALED shares where configured |
| Partition (robot↔box) | AP reads with staleness notes; CP ops queue | resync = set union + epoch replay |

---

## 9. Parameter registry (defaults; every value tunable, every hypothesis killable)

| Param | Default | Layer | Bound experiment |
|---|---|---|---|
| dims / proj_dim | 230 or 460 / 2048 | L0 | [PROVEN] |
| τ_dom | 0.15 | L0 | X6 |
| K1 / K2 | 50 / 64 | L1/L2 | X1 |
| α (sim floor 0.3) / μ | 0.7 / 0.7 | L2 | X1 |
| η / T_half / D_min / D_max | 0.1 / 7 d / 0.05 / 8.0 | L2/L | X1 |
| β / S / τ_snap | 2.0 / 1–3 / 0.85 | L3 | X3 |
| τ_rec / τ_conf / τ_amb | 0.70 / 0.90 / 0.05 | L3 | X7 |
| confusion cap / pen step | 8 / 0.1 | L4/L | — |
| depth / limit (Mode B) | 3 / 32 | L5 | X7b |
| τ_dup / τ_coll | 0.985 / 0.95 | W-S4 | X5 |
| P_max | 4 | W/R4 | X7 |
| bloom FPR / ledger cap | 1 % / 4 M edges | W/L | — |
| HotSet size / sync | 512 / 60 s | robot | X1b |

---

## 10. Validation plan and kill criteria

**Golden-vector discipline (unchanged from 1.0 report):** Python reference generates canonical inputs → fp/z (rel ≤ 1e-5 cross-language), canonical bytes + BLAKE3 (bit-exact), retrieval top-1 parity on soak suite.

**Falsification experiments (each an afternoon-scale minimum test, per capstone T9):**

| ID | Tests | Kill criterion |
|---|---|---|
| X1 | Physarum conductance vs static HNSW vs decay-only, skewed workloads | < 5 % recall-or-latency win at matched compute ⇒ set α = 1 permanently (L2 becomes pass-through) |
| X1b | HotSet-synced robot ring vs recency-only ring | no reflex-hit-rate gain ⇒ recency-only sync |
| X2 | rFFT vs WHT vs DCT vs scattering (+projection each) on real states: correlation + capacity | if an alternative cuts mean \|cos\| materially at equal cost ⇒ new default transform (space migration) |
| X3 | L3 relaxation+cleanup vs pure cosine-kNN on noisy/composed/masked cues | no completion gain ⇒ S = 0 (kNN-only), keep cleanup |
| X4 | Partition simulation: AP vs CP interpretation under robot↔box splits | correctness violations in AP beyond staleness bound ⇒ tighten regime defaults |
| X5 | Proofread: inject malformed/contradictory writes; detection at S4 vs at repair | S4 catches < 70 % of what R3 eventually catches ⇒ demote proofread to repair-only |
| X6 | Domain gate: SFM vs retrieval error on graph/discrete/code states | no correlation ⇒ drop gate, keep domain-of-validity as documentation |
| X7 | Lineage variants vs scalar reweighting under query drift | no adaptation-speed win ⇒ shelve lineage (section stays reserved) |
| X7b | Mode B typed-link traversal vs similarity-only on contradiction-discovery tasks | finds nothing similarity misses ⇒ demote LINKS to metadata |

---

## 11. Deviations from Foundation v2 §1 (deliberate, argued)

1. **L3 "multi-footprint kernel" removed** — collapses under unit normalization (affine equivalence of dot and neg-sq-dist). Replaced by what actually does the work: relaxation + cleanup.
2. **Stigmergy fully absorbed** into the Physarum EMA (foundation hinted at this; here it is normative — one plasticity mechanism, one signal).
3. **Robot adaptivity via box-computed HotSet snapshots** — foundation left robot/adaptivity tension unresolved; this keeps the robot bit-deterministic while letting flow shape it.
4. **Store reframed as a G-Set CRDT** — makes federation set-union, makes CALM classification exact (appends monotone, effectiveness non-monotone), and gives the self-stabilization closure of the storage layer for free.
5. **Lineage fitness moved out of the certificate** into the Node Ledger — a mutable score inside an immutable record was a spec contradiction in the foundation table.
6. **Proofread ordered before hashing** so its audit record is inside the immutable cert (foundation implied post-hoc write, which would have required mutation).
7. **PARENTS split from LINKS** — Merkle derivation (acyclic, provenance-bearing) and lateral assertions (typed, cyclable) have different invariants; conflating them breaks L2's closure argument.

## 12. Deferred consumers (slots exist, nothing built here)

COMPOSE/self-shape engine (reads projected space + cleanup), consolidation/eviction engine (reads conductance + `distortion_class`), write-gating control loop (S1 hook; Nyquist/R(D) formulation per capstone T4), Titans-style auxiliary re-ranker (outside GFR, post-L6), Actor-model multi-agent isolation, full Shamir topology, semantic link inference. Each plugs into a named seam above; none is required for the foundation to run.
