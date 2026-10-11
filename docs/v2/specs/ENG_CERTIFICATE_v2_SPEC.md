# `.eng` v2 — EGRV2 Certificate Format
## Cornerstone B: the ENGRAM memory record — full data-schema specification
### Companion to `GFR_v2_ALGORITHM_SPEC.md` (Cornerstone A). Designed as one coupled pair.
### Status: engineering spec, 2026-08-18 · supersedes the sketch in `engram-foundation-v2-algorithm-and-certificate.md` §2

---

## 0. Design axioms

1. **Immutable body, mutable envelope.** The hashed **BODY** never changes after commit. Every piece of state that *must* change (repair flags, tombstones, index bookkeeping) lives in a separate, unhashed **ENVELOPE** stored beside the body. This resolves the contradiction in the foundation sketch, where repair actions (ORPHAN flags, retirement) would have mutated an immutable record.
2. **Correction is appending, not editing.** Supersession, contradiction, redaction, retirement, consolidation are all new certificates of `role = MARGIN` pointing at an untouched `role = CORE`. (Talmudic layering, v5-T5.) Consequence: the store is a **grow-only set** → CRDT semantics → coordination-free replication (GFR §6).
3. **Content-addressed.** `hash = BLAKE3-256(canonical BODY)`. Identical bodies collapse to one object — deduplication is a property of the format, not a subsystem. (Merkle/Git, v3-V4.)
4. **Core section is byte-identical to v1.3.** A v1.3 reader must parse a v2 certificate's core and safely skip the tail. Everything new is a **type-tagged, length-prefixed, independently-skippable section**.
5. **Derivable data is not stored.** The 2048-d projected vector `z` is one matvec from `fp` and lives only in the index. Ledger state (conductance, hits/wins/fitness) is never a certificate field.
6. **Every mutable scalar someone wants to put in here is a design smell.** If a field would be written more than once, it belongs in the Ledger or in a MARGIN certificate. This rule is what keeps §0.1–0.3 true.
7. **Honesty tiers:** **[PROVEN]** measured in ENGRAM · **[PROVEN ELSEWHERE]** formal result in source domain, unvalidated here · **[HYPOTHESIS]** design bet with a kill criterion.

---

## 1. Container, canonical encoding, and hashing

Two encodings, deliberately separated — this is the single most important structural decision in the format:

| | **Canonical form (hashed)** | **Container form (stored/shipped)** |
|---|---|---|
| Encoding | deterministic TLV: sections ascending by `type_id`, no padding, fixed-width LE ints, no optional-field ambiguity | FlatBuffers `EGRV2` (the Anki way — direct precedent in Vector's animation binaries) |
| Purpose | defines `content_hash` | zero-copy `mmap` reads on ARM, `flatc` codegen for C++/Go/Python |
| Rule | one body ⇒ exactly one canonical byte string ⇒ one hash | container may be re-encoded, re-packed, or re-ordered freely |

FlatBuffers is **not** canonical (vtable layout and builder order vary), so hashing the container bytes would be a correctness bug. Hash the canonical TLV; store the container. Both C++ and Go emit the canonical form through the same code path, validated bit-exact against the Python reference (golden vectors).

```
hash256    = BLAKE3-256(canonical_body)      // full identity, used in PARENTS/links/audit
hash16     = hash256[0:16]                   // in-memory refs, ledger keys, bloom
file layout: [EGRV2 magic 'E','G','R','2'][u16 container_ver][BODY (FlatBuffers)][ENVELOPE (separate region/file)]
```

Rationale for BLAKE3 over SHA-256: tree-hashable (cheap Merkle over sections), ~5–10× faster on ARM, no length-extension footgun. Domain-separated: `BLAKE3(0x45 ‖ "engram.body.v2" ‖ canonical_body)`.

---

## 2. `SpaceDescriptor` — the type system for cosine space

The hard partition that makes "two senses never cross-compared" a **type error** rather than a convention.

```fbs
table SpaceDescriptor {
  sense           : Sense;    // REFLEX | MEANING | (future senses)
  transform_id    : uint16;   // 1 = rfft.raw.v1 (230-d) · 2 = rfft+gdf.v1 (460-d) · 3.. reserved
  dims            : uint16;   // pre-projection dimensionality
  proj_dim        : uint16;   // 2048 default
  proj_seed       : uint64;   // deterministic generation of P_s
  norm_scheme     : uint8;    // per-frequency-block L2 (v1 default)
  source_kind     : uint8;    // KV_CACHE | FUSED_PERCEPTION | TEXT_EMBED | TOOL_STATE | ...
  source_cfg_hash : uint64;   // sensor suite / model family / embedder version
}
space_key = BLAKE3(canonical(SpaceDescriptor))[0:8]
```

`space_key` is carried in the core header of every certificate and is the first thing GFR checks. Changing a transform, a projection seed, an embedder version, or a sensor suite ⇒ new `space_key` ⇒ new index partition ⇒ explicit migration. Nothing silently drifts into comparing incomparable vectors.

**Why this exists:** the transform question is genuinely open (capstone T5: FFT-magnitude may not be the right primitive; experiment X2 will test WHT/DCT/scattering). A space-typed store makes swapping the primitive a *migration*, not a rewrite — and lets two transforms coexist during evaluation.

---

## 3. CORE section (mandatory, v1.3-compatible)

```fbs
enum Precision : byte { F32 = 0, F16 = 1, I8 = 2 }
enum Role      : byte { CORE = 0, MARGIN = 1 }
enum Sense     : byte { REFLEX = 0, MEANING = 1 }

table CertCore {
  // --- identity & space -------------------------------------------------
  format_ver     : uint16 = 0x0200;   // 2.0
  space_key      : uint64;            // §2 — hard partition
  role           : Role;
  timestamp_us   : uint64;            // wall clock at commit (NOT an ordering source; see §4.4)
  session_id     : uint64;
  label          : string;            // optional symbolic label / namespace
  // --- the fingerprint (v1.3 layout, unchanged) ------------------------
  dims           : uint16;
  precision      : Precision = F16;
  data           : [ubyte] (required, force_align: 16);  // fp bytes, NEON-aligned
  joint_center   : [ubyte];           // embedded 128-d basis — self-contained fold-in, unchanged
  // --- confidence (v1.3, unchanged) ------------------------------------
  scs            : float;             // Semantic Coverage Score
  retrieval_margin : float;
  // --- v1.3 embodiment tail (unchanged, optional-empty for non-embodied) 
  robot_id       : string;
  sensor_config  : uint64;
  activity       : uint16;
  place          : uint32;
  emotion_state  : [float];           // valence, arousal, engagement, frustration
  // --- integrity --------------------------------------------------------
  crc32          : uint32;            // cheap corruption check, independent of hash
}
```

Size: ~856 B at `dims=230, F16` — **unchanged from v1.2/v1.3**, which is the point. All 220 existing tests, the round-trip suite, and the 6-LLM-family compatibility results remain valid for the core.

**Deliberately absent from CORE:** projected vector `z` (derivable), conductance (ledger), hits/wins/fitness (ledger), any flag repair might set (envelope).

---

## 4. Optional sections

Each section: `[u16 type_id][u32 byte_len][payload]`. Unknown `type_id` ⇒ skip by length. Sections appear at most once, ascending by `type_id` in canonical form.

| id | Section | Tier | Purpose |
|---|---|---|---|
| 0x10 | PARENTS | Proven elsewhere | Merkle derivation DAG |
| 0x11 | LINKS | Hypothesis (X7b) | typed lateral assertions |
| 0x12 | ROLE_MARGIN | Hypothesis | margin semantics + target |
| 0x13 | FACETS | Adopt | 5 orthogonal query axes |
| 0x14 | CLOCK | Proven elsewhere | Lamport / vector clock |
| 0x15 | PROOFREAD | Hypothesis (X5) | write-time self-check audit |
| 0x16 | LINEAGE | Hypothesis (X7) | variant ancestry |
| 0x17 | DISTORTION | Proven elsewhere | fidelity class declaration |
| 0x18 | REGIME | Proven elsewhere | CAP/CALM operation class |
| 0x19 | SEALED | Proven elsewhere | Shamir share payload |
| 0x1A | ECC | Hypothesis, unproven | Hamming/RS overlay on VQ codes |
| 0x1B | ERASURE_PROOF | — | integrity survival after payload zeroing |

### 4.1 PARENTS (0x10) — derivation DAG

```
parents : [hash256]        // certificates this one was derived FROM
trust_root_id : uint32?    // set only on TrustRoot certs (self-rooted)
```
**Acyclic by construction** (a parent must exist and be hashed before the child can be hashed). Supports GFR-R rule R2 (provenance closure → ORPHAN flag) and gives verifiable, tamper-evident lineage: any change anywhere in an ancestry chain changes every descendant hash.

### 4.2 LINKS (0x11) — typed lateral assertions

```
links : [{ target : hash256,
           type   : LinkType,
           rationale_ptr : hash256?,   // optional cert holding the "why" — the why is itself a memory
           created_by : uint32 }]
```

```
LinkType = CONTINUATION | DERIVED_FROM | CONTRADICTS | SUPERSEDES
         | REFINES | SURPRISING_CONNECTION | ANSWERS | INVALIDATED_BY
```

May contain cycles (`CONTRADICTS` is symmetric in practice); traversal is BFS with a visited set. This is the **assertion graph** — Mode B (GFR §2 L5) traverses it and returns the path as the explanation. It is the concrete answer to "no causally-grounded retrieval", flagged unresolved in v2, v3, v4 and v5 alike.

**Training-free link creation only** (GFR §3 S7): session-predecessor, declared derivation, proofread collision, caller-supplied. Semantic link inference is an external process that writes MARGIN certs later — never smuggled into the core write path.

### 4.3 FACETS (0x13) — Ranganathan orthogonality

```
actor_id      : uint32   // PERSONALITY — who
action_type   : uint16   // ENERGY      — what happened
object_hash   : uint64   // MATTER      — what it was about
context_id    : uint64   // SPACE       — session / place / harness
time_bucket   : uint32   // TIME        — coarse bucket, NOT the timestamp
```

Five independently-indexed axes; queries are bitmap intersections, so no combination index is ever built. Vocabulary ids come from a controlled, versioned table (`vocab_ver` in the envelope of the space registry, not per-cert).

**Discipline that makes it real:** repair R-audit periodically measures pairwise mutual information between facets; MI above threshold = schema smell, logged for redesign. Orthogonality is *monitored*, not structurally enforced — exactly Ranganathan's own practice.

### 4.4 CLOCK (0x14) — causal, not chronological

```
lamport   : uint64
vclock    : [{ agent_id : uint32, counter : uint64 }]?   // multi-agent only
```
`lamport = 1 + max(lamport of PARENTS ∪ causal LINKS ∪ session predecessor)`. **Wall-clock `timestamp_us` in CORE is metadata; `lamport` is the ordering source.** This distinguishes "these happened near each other" from "this could have influenced that" — the distinction similarity- and recency-based retrieval structurally cannot make (v3-V9).

### 4.5 PROOFREAD (0x15) — DNA-polymerase analogue

```
status              : PASSED | REGENERATED | FLAGGED | DEDUP_MARGIN
self_retrieval_rank : uint16     // rank of self when querying with own content
self_margin         : float      // cos gap to nearest unrelated cert
collision_ref       : hash256?   // the cert it nearly collided with
transform_id_used   : uint16
```
Written *before* hashing, so the audit is inside the immutable record — the certificate carries its own birth certificate. Costs nothing new to build: it is GFR-Q's L1–L4 run once against the store at write time (v5-T1).

### 4.6 LINEAGE (0x16) — affinity maturation

```
lineage_root  : hash256
generation    : uint16          // 0 for non-variants
parent_variant: hash256?
op_id         : uint16          // registered perturbation operator
op_seed       : uint64          // reproducible: variant = op(parent, seed)
```
**Fitness is deliberately absent** — it is `(wins+1)/(hits+2)` in the Node Ledger, because it changes constantly and a mutable field here would break immutability. Live variants per `lineage_root` are capped at `P_max = 4` by repair R4; promotion/retirement are MARGIN certs (v5-T0).

### 4.7 ROLE_MARGIN (0x12)

```
core_id     : hash256
margin_type : CONSOLIDATION | CORRECTION | CONTRADICTION_FLAG | REDACTION
            | RETIREMENT | PROMOTION | RE_OBSERVATION | ANNOTATION
produced_by : uint32           // agent / process id
```
Precedence when GFR resolves an effective view: `REDACTION > CORRECTION > CONSOLIDATION > base`. `CONTRADICTION_FLAG` never resolves — both sides surface, flagged. Margins can themselves carry margins (commentary on commentary), which is the whole point of the Talmudic model.

### 4.8 DISTORTION (0x17)

```
class          : FULL_FIDELITY | SUMMARY | GIST
source_budget  : uint32?   // bytes of source material this compresses
```
The seam for rate-distortion policy (capstone T0/T1, arXiv:2607.08032 / 2605.10870). GFR-L6 uses the class to pick the recommended act-threshold on `fcr`. The optimal-policy machinery is deferred; the field that policy will read exists now.

### 4.9 REGIME (0x18)

```
regime         : LOCAL | AP | CP
monotone       : bool      // true = pure append (coordination-free per CALM)
epoch_required : uint64?   // for non-monotone effects: epoch at which this takes effect
```
Per-certificate, per-operation-class CAP declaration. Appends are monotone → always available. Retraction/uniqueness effectiveness is non-monotone → declares CP (quorum-acked) or AP (optimistic + staleness note).

### 4.10 SEALED (0x19) — Shamir threshold payload

```
share_index : uint16
threshold_k : uint16
total_n     : uint16
field_id    : uint8        // GF(2^8) / GF(2^128)
share_value : [ubyte]      // replaces payload for this holder
scope       : PAYLOAD_ONLY | FULL
```
`PAYLOAD_ONLY`: header/facets/links remain readable (searchable but unreadable). `FULL` ("dark"): only the hash and a marker exist locally — invisible to retrieval without ≥ k shares. Conditional: worth the complexity only with a genuine multi-location topology, per v5-T4's own caveat.

### 4.11 ECC (0x1A) — explicitly unproven, pluggable

```
scheme     : HAMMING_SECDED | REED_SOLOMON | NONE
codeword   : [ubyte]      // over VQ/RVQ code indices, not over float dims
min_dist   : uint8
```
Present so that Hamming-style correction radius (v3-V1) is A/B-testable without a format change. Default `NONE`. Applies to quantized codes only — continuous embedding drift is not a bit-flip channel, and pretending otherwise is exactly the failure mode the archaeology geodesic warned about.

### 4.12 ERASURE_PROOF (0x1B)

Written when a payload is destroyed for compliance: keeps `original_hash256` + `erased_at_epoch` + `authorized_by`, so the Merkle chain and every descendant hash stay verifiable after the bytes are gone. Erasure removes content, never provenance.

---

## 5. ENVELOPE (mutable, unhashed, stored beside the body)

```
cert_hash        : hash256           // the join key
flags            : bitfield { ORPHAN, TOMBSTONED, PAYLOAD_ERASED, PROVISIONAL,
                              QUARANTINED, INDEX_PENDING, DOMAIN_WARNING }
shard_id         : uint32
index_state      : NONE | INSERTED | REBUILDING
last_repair_pass : uint32
replica_epoch    : uint64
```
Rules: envelope flags are **monotone within a repair epoch** (set, not toggled) so repair rules stay convergent; the envelope is fully **rebuildable** from CAS + repair rules — losing it costs one repair pass, never data. No retrieval-quality state ever lives here (that is the Ledger). No semantic content ever lives here (that is a MARGIN cert).

---

## 6. Invariants (the certificate half of legal-state predicate **L**)

| # | Invariant | Enforced by |
|---|---|---|
| C1 | Body bytes never change after commit | write path; verified by `hash(canonical_body) == stored hash` |
| C2 | `content_hash` collision ⇒ same body (dedup is automatic) | BLAKE3 + canonical form |
| C3 | PARENTS is acyclic | hash-before-reference ordering |
| C4 | Every non-root cert's parent closure reaches a TrustRoot, else ORPHAN | repair R2 |
| C5 | `role = MARGIN` ⇒ ROLE_MARGIN present with a resolvable `core_id` | write validation |
| C6 | All fingerprint comparisons share `space_key` | GFR pre-filter (type error) |
| C7 | Live variants per `lineage_root` ≤ P_max | repair R4 |
| C8 | `PAYLOAD_ERASED` ⇒ payload bytes zeroed everywhere + ERASURE_PROOF present | repair R6 |
| C9 | Facet pairwise MI below threshold (monitored) | repair R-audit |
| C10 | Canonical serialization is byte-identical across C++/Go/Python | golden vectors, CI |

---

## 7. Size budget

| Configuration | Approx. bytes |
|---|---|
| CORE only, `dims=230`, F16 (= v1.2 baseline) | ~856 |
| + v1.3 embodiment tail | ~900 |
| + PARENTS(1) + CLOCK + FACETS + PROOFREAD | ~1,010 |
| + LINKS(3) + ROLE_MARGIN + DISTORTION + REGIME | ~1,180 |
| GDF space (`dims=460`) core | ~1,320 |
| Typical MARGIN cert (no fingerprint payload) | ~200–300 |

At 30 fps continuous embodied writes with typical sections: ≈ 35 KB/s ≈ 3 GB/day before dedup/gating — which is precisely why the S1 sample-gate and DISTORTION class seams exist, and why the bloom dedup path is on the hot write path.

---

## 8. Versioning and migration

- `format_ver` in CORE; new capability ⇒ **new section type_id**, never a changed core field.
- Readers MUST skip unknown sections by length; writers MUST NOT reorder canonical sections.
- **v1.x → v2 migration:** Python tool reads EGRV v1, re-emits canonical v2 body (core only), computes `content_hash`, verifies CRC, writes container. Golden vectors generated from v1 and verified against v2 reads (bit-exact core, per the ENGRAM 1.0 report's migration discipline).
- **Space migration** (e.g. if experiment X2 replaces the transform): new `space_key`, dual-write during evaluation, index partitions coexist, old space frozen not deleted. Certificates are never rewritten — a new space means new certificates and a MARGIN link (`DERIVED_FROM`) from new to old.

---

## 9. How the two cornerstones feed each other

| Certificate provides | GFR consumes it in |
|---|---|
| `space_key` | pre-filter (type check), index partition selection |
| `data` + `joint_center` | L0 (fingerprint → projection) |
| FACETS | pre-filter bitmap intersection |
| PARENTS + LINKS | the assertion graph: L2's expansion set, L5's traversal |
| CLOCK | L5 ordering, contradiction precedence |
| ROLE_MARGIN | L5 effective-view resolution |
| DISTORTION | L6 act-threshold selection |
| REGIME | L5/L6 staleness handling, CP refusal under partition |
| SEALED | pre-filter visibility gate |
| ENVELOPE flags | pre-filter exclusions (tombstoned, quarantined, provisional) |

| GFR produces | Written into |
|---|---|
| write-time self-retrieval result | PROOFREAD section (before hashing) |
| ambiguity detection at L3 | LINEAGE section of spawned variants |
| proofread collision | confusion pair (Ledger) **+** auto `CONTRADICTS` link |
| repair rule outcomes | ENVELOPE flags **+** MARGIN certificates |
| conductance, hits/wins, confusion | Ledger only — **never** a certificate field |

Neither document is complete alone: the certificate defines the graph the algorithm flows over; the algorithm's outputs are the fields the certificate was designed to hold.

---

## 10. Open questions carried honestly forward

1. **Monitored-key scope for contradiction (C-L3).** Currently facet-equal + claim-hash-equal. Genuine semantic contradiction detection is out of scope and stated as such.
2. **Facet vocabulary evolution.** Versioned tables are specified; migration of live postings when a vocabulary splits is unsolved.
3. **Perturbation operator design (LINEAGE).** Biology constrains mutation by structural validity; embeddings have no equivalent constraint. Operators are registered and reproducible, but "meaningful neighborhood" is unproven (X7).
4. **ECC over continuous drift.** Only defensible over quantized codes; whether that captures the failure mode that actually matters is untested (X-ECC, unscheduled).
5. **Erasure vs holographic bundling.** Once a fingerprint has been bundled into a composed identity vector, certificate-level erasure does not remove it from the bundle. This is a known, unsolved VSA limitation and a real legal exposure — flagged here rather than papered over; the current mitigation is scope discipline (keep sensitive modalities in separate, unbundled certificates).
6. **Whether FFT-magnitude is the right primitive at all** (capstone T5). The format's answer is architectural rather than substantive: `space_key` makes the question testable and the answer swappable.
