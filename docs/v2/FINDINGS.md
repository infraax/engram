# ENGRAM — Findings and Corrections

**v1 post-mortem + v2 milestone M0** · 2026-10-11 · branch `v2` · Dexter Bitel (Λ)

Every number in this document comes from a run file in [`results/v2/2026-10-11/`](../../results/v2/2026-10-11/) (git SHA, host, config hash and full metrics per run), produced by the `engram_bench` harness in this branch. Full tables: [`results/M0_RESULTS_TABLES.md`](results/M0_RESULTS_TABLES.md). Narrative results: [`results/M0_RESULTS.md`](results/M0_RESULTS.md).

---

## TL;DR

- **v1's headline result measured identity, not retrieval.** The "98 % Recall@1" matched a document's fingerprint against fingerprints *of the same text*. No query was ever a question, a paraphrase or any other text.
- **On real semantic retrieval, KV-cache fingerprints lose clearly to small text embedders.** Best KV result on BEIR SciFact: nDCG@10 **0.252** vs **0.706** for BGE-small (33 M parameters). Same picture on 3 models from 2 families (0.5 B – 1.7 B).
- **v1 Fourier fingerprints are almost all shared offset.** Mean |cos| between *unrelated* documents: **0.94 – 0.98**. The v1 margins (~0.007) were the small remainder on top of that offset.
- **v1 per-document SVD vectors cannot be compared across documents.** R@1 ≤ 0.025 on every task.
- **What holds:**
  - The KV fingerprint is an excellent **identity key**: id-R@1 = **1.00** under 7 text perturbations, computed for free during prefill.
  - **Removing the top shared principal directions** is the lever that makes KV geometry discriminative. Question R@1 rises ×1.4 – ×2.5 over centering alone, and ×2.4 – ×3.5 over the v1 fingerprint.
  - **Inverse RoPE is exact** (cos ≥ 0.999998 against hooked pre-RoPE keys).
- **Decision (D-010, final for M0):** the KV fingerprint becomes the identity / dedup / restore key. Semantic retrieval goes to a dedicated text embedder (the MEANING sense). v2 focuses on a **portable, verifiable KV-state certificate**.

---

## 1. v1 claims and their status

Status legend: **Holds** · **Narrowed** (true only in a smaller sense than claimed) · **Disproven** · **Unverified** (not re-measured).

| # | v1 claim | Where | Status | Evidence |
|---|---|---|---|---|
| C1 | "Persistent cross-session **semantic** retrieval" from KV fingerprints | README L3, paper abstract | **Disproven** as stated | §3, H5; SciFact 0.252 vs 0.706 |
| C2 | f0+f1 Fourier fingerprint: **98 % Recall@1** at N = 200 | paper Tab. 1 / Tab. 8, README "Recall@1 100 %" | **Narrowed** — identity matching of the same text | E-09 |
| C3 | 4-stage geodesic pipeline reaches **100 % recall** | paper abstract | **Narrowed** — same protocol as C2 | E-09 |
| C4 | Fingerprint is **architecture-invariant** | paper abstract, contributions | **Narrowed** — same-text identity across Llama 3.2 3B ↔ 3.1 8B only (same tokenizer, same KV width) | E-07 |
| C5 | Fingerprint is **corpus-independent** | paper abstract | **Disproven** — usable discrimination needs a mean / PC basis fitted on a calibration corpus | E-02, §3.3 |
| C6 | Margin follows a power law with **"no collapse point"** | paper abstract, §scaling | **Unverified / unsupported** — fit over N ≤ 200 | E-06 |
| C7 | Keys are **pre-RoPE** | docs, patent text | **Wrong** — llama.cpp stores post-RoPE keys | E-05 |
| C8 | Per-document SVD + `gate_start=6` is the strongest same-model discriminator | `docs/PAPER_STRUCTURE.md` | **Disproven** | E-10 |
| C9 | Cross-model transfer via FCDB, +0.124 margin | paper abstract | **Narrowed** — paper's own data: N = 50, collapses at N = 100 | paper §Cross-Model |
| C10 | KV warm start 27 – 67 × TTFT | paper Tab. 3 | **Unverified** in M0. Plausible, but this is generic prefix-cache behaviour, not specific to ENGRAM | — |
| C11 | HNSW 5.65 × at 51.8 µs; INT8 1.97 × at cos 0.99998 | paper | **Unverified** in M0. Engineering results, not disputed | — |

---

## 2. What went wrong — root causes

1. **The evaluation measured the wrong thing.**
   - Each query was the fingerprint of the *same document text*. The saved v1 artefacts (`results/absolute/layer_keys_3b.pt`, `layer_keys_8b.pt`; 200 docs each) only support one reading: the same text encoded by the other Llama model.
   - Same model + same text is trivially 100 % (identical vectors). The reported 86 % / 98 % only make sense with a second encoding.
   - The evaluation script is not in the repository, so the exact protocol cannot be reconstructed. [`scripts/v2/v1_identity_recheck.py`](../../scripts/v2/v1_identity_recheck.py) recomputes it from the saved artefacts:

     | Fingerprint | 3B → 8B R@1 | 8B → 3B R@1 | 3B → 3B (same model) |
     |---|---|---|---|
     | f1 | 0.995 | 0.970 | 1.000 (trivial) |
     | f0+f1 | 0.995 | 0.990 | 1.000 (trivial) |

   - This doesn't reproduce 86 % / 98 % exactly (likely a different subsampling protocol). In no reading was the query a different text.
2. **No baseline.**
   - v1 never compared against a plain text embedder.
   - For identity matching, a hash of the token IDs scores 100 % at zero cost. For semantic retrieval, a 22 M-parameter MiniLM beats every KV pipeline (§3).
3. **The shared offset hid in plain sight.**
   - `|rFFT|` is non-negative, and every document shares a large layer-mean component. Unrelated documents therefore sit at cos 0.94 – 0.98.
   - In v1's own artefacts, the nearest *unrelated* document has mean cos **0.975** to each document's fingerprint (3B, f0+f1).
   - Margins of ~0.007 were read as signal; they are what remains above that offset.
4. **Per-document basis.**
   - v1's MAR / SVD path projects each document onto its *own* SVD basis, with arbitrary signs. Coordinates are not shared between documents.
   - Short inputs (rows < rank) also produced shorter vectors, which crashed batch evaluation (fixed in v2; the method is retired anyway).
5. **Position.**
   - Keys taken from llama.cpp are post-RoPE, with no inverse rotation.
   - Measured effect on mean-pooled fingerprints is small, except under a long prefix (§3.4).
6. **Thin evidence base.**
   - A 200-document self-built corpus of single sentences (plus 1 000 synthetic).
   - Margin claims resting on 3 trials with 2 documents.
   - No public benchmark.
7. **FFT over the layer axis has no physical meaning.**
   - Layers are not time. Concatenating mid-layer keys beats the layer-FFT in every M0 run (§3.2).
   - Caveat: the two pipelines also differ in layer band (mid 25 – 75 % vs all layers), so this isn't a pure FFT-vs-no-FFT test.

None of these is exotic. Together they are the standard failure mode where the evaluation flatters the method.

---

## 3. M0 — what was measured

### 3.1 Setup

| Item | Value |
|---|---|
| KV models (REFLEX sense) | Qwen2.5-0.5B-Instruct (fp32), Qwen2.5-1.5B-Instruct (fp16), SmolLM2-1.7B-Instruct (fp16) |
| Text embedders (MEANING sense) | all-MiniLM-L6-v2 (22 M), bge-small-en-v1.5 (33 M) |
| `c200-question` | v1 corpus (200 docs, 10 domains). One question per doc, **LLM-generated** (qwen2.5:7b via Ollama, temperature 0) → [`data/c200_queries.jsonl`](../../data/c200_queries.jsonl) |
| `c200-paraphrase` | Same docs, LLM-generated paraphrases |
| `scifact` | BEIR SciFact: 5 183 abstracts, 300 human-written queries, human relevance labels |
| Identity suite | 7 perturbations of each c200 doc: leading / trailing newline, leading space, "Note: " prefix, long prefix sentence, one typo, lowercase |
| Pipelines | 14 per model from a single capture per text ([`engram_bench/catalog.py`](../../engram_bench/catalog.py)) |
| Calibration | Always **out-of-sample**: c200 runs use 256 SciFact docs; SciFact runs use c200 docs + questions |
| Max tokens | 512 per text |
| Hardware | RTX 2060 6 GB (CUDA), torch 2.14.1, transformers 5.19 |
| Harness check | MiniLM SciFact nDCG@10 **0.648** (published 0.645); BGE-small **0.706** (published 0.713) |

Pipeline names used below:

- `v1_fourier`: the v1 fingerprint (post-RoPE keys, f0+f1 over all layers).
- `fourier+PCrm`: de-RoPE, drop the first token, f0+f1, center + remove top PCs.
- `concat_mid+PCrm`: de-RoPE, drop the first token, concatenate mean keys of the middle 50 % of layers, center + remove top PCs.
- `hidden_mid+PCrm`: the same on hidden states instead of keys.

### 3.2 Headline results

**Question → document, R@1 (`c200-question`)**

| Pipeline | Qwen-0.5B | Qwen-1.5B | SmolLM2-1.7B |
|---|---|---|---|
| v1_fourier | 0.190 | 0.290 | 0.235 |
| v1 per-doc SVD (gate 6) | 0.015 | 0.010 | 0.005 |
| fourier+PCrm | 0.355 | 0.430 | 0.680 |
| **concat_mid+PCrm** | **0.450** | **0.740** | **0.815** |
| hidden_mid+PCrm | 0.270 | 0.260 | 0.220 |
| *MiniLM / BGE-small* | *0.970 / 0.975* | | |

**Paraphrase → document, R@1 (`c200-paraphrase`)**

| Pipeline | Qwen-0.5B | Qwen-1.5B | SmolLM2-1.7B |
|---|---|---|---|
| v1_fourier | 0.955 | 0.960 | 0.970 |
| concat_mid+PCrm | 0.985 | 1.000 | 1.000 |
| *MiniLM / BGE-small* | *0.990 / 1.000* | | |

**SciFact, nDCG@10 (public benchmark, human queries)**

| Pipeline | Qwen-0.5B | Qwen-1.5B | SmolLM2-1.7B |
|---|---|---|---|
| v1_fourier | 0.006 | 0.011 | 0.005 |
| v1 per-doc SVD (gate 6) | 0.000 | 0.000 | 0.000 |
| fourier+PCrm | 0.040 | 0.112 | 0.244 |
| concat_mid+PCrm | 0.122 | 0.227 | 0.252 |
| **hidden_mid+PCrm** | **0.182** | **0.286** | **0.324** |
| *MiniLM / BGE-small* | *0.648 / 0.706* | | |

**Shared offset: mean |cos| between unrelated documents**

| Pipeline | c200 (3 models) | SciFact (3 models) |
|---|---|---|
| v1_fourier | 0.941 – 0.972 | 0.960 – 0.983 |
| concat_mid+PCrm | 0.252 – 0.281 | 0.693 – 0.805 |
| *MiniLM / BGE-small* | *0.081 / 0.494* | *0.148 / 0.627* |

### 3.3 Offset removal is the lever, centering alone is not

| R@1 on c200-question | Qwen-0.5B | Qwen-1.5B | SmolLM2-1.7B |
|---|---|---|---|
| fourier, center only | 0.185 | 0.310 | 0.325 |
| fourier, center + PC removal | 0.355 | 0.430 | 0.680 |
| concat_mid, center only | 0.280 | 0.360 | 0.325 |
| concat_mid, center + PC removal | 0.450 | 0.740 | 0.815 |

- The PC cut used here is a crude Marchenko–Pastur edge (`median eigenvalue × (1+√γ)²`). It removed **73 – 113 directions**, far more than v1's `gate_start=6`.
- So v1's gate was the right *principle* (strip the shared top directions), at the wrong scale. A proper noise-edge estimate is open work (H3).

### 3.4 Identity and position

| Min id-R@1 over 7 perturbations | Qwen-0.5B | Qwen-1.5B | SmolLM2-1.7B |
|---|---|---|---|
| v1_fourier (post-RoPE) | 0.81 | 0.98 | 0.90 |
| fourier on de-RoPE'd keys | 0.92 | 0.99 | 0.98 |
| concat_mid+PCrm | **1.00** | **1.00** | **1.00** |
| v1 per-doc SVD | 0.01 | 0.00 | 0.01 |
| *MiniLM / BGE-small* | *1.00 / 1.00* | | |

- In every case the minimum comes from the **long-prefix** perturbation. All other perturbations score 1.00 for the Fourier pipelines.
- Inverse RoPE: `inverse_rope(post)` vs hooked pre-RoPE keys, min cos ≥ **0.999998** on all three models.
- v1's "a leading newline flips the margin sign" is better explained by E-10 (per-document SVD) than by RoPE.

### 3.5 Scale

| Model | Best KV question R@1 | SciFact nDCG@10 (keys / hidden) |
|---|---|---|
| Qwen2.5-0.5B | 0.450 | 0.122 / 0.182 |
| Qwen2.5-1.5B | 0.740 | 0.227 / 0.286 |
| SmolLM2-1.7B | 0.815 | 0.252 / 0.324 |
| *BGE-small (33 M)* | *0.975* | *0.706* |

- Quality rises with model size, but on SciFact the gap to a 33 M embedder stays ≈ 0.4 – 0.6 nDCG at every size.
- Mean-pooled base-LLM states are not trained for retrieval. Closing that gap needs contrastive training, which turns the model into an embedder.

---

## 4. Proven · disproven · open

**Proven (measured in M0, 3 models)**

- The KV fingerprint is a robust identity key: id-R@1 1.00 under all 7 perturbations, with PC removal.
- Removing shared top directions is required and sufficient to make KV geometry discriminative.
- Inverse RoPE is exact; position fragility of mean-pooled fingerprints is small except under long prefixes.
- KV keys match text embedders only on **near-surface paraphrase** (R@1 1.000).
- On short texts, keys beat the same model's hidden states; on long texts (SciFact), hidden states beat keys.

**Disproven**

- KV-cache fingerprints as a semantic retrieval representation competitive with text embedders (H5 null hypothesis).
- v1 Fourier fingerprint as a discriminative representation without offset removal.
- v1 per-document SVD state vectors as a cross-document representation.
- "Corpus-independent" fingerprints.

**Open**

- **H6:** multi-vector KV keys with MaxSim on SciFact. This is the last test of KV semantics; G0-A reopens only if it gets within 5 points of BGE-small nDCG@10.
- **H3:** principled noise-edge PC cut; transfer of the cut across model families.
- **H4 / E-06:** margin vs N at 10³ – 10⁵; finite collapse point.
- **H7:** answer quality after KV restore (not only TTFT).
- Inputs beyond 512 tokens; models above 1.7 B; cross-family identity.
- `c200` queries are LLM-generated. SciFact is the only human-query benchmark here.

---

## 5. Error register (consolidated)

Scope: **v1** = code and claims in this repository. **ENGRAM-R** = the robot-memory design documents (RCL v3, EGR3; the `vector_engram` lineage in `chimera-brain`), which are not in this repository. They are listed so the register stays complete.

| ID | Scope | Issue | Status | Evidence |
|---|---|---|---|---|
| E-01 | ENGRAM-R | RCL v3 field stability ceiling stated in ℓ1. The valid bounds are ℓ∞ or a π-weighted ℓ1. | Confirmed; design doc to be corrected | Star-graph simulation: ‖x‖₁ = 278 while stable, vs claimed ceiling 1.67 |
| E-02 | v1 + ENGRAM-R | All-positive / shared offset dominates fingerprints | **Confirmed** | KV: mean \|cos\| 0.94 – 0.98 (M0). vector_engram synthetic harness: centering → 0.673 → 0.053 |
| E-03 | ENGRAM-R | Drive threshold θ = 0.55 uncalibrated for the RCL fingerprint | Open | — |
| E-04 | ENGRAM-R | Field capacity (2 048 anchors) never measured | Open | — |
| E-05 | v1 | Keys from llama.cpp are post-RoPE; docs and patent text say pre-RoPE | **Confirmed**; fixed in v2 (exact inverse RoPE) | §3.4 |
| E-06 | v1 | "Graceful decay, no collapse point" from a power-law fit over N ≤ 200 | Open (H4) | — |
| E-07 | v1 | "Architecture-invariant" | **Narrowed**: same-text identity across one model family | §1 C4, §2.1 |
| E-08 | v1 | Thin evidence; no embedding baseline | **Closed** by M0 | §3 |
| E-09 | v1 | Headline Recall@1 is same-text identity matching, not semantic retrieval | **Confirmed** | §2.1 |
| E-10 | v1 | Per-document SVD vectors not comparable across documents; variable length for short inputs | **Confirmed**; length fixed, method retired | §3.2, §3.4 |

---

## 6. What changes in v2

- **Two senses, never cross-compared** ([`DECISION_LOG.md`](DECISION_LOG.md) D-010):
  - **REFLEX:** the KV fingerprint, used as identity / dedup / restore key. Open-weight models only.
  - **MEANING:** a text-embedding fingerprint for semantic routing. Works with any model, including closed ones.
- **Certificate first.** The `.eng` v2 / EGR2 certificate ([`specs/ENG_CERTIFICATE_v2_SPEC.md`](specs/ENG_CERTIFICATE_v2_SPEC.md)) becomes the core product: a portable, verifiable record of a model state. Planned fields:
  - `space_key` per sense
  - model binding
  - `PAYLOAD_REF` (content hash of the stored KV state)
  - `TOKEN_SPAN`
  - integrity hash
  - REFLEX and MEANING fingerprint sections
- **Claims policy.** Nothing goes into the README or paper unless an `engram_bench` run file with git SHA and config hash backs it ([`PLAN.md`](PLAN.md) §principles).
- **Next:**
  1. H6 kill test.
  2. M2 certificate implementation.
  3. H7 restore quality.
  4. Paper v2, reframed as "KV identity keys + semantic routing + verifiable restore", with E-09 and E-10 disclosed.

---

## 7. Reproduce

```bash
git checkout v2
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e . --no-deps
uv pip install -r requirements-v2.txt
export PYTHONPATH=.

pytest tests/v2 -q                       # core tests (tiny HF models; skip offline)

# data/c200_queries.jsonl is committed; regenerate with:
#   python scripts/v2/gen_c200_queries.py --model qwen2.5:7b      (needs Ollama)

python scripts/v2/run_m0.py --task baselines --datasets c200,scifact
python scripts/v2/run_m0.py --task kv --model Qwen/Qwen2.5-0.5B-Instruct        --dtype fp32 --datasets c200,scifact
python scripts/v2/run_m0.py --task kv --model Qwen/Qwen2.5-1.5B-Instruct        --dtype fp16 --datasets c200,scifact
python scripts/v2/run_m0.py --task kv --model HuggingFaceTB/SmolLM2-1.7B-Instruct --dtype fp16 --datasets c200,scifact
python -m engram_bench.report --md docs/v2/results/M0_RESULTS_TABLES.md

# E-09 recheck (needs the v1 artefacts in results/absolute/, which are gitignored)
python scripts/v2/v1_identity_recheck.py
```

[`scripts/v2/m0_batch_tuf.sh`](../../scripts/v2/m0_batch_tuf.sh) is the exact batch used for these results. Runs are deterministic up to GPU kernel non-determinism (fp16).

---

## 8. Limitations of these findings

- Models are small (0.5 – 1.7 B). Larger models will score higher; the scaling trend (§3.5) shows no sign of closing the gap to embedders.
- One public benchmark (SciFact). The c200 queries are synthetic.
- All texts are truncated at 512 tokens. Long-context behaviour is untested.
- Mean pooling only. Multi-vector representations (H6) may change the semantic picture; that is the explicit open question.
- The v1 evaluation script is missing. E-09 rests on the saved artefacts and on the paper text.
