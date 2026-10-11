# ENGRAM v2 — M0 Results (P0 first pass)
### 2026-10-11 · TUF (RTX 2060, CUDA) · branch `v2` · harness `engram_bench` · tables: `M0_RESULTS_TABLES.md`
### Status: G0 = **B (final for M0)**. Confirmed on 3 models (Qwen2.5-0.5B, Qwen2.5-1.5B, SmolLM2-1.7B) — see §7 addendum.

---

## 1. Setup

| Item | Value |
|---|---|
| Models (REFLEX / KV) | Qwen2.5-0.5B-Instruct (fp32), Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct (fp16) |
| Baselines (MEANING) | all-MiniLM-L6-v2 (22M), bge-small-en-v1.5 (33M). nomic-embed dropped: remote code breaks under transformers 5 |
| Datasets | `c200-question` (v1 corpus, 200 one-sentence docs, LLM-generated questions — **synthetic**, qwen2.5:7b via Ollama) · `c200-paraphrase` (same, paraphrases) · `scifact` (BEIR SciFact, 5183 abstracts, 300 human queries) |
| Identity suite | 7 perturbations of each c200 doc: leading/trailing newline, leading space, "Note: " prefix, long prefix sentence, typo, lowercase |
| Pipelines | 14 per model, one capture per text (catalog in `engram_bench/catalog.py`) |
| Calibration | Out-of-sample: c200 runs → 256 SciFact docs; SciFact runs → c200 docs + questions |
| Max tokens | 512 per text |
| Harness check | MiniLM SciFact nDCG@10 **0.648** (published 0.645) · BGE-small **0.706** (published 0.713) → harness reproduces public numbers |

---

## 2. Headline table

| Task (metric) | BGE-small | MiniLM | Best KV — SmolLM2-1.7B | Best KV — Qwen-0.5B | v1 Fourier (best model) | v1 per-doc SVD |
|---|---|---|---|---|---|---|
| Question → doc (R@1) | **0.975** | 0.970 | 0.815 (concat_mid, PC-removed) | 0.450 | 0.235 | ≤ 0.015 |
| Paraphrase → doc (R@1) | 1.000 | 0.990 | **1.000** | 0.985 | 0.970 | ≤ 0.025 |
| SciFact (nDCG@10) | **0.706** | 0.648 | 0.252 (K) · 0.324 (hidden) | 0.122 (K) · 0.182 (hidden) | 0.006 | 0.000 |
| Identity under perturbation (min id-R@1) | 1.00 | 1.00 | 1.00 | 0.99 | 0.81–0.90 | 0.00–0.02 |
| Mean \|cos\| between unrelated docs | 0.49 | 0.08 | 0.25 (after PC removal) | 0.27 | **0.95–0.98** | — |

---

## 3. Findings per hypothesis

**H5 — null hypothesis (K-fingerprint ≥ text embedder): FALSE.**
- On semantic questions, the best KV pipeline reaches R@1 0.815 vs 0.975 for a 33M embedder.
- On the only public benchmark (SciFact), the KV fingerprint reaches 36 % of BGE's nDCG@10 (0.252 vs 0.706).
- The KV fingerprint only matches the embedders on **near-surface paraphrase** (R@1 1.000).

**H2 — outlier removal is the lever: SUPPORTED (reframed).**
- Centering alone does nothing for retrieval: R@1 0.305 → 0.325 on SmolLM; it even drops on Qwen.
- Removing the dominant principal directions (centering + top-PC removal, crude Marchenko–Pastur cut) is what works:
  - Fourier R@1 on SmolLM: 0.285 → 0.680.
  - Mid-layer concat R@1 on SmolLM: 0.325 → 0.815.
- The direct comparison with v1's `gate_start=6` is impossible, because the v1 per-doc SVD is broken (E-10). The principle behind it (strip the shared top directions) is confirmed.

**H1 — RoPE / position: TRUE BUT SMALL for mean-pooled fingerprints.**
- `inverse_rope(post)` equals the hooked pre-RoPE keys to cos **1.000000** (min 0.999999) on every model → stage S1 is exact.
- Fourier fingerprints were already position-stable under most perturbations (self-cos ≥ 0.98).
- De-RoPE only helps the long-prefix case: identity R@1 0.81 → 0.92 (Qwen), 0.90 → 0.98 (SmolLM).
- The v1 "leading newline flips the sign" result is better explained by E-10 than by RoPE.

**Scale:**
- 0.5B → 1.7B roughly doubles semantic quality: question R@1 0.45 → 0.815; SciFact nDCG 0.122 → 0.252.
- Still far below a dedicated embedder at 1/50th of the parameters.

**Long documents:**
- Mean pooling over up to 512 tokens collapses discrimination on SciFact abstracts.
- Even after PC removal, mean \|cos\| between unrelated docs stays 0.6–0.8.
- This is the case H6 (multi-vector / MaxSim) is meant to test.

**K vs hidden states:**
- On short texts (c200), K-geometry beats the model's own mid-layer hidden states.
- On long texts (SciFact), hidden states beat K: 0.324 vs 0.252.
- K is not uniquely semantic.

---

## 4. New errors (register continues from MATH_CROSSMAP E-02…E-08)

| ID | Issue | Evidence |
|---|---|---|
| **E-09** | v1's headline "98 % Recall@1" never tested semantic retrieval. The query was the *same text*, fingerprinted by a second model of the same family (cross-model identity matching). | `paper/engram.tex` §Same-Model Retrieval; `results/absolute/ENGRAM_ABSOLUTE_PROOF.json` ("cross-model recall"); `MULTIFREQ_PROOF.json` |
| **E-10** | v1 per-doc SVD state vectors are not comparable across documents. Each document is projected onto its *own* SVD basis, with arbitrary signs. It also produced **shorter vectors for short inputs** (rows < rank) — fixed in v2 (fixed-length output). | R@1 0.000–0.025 on every task; identity R@1 ≤ 0.02 even for a trailing newline; Qwen-1.5B crash on short queries |
| **E-02** (confirmed for KV) | Positive/shared offset dominates KV fingerprints. Mean \|cos\| between unrelated v1 Fourier fingerprints is 0.95–0.98, much worse than the 0.64 predicted for half-normal vectors. | `offset.pair_abs_cos_mean` in all kv runs |

---

## 5. G0 decision (final for M0 — confirmed by §7)

**G0-B:**
- The **KV fingerprint ships as the identity / dedup / restore key.** Strengths: identity R@1 = 1.00 under all perturbations, zero extra model (computed during prefill), tied exactly to the restorable state.
- **Semantic retrieval ships on the MEANING sense** (dedicated text embedder).
- The certificate (EGR2) and GFR-A designs are unaffected: both were already written for two senses.

Not G0-A (KV is clearly below embedders on semantics). Not G0-C (the KV key has a real, measured job).

**What would reopen G0-A:** H6 (multi-vector K with MaxSim) reaching within 5 pt nDCG@10 of BGE-small on SciFact, at an equal or smaller index.

---

## 6. Next steps

1. ~~Append Qwen2.5-1.5B; finalise G0~~ — done (§7, D-010 final).
2. **H6:** multi-vector KV (top-k / blockwise token keys, MaxSim) on SciFact — the last test for KV semantics.
3. **H3:** replace the crude "bbp" cut with a proper noise-edge estimate; sweep the number of removed PCs; check transfer across the 3 families.
4. **Restore path (H7):** retrieve (MEANING) → locate the cached state by its KV identity key → inject → answer quality. This is where the KV key earns its place.
5. M2 certificate work can start now. The fields G0-B needs are already in the plan:
   - `space_key` per sense
   - `PAYLOAD_REF`
   - `TOKEN_SPAN`
   - the KV identity fingerprint as CORE `fp`, plus a MEANING fingerprint section
6. Paper v2 framing: from "Fourier fingerprints are semantic memory" → "KV identity keys + semantic routing + restore". E-09 and E-10 are disclosed.

---

## 7. Addendum — Qwen2.5-1.5B-Instruct (rerun with fixed-length SVD, E-10 fix)

| Task (metric) | v1 Fourier | Fourier + PC removal | Best KV (concat_mid_derope_sink1_abtt) | Hidden mid (abtt) |
|---|---|---|---|---|
| Question → doc (R@1) | 0.290 | 0.430 | **0.740** | 0.260 |
| Paraphrase → doc (R@1) | 0.960 | 0.980 | **1.000** | 0.960 |
| SciFact (nDCG@10) | 0.011 | 0.112 | 0.227 | **0.286** |
| Identity, min id-R@1 (7 perturbations) | ≥0.99 | 1.00 | **1.00** | 1.00 |
| Mean \|cos\| unrelated docs | 0.941 | 0.264 | 0.281 | 0.742 |

- RoPE check: derope(post) vs pre cos 0.999999 (min 0.999998) → S1 exact on a third model.
- v1 per-doc SVD still ≤ 0.020 R@1 → E-10 is a design flaw, not the short-input crash; the crash fix only made it runnable.
- Same ranking as the other two models: PC removal is the lever; mid-layer concat beats Fourier; hidden beats K only on long texts.

**Scale curve (best KV):**

| Model | Question R@1 | SciFact nDCG@10 (K / hidden) |
|---|---|---|
| Qwen2.5-0.5B | 0.450 | 0.122 / 0.182 |
| Qwen2.5-1.5B | 0.740 | 0.227 / 0.286 |
| SmolLM2-1.7B | 0.815 | 0.252 / 0.324 |
| BGE-small (33M embedder) | 0.975 | 0.706 |

- Monotone with size, but the gap to a 33M embedder on SciFact stays ≈ 0.45 nDCG at every size → mean-pooled KV does not close the semantic gap by scaling within reach.
- **G0-B is final for M0.** Reopen only via H6 (multi-vector K + MaxSim within 5 pt of BGE on SciFact).
