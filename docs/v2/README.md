# ENGRAM v2 — documentation index

**Start here:** [FINDINGS.md](FINDINGS.md) explains what v1 got wrong, what M0 measured, and what is proven, disproven or open.

| Document | What it is |
|---|---|
| [FINDINGS.md](FINDINGS.md) | v1 post-mortem + M0 findings, claim-by-claim status, consolidated error register E-01…E-10 |
| [results/M0_RESULTS.md](results/M0_RESULTS.md) | M0 narrative results per hypothesis (H1, H2, H5) + Qwen2.5-1.5B addendum |
| [results/M0_RESULTS_TABLES.md](results/M0_RESULTS_TABLES.md) | Full generated tables (all 14 pipelines × 3 models + 2 embedders × 3 datasets) |
| [DECISION_LOG.md](DECISION_LOG.md) | Dated decisions D-000…D-010 (G0 = B is D-010) |
| [REVIEW_v1.md](REVIEW_v1.md) | Code-level review of v1 before M0 |
| [PLAN.md](PLAN.md) | v2 development plan: phases, pipeline stages S0–S8, gates |
| [MATH_CROSSMAP.md](MATH_CROSSMAP.md) | Cross-map of math results onto the design, hypotheses H1–H10, early error register |
| [specs/ENG_CERTIFICATE_v2_SPEC.md](specs/ENG_CERTIFICATE_v2_SPEC.md) | `.eng` v2 / EGR2 certificate spec (draft) |
| [specs/GFR_v2_ALGORITHM_SPEC.md](specs/GFR_v2_ALGORITHM_SPEC.md) | GFR-A retrieval and write engine spec (draft) |

Code: [`engram_core/`](../../engram_core) (space descriptors, capture, inverse RoPE, fingerprint pipelines) · [`engram_bench/`](../../engram_bench) (datasets, metrics, baselines, run registry, report) · [`scripts/v2/`](../../scripts/v2) · [`tests/v2/`](../../tests/v2) · run files in [`results/v2/`](../../results/v2).

Specs and the plan were written **before** M0. Where they assume KV fingerprints carry semantics, FINDINGS.md and D-010 take precedence.
