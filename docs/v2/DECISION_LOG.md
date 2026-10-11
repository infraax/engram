# ENGRAM v2 — Decision Log
Format: `YYYY-MM-DD | ID | decision | evidence (run_id / doc) | status`

2026-10-11 | D-000 | Branch `v2` created as worktree `~/IDEAS/ENGRAM/engram-v2` from origin/main @ d45273e; `kvcos/` frozen as v1 reference | — | done
2026-10-11 | D-001 | Machines: M3 = reference/interactive; TUF (`tuf-ts`) = batch worker | PLAN.md §1.3 | done
2026-10-11 | D-002 | Env M3: py3.12 venv, torch 2.14.1 (MPS), transformers 5.19, faiss 1.15.1, numpy 2.5, llama-cpp-python 0.3.36 (Metal), mlx; v1 tests 201 passed / 19 skipped under numpy 2 | setup log | done
2026-10-11 | D-003 | Env TUF: uv-managed py3.12, torch 2.14.1+cu126 (RTX 2060 OK), transformers 5.19, faiss 1.15.1; llama-cpp-python CPU build in progress | setup log | done
2026-10-11 | D-004 | v1 dependency pins (numpy<2, safetensors<0.5) are overridden in the v2 env; v2 gets its own pyproject at M0 | — | open
2026-10-11 | D-005 | M0 harness validated: MiniLM/BGE SciFact nDCG@10 0.648/0.706 vs published 0.645/0.713 | results/v2/2026-10-11/*base-* | done
2026-10-11 | D-006 | S1 inverse RoPE exact (cos 1.000000 vs hooked pre-RoPE keys, all models) | kv runs extra.rope_check | done
2026-10-11 | D-007 | H5 null hypothesis holds: KV fingerprint < text embedder on semantic retrieval (c200-q R@1 0.815 vs 0.975; SciFact nDCG 0.252 vs 0.706) | M0_RESULTS.md §2 | done
2026-10-11 | D-008 | H2 reframed+supported: top-PC removal (not centering alone) is the lever (SmolLM Fourier R@1 0.285→0.680) | M0_RESULTS.md §3 | done
2026-10-11 | D-009 | E-09 (v1 R@1 = cross-model identity, not semantic) and E-10 (v1 per-doc SVD not comparable across docs; variable length) logged; svd fixed-length fix in v2 | M0_RESULTS.md §4 | done
2026-10-11 | D-010 | G0 provisional = B: KV fingerprint = identity/dedup/restore key; semantics via MEANING embedder; reopen G0-A only if H6 multi-vector reaches BGE-5pt on SciFact | M0_RESULTS.md §5, §7 | **final** — confirmed on Qwen2.5-1.5B (question R@1 0.740, SciFact 0.227 vs BGE 0.706); 3 models, 2 families
