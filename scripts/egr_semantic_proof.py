"""
ENGRAM Protocol — EGR Semantic Proof Script
Definitive K→K retrieval validation with diverse, non-repeated documents.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=. \
    .venv/bin/python scripts/egr_semantic_proof.py \
        --model /path/to/model.gguf \
        --ctx 16384 --n-trials 3 --layer-range 8 24 \
        --output results/egr_semantic_proof_8B_14K.json --verbose
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

# ── Documents ─────────────────────────────────────────────────────────────────

DOC_A = """
The transformer architecture introduced in "Attention Is All You Need"
replaced recurrent networks with self-attention as the core computational
primitive. Self-attention computes a weighted sum of value vectors, where
weights derive from the compatibility between query and key vectors.
For a sequence of length n, the attention matrix has shape n×n,
making vanilla attention quadratic in both time and memory.

Multi-head attention partitions the embedding dimension into h parallel
subspaces. Each head independently computes attention using its own
learned projections W_Q, W_K, W_V of dimension d_model/h. The outputs
are concatenated and projected back to d_model via W_O. This allows
different heads to specialize in different relational patterns:
some heads track syntactic dependencies, others semantic similarity,
others coreference chains across longer distances.

Grouped-query attention generalizes multi-head and multi-query attention.
Rather than one KV pair per query head (MHA) or one KV pair for all
heads (MQA), GQA assigns one KV pair per group of g query heads.
Llama 3 uses GQA with 8 KV heads for 32 query heads, reducing
KV cache memory by 4× with minimal quality degradation.

Rotary position embeddings encode absolute position by rotating
query and key vectors in 2D subspaces of the head dimension.
Unlike learned absolute embeddings or sinusoidal encodings,
RoPE naturally extrapolates to sequences longer than those seen
during training by preserving the inner product between positions
i and j as a function only of their relative offset i-j.

The KV cache enables efficient autoregressive generation by storing
computed key and value matrices from all previous positions.
Without caching, generating a sequence of length L requires O(L²)
attention operations. With caching, each new token requires only
O(L) operations — one attention pass over the cached KV pairs.

Flash attention avoids materializing the full n×n attention matrix
by tiling the computation into blocks that fit in SRAM. The forward
pass fuses the softmax and matrix multiply into a single kernel,
achieving O(n) memory complexity while maintaining exact numerical
equivalence to standard attention.

Mixture-of-experts transformer variants route each token to a sparse
subset of feed-forward experts using a learned routing function.
Mistral's Mixtral 8×7B activates 2 of 8 experts per token,
achieving 7B-parameter inference cost with 47B total parameters.
Expert specialization emerges: some experts process syntactic
patterns, others domain-specific content, without explicit supervision.

Layer normalization applied before the attention sublayer (Pre-LN)
stabilizes training compared to Post-LN by ensuring gradients flow
through the residual stream without vanishing through normalized paths.
Modern architectures including Llama, Mistral, and GPT-NeoX all
adopt Pre-LN with RMSNorm, dropping the learned bias parameters.
"""

DOC_B = """
DNA replication in eukaryotic cells initiates at multiple origins
of replication simultaneously, enabling the duplication of genomes
containing billions of base pairs within hours. The origin recognition
complex marks these sites, recruiting CDC6 and CDT1 to load the
MCM helicase onto double-stranded DNA during G1 phase.

The MCM complex unwinds the double helix at replication forks,
separating the complementary strands to serve as templates.
DNA polymerase delta and epsilon synthesize the lagging and leading
strands respectively, both requiring a short RNA primer synthesized
by primase to provide a free 3'-OH group for extension.

Topoisomerase II resolves the positive supercoils that accumulate
ahead of the replication fork as the helix is unwound. Without
topoisomerase activity, the torsional stress would stall replication.
Type II topoisomerases cleave both strands simultaneously, pass
a second duplex through the break, and religate — changing
the linking number by two per catalytic cycle.

Protein synthesis begins with mRNA recognition by the 43S
pre-initiation complex, comprising the 40S ribosomal subunit,
eIF2-GTP-Met-tRNA, and accessory factors. The complex scans
5' to 3' until it encounters the AUG start codon in a favorable
Kozak context. The 60S subunit then joins to form the 80S ribosome.

Elongation proceeds by aminoacyl-tRNA accommodation at the A-site,
peptide bond formation catalyzed by the peptidyl transferase center
of the 23S rRNA, and translocation driven by EF-G and GTP hydrolysis.
Each elongation cycle advances the ribosome by exactly one codon,
consuming one GTP equivalent and incorporating one amino acid.

Cell signaling cascades amplify extracellular signals through
phosphorylation networks. The MAPK/ERK pathway converts growth
factor receptor activation into nuclear transcription factor
phosphorylation through RAF, MEK, and ERK kinases. Signal amplitude
and duration encode distinct transcriptional outcomes — transient
ERK activation drives proliferation while sustained activation
drives differentiation in PC12 cells.

CRISPR-Cas9 genome editing exploits the bacterial adaptive immunity
system in which Cas9 endonuclease is guided by a 20-nucleotide
spacer sequence in the sgRNA to cleave complementary genomic DNA.
The PAM sequence NGG immediately 3' of the target site is required
for Cas9 binding and R-loop formation. Double-strand breaks are
repaired by NHEJ (causing indels) or HDR (enabling precise edits).
"""

QUERY = "How does the attention mechanism use keys and queries to compute weighted context representations in transformer models?"


def run_trial(
    llm,
    n_kv_heads: int,
    head_dim: int,
    spec: dict,
    extractor,
    doc_a: str,
    doc_b: str,
    query: str,
    trial_id: int,
    verbose: bool,
) -> dict:
    """Run a single EGR semantic proof trial."""
    from kvcos.core.blob_parser import parse_state_blob
    from kvcos.core.manifold_index import IndexEntry, ManifoldIndex

    dim = extractor.output_dim(spec)
    index = ManifoldIndex(dim=dim)

    # ── Session A ─────────────────────────────────────────
    llm.reset()
    t0 = time.perf_counter()
    llm(doc_a, max_tokens=1, temperature=0.0)
    cold_ms = (time.perf_counter() - t0) * 1000
    n_tok_a = llm.n_tokens

    state_a = llm.save_state()
    blob_a = bytes(state_a.llama_state)
    blob_mb = len(blob_a) / 1024 / 1024

    # Warm TTFT
    llm.reset()
    gc.collect()
    t0 = time.perf_counter()
    llm.load_state(state_a)
    llm(" ", max_tokens=1, temperature=0.0)
    warm_ms = (time.perf_counter() - t0) * 1000
    speedup = cold_ms / warm_ms if warm_ms > 0 else float("inf")

    # Parse + extract A
    t0 = time.perf_counter()
    parsed_a = parse_state_blob(blob_a, n_kv_heads=n_kv_heads, head_dim=head_dim)
    parse_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    ext_a = extractor.extract(parsed_a.keys, spec)
    extract_ms = (time.perf_counter() - t0) * 1000

    entry_a = IndexEntry(
        cache_id="session-a",
        task_description="Transformer attention mechanisms",
        model_id=spec["model_id"],
        created_at=datetime.now(timezone.utc).isoformat(),
        context_len=parsed_a.n_cells,
        l2_norm=ext_a.l2_norm,
    )
    index.add(ext_a.state_vec, entry_a)

    # ── Session B ─────────────────────────────────────────
    llm.reset()
    llm(doc_b, max_tokens=1, temperature=0.0)
    n_tok_b = llm.n_tokens
    state_b = llm.save_state()
    blob_b = bytes(state_b.llama_state)
    parsed_b = parse_state_blob(blob_b, n_kv_heads=n_kv_heads, head_dim=head_dim)
    ext_b = extractor.extract(parsed_b.keys, spec)

    entry_b = IndexEntry(
        cache_id="session-b",
        task_description="DNA replication and molecular biology",
        model_id=spec["model_id"],
        created_at=datetime.now(timezone.utc).isoformat(),
        context_len=parsed_b.n_cells,
        l2_norm=ext_b.l2_norm,
    )
    index.add(ext_b.state_vec, entry_b)

    # ── Query ─────────────────────────────────────────────
    llm.reset()
    llm(query, max_tokens=1, temperature=0.0)
    n_tok_q = llm.n_tokens
    state_q = llm.save_state()
    blob_q = bytes(state_q.llama_state)
    parsed_q = parse_state_blob(blob_q, n_kv_heads=n_kv_heads, head_dim=head_dim)

    t0 = time.perf_counter()
    ext_q = extractor.extract(parsed_q.keys, spec)
    t1 = time.perf_counter()
    results = index.search(ext_q.state_vec, top_k=2)
    t2 = time.perf_counter()

    search_ms = (t2 - t1) * 1000
    egr_total_ms = (t2 - t0) * 1000 + extract_ms  # query extract + search + index extract

    # Score extraction
    score_a = next((r["similarity"] for r in results if "attention" in r["task_description"].lower() or "transformer" in r["task_description"].lower()), None)
    score_b = next((r["similarity"] for r in results if "dna" in r["task_description"].lower() or "molecular" in r["task_description"].lower()), None)

    if score_a is None or score_b is None:
        # Fallback: use position
        score_a = results[0]["similarity"] if results else 0
        score_b = results[1]["similarity"] if len(results) > 1 else 0

    margin = score_a - score_b
    correct = len(results) > 0 and (
        "attention" in results[0]["task_description"].lower()
        or "transformer" in results[0]["task_description"].lower()
    )

    layer_range_used = list(extractor.layer_range) if extractor.layer_range else "spec_default"

    trial = {
        "trial_id": trial_id,
        "n_cells_a": parsed_a.n_cells,
        "n_cells_b": parsed_b.n_cells,
        "n_cells_q": parsed_q.n_cells,
        "score_a": round(score_a, 6),
        "score_b": round(score_b, 6),
        "margin": round(margin, 6),
        "correct": correct,
        "cold_ms": round(cold_ms, 1),
        "warm_ms": round(warm_ms, 1),
        "speedup": round(speedup, 1),
        "parse_ms": round(parse_ms, 1),
        "extract_ms": round(extract_ms, 1),
        "search_ms": round(search_ms, 1),
        "egr_total_ms": round(egr_total_ms, 1),
        "blob_size_mb": round(blob_mb, 1),
        "layer_range_used": layer_range_used,
        "n_layers_used": extractor.layer_range[1] - extractor.layer_range[0] if extractor.layer_range else len(spec.get("extraction_layers", ())),
        "svd_rank": extractor.rank,
        "output_dim": dim,
    }

    if verbose:
        print(f"  Trial {trial_id}: margin={margin:.4f} correct={correct} "
              f"cold={cold_ms:.0f}ms warm={warm_ms:.0f}ms "
              f"egr={egr_total_ms:.1f}ms cells_a={parsed_a.n_cells}")

    return trial


def main() -> int:
    parser = argparse.ArgumentParser(description="ENGRAM EGR Semantic Proof")
    parser.add_argument("--model", "-m", required=True, help="Path to GGUF model")
    parser.add_argument("--ctx", type=int, default=16384, help="Context window")
    parser.add_argument("--n-trials", type=int, default=3, help="Number of trials")
    parser.add_argument("--layer-range", type=int, nargs=2, default=[8, 24], help="Layer range start end")
    parser.add_argument("--gate-start", type=int, default=0, help="Skip top N singular values (0=none)")
    parser.add_argument("--compression", default="FP16", help="Compression method: FP16, INT8, Q8_0")
    parser.add_argument("--output", "-o", default="results/egr_semantic_proof.json", help="Output JSON path")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    from llama_cpp import Llama
    import llama_cpp as lc

    from kvcos.core.cache_spec import make_spec_from_metadata
    from kvcos.core.types import StateExtractionMode
    from kvcos.core.state_extractor import MARStateExtractor

    layer_range = tuple(args.layer_range)

    print(f"ENGRAM EGR Semantic Proof — {args.n_trials} trials")
    print(f"Model: {args.model}")
    print(f"Context: {args.ctx}, Layer range: {layer_range}")
    print()

    trials: list[dict] = []
    for trial_id in range(args.n_trials):
        print(f"Trial {trial_id + 1}/{args.n_trials}...")

        llm = Llama(model_path=args.model, n_ctx=args.ctx, n_gpu_layers=-1, verbose=False)
        meta = llm.metadata
        n_layers = int(meta.get("llama.block_count", "32"))
        n_heads = int(meta.get("llama.attention.head_count", "32"))
        n_kv_heads = int(meta.get("llama.attention.head_count_kv", "8"))
        head_dim = int(meta.get("llama.embedding_length", "4096")) // n_heads
        model_name = meta.get("general.name", Path(args.model).stem)

        spec = make_spec_from_metadata(
            model_id=model_name, n_layers=n_layers, n_heads=n_heads,
            n_kv_heads=n_kv_heads, head_dim=head_dim,
        )

        extractor = MARStateExtractor(
            mode=StateExtractionMode.SVD_PROJECT,
            rank=min(160, head_dim),
            layer_range=layer_range,
            gate_start=args.gate_start,
        )

        trial = run_trial(
            llm=llm, n_kv_heads=n_kv_heads, head_dim=head_dim,
            spec=spec, extractor=extractor,
            doc_a=DOC_A.strip(), doc_b=DOC_B.strip(), query=QUERY.strip(),
            trial_id=trial_id, verbose=args.verbose,
        )
        trials.append(trial)

        del llm
        gc.collect()

    # ── Summary statistics ────────────────────────────────
    margins = [t["margin"] for t in trials]
    speedups = [t["speedup"] for t in trials]
    egr_times = [t["egr_total_ms"] for t in trials]
    n_correct = sum(1 for t in trials if t["correct"])

    mean_margin = sum(margins) / len(margins)
    std_margin = math.sqrt(sum((m - mean_margin) ** 2 for m in margins) / max(len(margins) - 1, 1)) if len(margins) > 1 else 0.0
    mean_speedup = sum(speedups) / len(speedups)
    std_speedup = math.sqrt(sum((s - mean_speedup) ** 2 for s in speedups) / max(len(speedups) - 1, 1)) if len(speedups) > 1 else 0.0
    mean_egr = sum(egr_times) / len(egr_times)
    std_egr = math.sqrt(sum((e - mean_egr) ** 2 for e in egr_times) / max(len(egr_times) - 1, 1)) if len(egr_times) > 1 else 0.0

    passed = (
        mean_margin > 0.05
        and n_correct == args.n_trials
        and mean_egr < 200
        and mean_speedup > 10
    )

    summary = {
        "mean_margin": round(mean_margin, 4),
        "std_margin": round(std_margin, 4),
        "mean_speedup": round(mean_speedup, 1),
        "std_speedup": round(std_speedup, 1),
        "mean_egr_ms": round(mean_egr, 1),
        "std_egr_ms": round(std_egr, 1),
        "n_correct": n_correct,
        "n_trials": args.n_trials,
        "min_margin": round(min(margins), 4),
        "max_margin": round(max(margins), 4),
        "pass": passed,
    }

    # ── Build output JSON ─────────────────────────────────
    doc_a_tokens = trials[0]["n_cells_a"] if trials else 0
    doc_b_tokens = trials[0]["n_cells_b"] if trials else 0
    query_tokens = trials[0]["n_cells_q"] if trials else 0

    output = {
        "metadata": {
            "model": model_name,
            "ctx": args.ctx,
            "layer_range": list(layer_range),
            "n_trials": args.n_trials,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": "Apple M3 / macOS",
            "llama_cpp_version": lc.__version__,
        },
        "documents": {
            "doc_a": {"description": "Transformer attention mechanisms (ML)", "n_tokens": doc_a_tokens},
            "doc_b": {"description": "DNA replication and molecular biology", "n_tokens": doc_b_tokens},
            "query": {"text": QUERY, "n_tokens": query_tokens},
        },
        "trials": trials,
        "summary": summary,
    }

    # ── Write JSON ────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {output_path}")

    # ── Print summary ─────────────────────────────────────
    print()
    sep = "=" * 55
    print(sep)
    print("ENGRAM EGR Semantic Proof — Summary")
    print(sep)
    print(f"Model:       {model_name}")
    print(f"Context:     {args.ctx}")
    print(f"Layer range: {layer_range}")
    print(f"Trials:      {args.n_trials}")
    print()
    print(f"K→K margin:  {mean_margin:.4f} ± {std_margin:.4f} (min={min(margins):.4f}, max={max(margins):.4f})")
    print(f"Correct:     {n_correct}/{args.n_trials}")
    print(f"Speedup:     {mean_speedup:.1f}x ± {std_speedup:.1f}x")
    print(f"EGR ms:      {mean_egr:.1f}ms ± {std_egr:.1f}ms")
    print()
    verdict = "PASS" if passed else "FAIL"
    reasons = []
    if mean_margin <= 0.05:
        reasons.append(f"margin {mean_margin:.4f} <= 0.05")
    if n_correct < args.n_trials:
        reasons.append(f"correct {n_correct}/{args.n_trials}")
    if mean_egr >= 200:
        reasons.append(f"egr {mean_egr:.1f}ms >= 200ms")
    if mean_speedup <= 10:
        reasons.append(f"speedup {mean_speedup:.1f}x <= 10x")
    reason_str = " | ".join(reasons) if reasons else "all criteria met"
    print(f"Verdict:     {verdict} ({reason_str})")
    print(sep)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
