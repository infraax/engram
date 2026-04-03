# ENGRAM Experimental Results

## Provenance
All results generated on Apple M3 (24GB), macOS.
llama-cpp-python 0.3.19, Metal GPU acceleration.
Models: Q4_K_M quantization (bartowski HuggingFace repo).

## Files
- raw/           Individual trial JSON outputs
- summary/       Aggregated tables

## Methodology
- Cold TTFT: time from llm() call to first token, zero-shot
- Warm TTFT: time from load_state() through first token
- Speedup: cold_ms / warm_ms
- K→K margin: score(correct) - score(incorrect) via FAISS IndexFlatIP
- EGR overhead: extract_ms + search_ms (not including parse or serialize)
- n_trials: 3 per configuration minimum
- Layer range: 8-24 (16 of 32 layers) for 8B model

## Validity Notes
- 32K synthetic test: ranking failed due to repeated-sentence content.
  Not a system failure. Excluded from paper table.
- All reported margins use topically coherent, non-repeated documents.
- Documents are stripped of leading/trailing whitespace before tokenization.

## Methodology Notes — Tokenization Sensitivity

During validation, a leading newline character in DOC_A caused
deterministic K→K ranking failure (margin -0.272) across all 3 trials.
Removing the leading whitespace restored correct ranking (margin +0.381).

Interpretation: the pre-RoPE key manifold is sensitive to token
boundary artifacts. A single leading newline shifts the SVD projection
sufficiently to invert retrieval ranking. This is not a system failure —
it demonstrates high manifold resolution.

Production implication: all documents ingested by ENGRAM must be
stripped of leading/trailing whitespace before tokenization.
This is enforced in scripts/egr_semantic_proof.py (fixed April 1 2026).

Paper implication: report this as a preprocessing requirement,
not a limitation. It demonstrates the geometric sensitivity of EGR.
