"""
Compute a Fixed Corpus Basis (FCB) for cross-document and
cross-model stable state vector extraction.

The FCB is the principal subspace of the key manifold computed
from a diverse reference corpus. Unlike per-document SVD,
the FCB is document-independent — all documents projected
with the same FCB exist in the same coordinate system.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from llama_cpp import Llama

from kvcos.core.blob_parser import parse_state_blob
from kvcos.core.state_extractor import MARStateExtractor
from scripts.generate_alignment_dataset import DOCUMENTS


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Fixed Corpus Basis")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer-range", type=int, nargs=2, default=[8, 24])
    parser.add_argument("--gate-start", type=int, default=6)
    parser.add_argument("--rank", type=int, default=122)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    llm = Llama(model_path=args.model, n_ctx=2048, n_gpu_layers=-1, verbose=False)
    meta = llm.metadata
    n_kv = int(meta.get("llama.attention.head_count_kv", "8"))
    head_dim = int(meta.get("llama.embedding_length", "4096")) // int(
        meta.get("llama.attention.head_count", "32")
    )
    model_name = meta.get("general.name", "unknown")

    print(f"Model: {model_name} ({n_kv} KV heads, {head_dim} head_dim)")
    print(f"Layer range: {args.layer_range}, gate_start: {args.gate_start}")
    print(f"Collecting key tensors from {len(DOCUMENTS)} documents...")

    key_tensors: list[torch.Tensor] = []
    for i, doc in enumerate(DOCUMENTS):
        llm.reset()
        llm(doc.strip(), max_tokens=1, temperature=0.0)
        s = llm.save_state()
        parsed = parse_state_blob(
            bytes(s.llama_state), n_kv_heads=n_kv, head_dim=head_dim
        )
        key_tensors.append(parsed.keys)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(DOCUMENTS)}")
    del llm
    gc.collect()

    print("Computing corpus SVD...")
    basis = MARStateExtractor.compute_corpus_basis(
        key_tensors=key_tensors,
        layer_range=tuple(args.layer_range),
        gate_start=args.gate_start,
        rank=args.rank,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "basis": basis,
            "model_name": model_name,
            "layer_range": args.layer_range,
            "gate_start": args.gate_start,
            "rank": args.rank,
            "n_corpus_docs": len(DOCUMENTS),
            "key_tensors": key_tensors,
        },
        str(output_path),
    )

    print(f"Basis shape: {basis.shape}")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
