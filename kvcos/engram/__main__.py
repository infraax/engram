"""
EIGENGRAM command-line interface.

Usage:
    python -m kvcos.engram encode  --model <gguf> --text "..." --out doc.eng
    python -m kvcos.engram search  --model <gguf> --query "..." index/*.eng
    python -m kvcos.engram inspect doc.eng
    python -m kvcos.engram list    index/*.eng

Commands:
    encode    Run a document through a GGUF model and write a .eng file.
    search    Query .eng files using a text query and a model.
    inspect   Print all fields from .eng files (no model needed).
    list      Print a summary table of .eng files (no model needed).
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import sys

import torch


def _resolve_paths(patterns: list[str]) -> list[str]:
    """Expand glob patterns, return sorted list of .eng paths."""
    paths = []
    for p in patterns:
        expanded = glob.glob(p)
        if expanded:
            paths.extend(expanded)
        elif os.path.exists(p):
            paths.append(p)
        else:
            print(f"Warning: no files matched '{p}'", file=sys.stderr)
    return sorted(set(paths))


def cmd_encode(args: argparse.Namespace) -> None:
    """Encode a document as a .eng EIGENGRAM file."""
    from kvcos.engram.writer import write_eigengram

    if args.text:
        text = args.text
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: input file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        text = open(args.file).read().strip()
    else:
        print("Error: provide --text or --file", file=sys.stderr)
        sys.exit(1)

    output_path = args.out or (
        os.path.splitext(args.file)[0] + ".eng" if args.file else "output.eng"
    )
    task_desc = args.description or text[:80]
    cache_id_val = args.id or text[:64]

    print(f"Encoding document...")
    print(f"  Model:  {args.model}")
    print(f"  Text:   {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"  Output: {output_path}")
    print()

    result = write_eigengram(
        model_path=args.model,
        text=text,
        output_path=output_path,
        cache_id=cache_id_val,
        task_description=task_desc,
        basis_path=args.basis,
    )

    print(f"Done.")
    print(f"  File size : {result['file_size_bytes']} bytes")
    print(f"  Model ID  : {result['model_id']}")
    print(f"  SCS       : {result['scs']:.4f}")
    print(f"  Basis rank: {result['basis_rank']}")


def cmd_search(args: argparse.Namespace) -> None:
    """Search .eng files using a text query."""
    from llama_cpp import Llama

    from kvcos.core.blob_parser import parse_state_blob
    from kvcos.engram.reader import load_eigengram_index
    from kvcos.core.manifold_index import ManifoldIndex

    paths = _resolve_paths(args.eng_files)
    if not paths:
        print("No .eng files found.", file=sys.stderr)
        sys.exit(1)

    fingerprint = args.fingerprint
    saved = torch.load(args.basis, weights_only=False)
    P = saved["basis"]
    center = saved["joint_center"]
    LR = (8, 24)
    GATE = 6
    RANK = P.shape[0]

    print(f"Query:        {args.query}")
    print(f"Index:        {len(paths)} files")
    print(f"Fingerprint:  {fingerprint}")
    print()

    llm = Llama(model_path=args.model, n_ctx=2048, n_gpu_layers=-1, verbose=False)
    meta = llm.metadata
    n_kv = int(meta.get("llama.attention.head_count_kv", "8"))
    hd = int(meta.get("llama.embedding_length", "4096")) // int(
        meta.get("llama.attention.head_count", "32")
    )
    llm.reset()
    llm(args.query.strip(), max_tokens=1, temperature=0.0)
    p_q = parse_state_blob(
        bytes(llm.save_state().llama_state), n_kv_heads=n_kv, head_dim=hd
    )
    del llm
    gc.collect()

    if fingerprint == "fourier":
        from kvcos.core.fingerprint import compute_fourier_fingerprint

        # Use ALL layers for Fourier (not sliced)
        layer_means = p_q.keys.float().mean(dim=2).reshape(p_q.keys.shape[0], -1)
        query_vec = compute_fourier_fingerprint(layer_means, freqs=[0, 1])
        dim = query_vec.shape[0]
    elif fingerprint == "perdoc":
        k_q = p_q.keys[LR[0] : LR[1]].float().reshape(-1, hd)
        _, _, Vh = torch.linalg.svd(k_q, full_matrices=False)
        proj_q = (k_q @ Vh[GATE : GATE + RANK].T).mean(0)
        query_vec = proj_q / (proj_q.norm() + 1e-8)
        dim = RANK
    else:  # fcdb
        k_q = p_q.keys[LR[0] : LR[1]].float().reshape(-1, hd)
        mean_q = k_q.mean(0)
        delta_q = mean_q - center
        delta_q = delta_q / (delta_q.norm() + 1e-8)
        query_vec = delta_q @ P.T
        query_vec = query_vec / (query_vec.norm() + 1e-8)
        dim = RANK

    vecs, entries = load_eigengram_index(paths, fingerprint=fingerprint)
    idx = ManifoldIndex(dim=dim)
    for v, e in zip(vecs, entries):
        idx.add(v, e)

    top_k = min(args.top_k, len(paths))
    results = idx.search(query_vec, top_k=top_k)

    print(f"Results (top {top_k}):")
    print(f"  {'#':<3} {'sim':>7}  {'cache_id':<20}  description")
    print(f"  {'---'} {'-------'}  {'--------------------'}  {'----------------------------------------'}")
    for i, r in enumerate(results):
        desc = r.get("task_description", "")[:40]
        cid = r.get("cache_id", "")[:20]
        print(f"  {i + 1:<3} {r['similarity']:>+.4f}  {cid:<20}  {desc}")


def cmd_inspect(args: argparse.Namespace) -> None:
    """Print all fields of .eng files in readable format."""
    from kvcos.engram.reader import read_eigengram

    paths = _resolve_paths(args.eng_files)
    if not paths:
        print("No .eng files found.", file=sys.stderr)
        sys.exit(1)

    for path in paths:
        try:
            rec = read_eigengram(path)
        except Exception as e:
            print(f"  {path}: ERROR - {e}")
            continue

        size = os.path.getsize(path)
        print(f"{'=' * 55}")
        print(f"  File:         {path}  ({size} bytes)")
        print(f"  Format:       EGR1 v{rec['version']}")
        print(f"  Created:      {rec['created_at']} UTC")
        print(f"  Model:        {rec['model_id']}")
        print(f"  cache_id:     {rec['cache_id']}")
        print(f"  Description:  {rec['task_description']}")
        print()
        print(f"  Basis rank:   {rec['basis_rank']}")
        print(f"  N corpus:     {rec['n_corpus']}")
        print(f"  Layer range:  {rec['layer_range']}")
        print(f"  Context len:  {rec['context_len']} KV rows")
        print(f"  L2 norm:      {rec['l2_norm']:.4f}")
        print(f"  SCS:          {rec['scs']:.4f}")
        print(f"  Margin proof: {rec['margin_proof']:.4f}")
        print(f"  Corpus hash:  {rec['corpus_hash']}")
        print(f"  vec_perdoc:   [{rec['vec_perdoc'].shape[0]}] norm={rec['vec_perdoc'].norm():.4f}")
        print(f"  vec_fcdb:     [{rec['vec_fcdb'].shape[0]}] norm={rec['vec_fcdb'].norm():.4f}")
        print()


def cmd_list(args: argparse.Namespace) -> None:
    """Print a one-line summary table of .eng files."""
    from kvcos.engram.reader import read_eigengram

    paths = _resolve_paths(args.eng_files)
    if not paths:
        print("No .eng files found.", file=sys.stderr)
        sys.exit(1)

    hdr = f"{'filename':<30}  {'model':<14}  {'scs':>6}  {'bytes':>5}  description"
    print(hdr)
    print("-" * len(hdr))

    for path in paths:
        fname = os.path.basename(path)[:29]
        try:
            rec = read_eigengram(path)
            size = os.path.getsize(path)
            print(
                f"{fname:<30}  {rec['model_id'][:14]:<14}  "
                f"{rec['scs']:>6.3f}  {size:>5}  "
                f"{rec['task_description'][:40]}"
            )
        except Exception as e:
            print(f"{fname:<30}  ERROR: {e}")


def main() -> None:
    """EIGENGRAM CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m kvcos.engram",
        description="EIGENGRAM CLI - encode and search KV-cache semantic certificates.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="Encode a document as a .eng file.")
    enc.add_argument("--model", required=True, help="Path to GGUF model file.")
    enc.add_argument("--text", help="Document text.")
    enc.add_argument("--file", help="Path to a text file to encode.")
    enc.add_argument("--out", help="Output .eng file path.")
    enc.add_argument("--id", help="Unique cache_id.")
    enc.add_argument("--description", help="Human-readable description.")
    enc.add_argument("--basis", default="results/corpus_basis_fcdb_v2.pt", help="FCDB v2 basis path.")

    srch = sub.add_parser("search", help="Search .eng files with a query.")
    srch.add_argument("--model", required=True, help="GGUF model for query encoding.")
    srch.add_argument("--query", required=True, help="Query text.")
    srch.add_argument("--fingerprint", default="fourier", choices=["perdoc", "fcdb", "fourier"])
    srch.add_argument("--top-k", type=int, default=5, dest="top_k")
    srch.add_argument("--basis", default="results/corpus_basis_fcdb_v2.pt")
    srch.add_argument("eng_files", nargs="+", help=".eng file paths or globs.")

    ins = sub.add_parser("inspect", help="Print all fields of .eng files.")
    ins.add_argument("eng_files", nargs="+")

    lst = sub.add_parser("list", help="Summary table of .eng files.")
    lst.add_argument("eng_files", nargs="+")

    args = parser.parse_args()
    {"encode": cmd_encode, "search": cmd_search, "inspect": cmd_inspect, "list": cmd_list}[args.command](args)


if __name__ == "__main__":
    main()
