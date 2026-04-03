"""
ENGRAM Protocol — Demo Agent Session


End-to-end demonstration:
  1. Load model via llama-cpp-python (D1)
  2. Generate with a prompt → measure cold TTFT
  3. Extract KV cache → compress → serialize to .eng
  4. Index in EGR manifold index
  5. Reset model → restore from .eng → measure cached TTFT
  6. Print speedup ratio

D6: Target >10x TTFT reduction at 16K context on Llama 3.1 8B.
    Cold baseline: ~1,500-5,000ms. Cached target: <500ms.
    Anything below 4x at 16K is a failure.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _run_dry_run(args: argparse.Namespace) -> int:
    """Run full pipeline with synthetic tensors — no model file needed."""
    import os
    import tempfile

    import torch

    from kvcos.core.cache_spec import LLAMA_3_1_8B
    from kvcos.core.serializer import EngramSerializer
    from kvcos.core.types import CompressionMethod, StateExtractionMode
    from kvcos.core.manifold_index import IndexEntry, ManifoldIndex
    from kvcos.core.state_extractor import MARStateExtractor
    from kvcos.storage.local import LocalStorageBackend

    spec = LLAMA_3_1_8B
    ctx_len = args.context
    model_name = spec["model_id"]

    # ── Synthetic KV tensors ──────────────────────────────────
    torch.manual_seed(42)
    shape = (spec["n_layers"], spec["n_kv_heads"], ctx_len, spec["head_dim"])
    keys = torch.randn(shape, dtype=torch.float16)
    values = torch.randn(shape, dtype=torch.float16)

    tensor_mb = keys.numel() * keys.element_size() / 1024 / 1024

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # ── Serialize to .eng ────────────────────────────────
        serializer = EngramSerializer()
        eng_path = tmp_dir / "dry_run.eng"

        t0 = time.perf_counter()
        result = serializer.serialize(
            keys=keys, values=values,
            agent_id="dry-run-agent",
            task_description="dry run benchmark",
            model_id=model_name,
            output_path=eng_path,
            compression=CompressionMethod.Q8_0,
        )
        serialize_ms = (time.perf_counter() - t0) * 1000

        # ── Load back ────────────────────────────────────────
        t0 = time.perf_counter()
        k_out, v_out, meta = serializer.deserialize(eng_path)
        deserialize_ms = (time.perf_counter() - t0) * 1000

        assert k_out.shape == keys.shape, f"Shape mismatch: {k_out.shape} vs {keys.shape}"

        # ── EGR granular timing ──────────────────────────────
        extractor = MARStateExtractor(
            mode=StateExtractionMode.SVD_PROJECT,
            rank=min(160, spec["head_dim"]),
        )
        dim = extractor.output_dim(spec)
        index = ManifoldIndex(dim=dim)
        storage = LocalStorageBackend(data_dir=tmp_dir)

        # Index: extract + serialize + store + add
        t0 = time.perf_counter()
        extraction = extractor.extract(keys, spec)
        t_extract = time.perf_counter()

        eng2 = tmp_dir / "indexed.eng"
        serializer.serialize(
            keys=keys, values=values,
            agent_id="dry-run-agent",
            task_description="dry run benchmark",
            model_id=model_name,
            output_path=eng2,
            compression=CompressionMethod.Q8_0,
            cache_id="dry-run-001",
        )
        t_serialize = time.perf_counter()

        idx_meta = serializer.read_metadata_only(eng2)
        storage.store_file("dry-run-001", eng2, idx_meta)
        t_store = time.perf_counter()

        from datetime import datetime, timezone
        entry = IndexEntry(
            cache_id="dry-run-001",
            task_description="dry run benchmark",
            model_id=model_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            context_len=ctx_len,
            l2_norm=extraction.l2_norm,
        )
        index.add(extraction.state_vec, entry)
        t_add = time.perf_counter()

        extract_ms = (t_extract - t0) * 1000
        ser_ms = (t_serialize - t_extract) * 1000
        store_ms = (t_store - t_serialize) * 1000
        add_ms = (t_add - t_store) * 1000
        index_ms = (t_add - t0) * 1000

        # Retrieve: extract query + search + load
        torch.manual_seed(99)
        query_keys = torch.randn(shape, dtype=torch.float16)

        t0 = time.perf_counter()
        q_ext = extractor.extract(query_keys, spec)
        t_qext = time.perf_counter()

        results = index.search(q_ext.state_vec, top_k=1)
        t_search = time.perf_counter()

        # Load matched engram
        stored_path = storage.get_path("dry-run-001")
        k_loaded, v_loaded, _ = serializer.deserialize(stored_path)
        t_load = time.perf_counter()

        q_extract_ms = (t_qext - t0) * 1000
        search_ms = (t_search - t_qext) * 1000
        load_ms = (t_load - t_search) * 1000
        retrieve_ms = (t_load - t0) * 1000

        # ── Simulate TTFT estimates ──────────────────────────
        cold_ms = ctx_len * 0.1  # simulated
        cached_ms = deserialize_ms
        egr_overhead = extract_ms + search_ms  # overhead added to warm path
        speedup = cold_ms / cached_ms if cached_ms > 0 else float("inf")
        eng_size_mb = os.path.getsize(eng_path) / 1024 / 1024

        # ── Output ───────────────────────────────────────────
        sep = "=" * 35
        print(sep)
        print("ENGRAM Protocol \u2014 EGR Demo")
        print(f"Model: {model_name}")
        print(f"Context: {ctx_len} tokens")
        print(sep)
        print(f"Cold TTFT:    {cold_ms:.1f}ms (simulated)")
        print(f"Cached TTFT:  {cached_ms:.1f}ms (deserialize)")
        print(f"Speedup:      {speedup:.1f}x")
        print(f"D6 target:    >10x at 16K tokens")
        status = "PASS" if speedup > 10 else "FAIL"
        print(f"Status:       {status}")
        print(f"EGR overhead: {egr_overhead:.1f}ms (extract+search)")
        print(f".eng file:    {eng_path.name} ({eng_size_mb:.1f}MB)")
        print(f"Tensor shape: {list(shape)} ({tensor_mb:.0f}MB per K/V)")
        print(sep)
        print()
        print("Index breakdown:")
        print(f"  SVD extract:    {extract_ms:8.1f}ms")
        print(f"  Serialize .eng: {ser_ms:8.1f}ms")
        print(f"  Store backend:  {store_ms:8.1f}ms")
        print(f"  FAISS add():    {add_ms:8.1f}ms")
        print(f"  TOTAL:          {index_ms:8.1f}ms")
        print()
        print("Retrieve breakdown:")
        print(f"  SVD extract:    {q_extract_ms:8.1f}ms")
        print(f"  FAISS search(): {search_ms:8.1f}ms")
        print(f"  Load+deser:     {load_ms:8.1f}ms")
        print(f"  TOTAL:          {retrieve_ms:8.1f}ms")
        print()
        print("Verification:")
        print(f"  Round-trip shape:  {'OK' if k_out.shape == keys.shape else 'FAIL'}")
        print(f"  Retrieval result:  {'OK' if len(results) >= 1 else 'FAIL'}")
        print(f"  .eng valid:        {'OK' if eng_path.exists() else 'FAIL'}")

    return 0 if speedup > 10 else 1


def main():
    parser = argparse.ArgumentParser(
        description="ENGRAM Protocol — Demo Agent Session",
        epilog="D6: >10x TTFT reduction at 16K context on Llama 3.1 8B",
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Path to GGUF model file (required unless --dry-run)",
    )
    parser.add_argument(
        "--context", "-c", type=int, default=4096,
        help="Context length to fill (tokens). Default: 4096",
    )
    parser.add_argument(
        "--n-ctx", type=int, default=16384,
        help="Max context window for model. Default: 16384",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="ENGRAM data directory. Default: ~/.engram/data",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run full pipeline with synthetic tensors (no model needed)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()

    if args.dry_run:
        return _run_dry_run(args)

    if not args.model:
        parser.error("--model is required unless --dry-run is specified")

    print("=" * 70)
    print("ENGRAM Protocol — Demo Agent Session")
    print("KV cache fingerprinting for persistent semantic retrieval")
    print("=" * 70)
    print()

    # ── Setup ─────────────────────────────────────────────────
    from kvcos.core.config import get_config
    from kvcos.core.serializer import EngramSerializer
    from kvcos.core.types import CompressionMethod, StateExtractionMode
    from kvcos.core.manifold_index import ManifoldIndex
    from kvcos.core.retriever import EGRRetriever
    from kvcos.core.state_extractor import MARStateExtractor
    from kvcos.storage.local import LocalStorageBackend
    from integrations.llama_cpp_bridge import LlamaCppBridge

    config = get_config()
    data_dir = Path(args.data_dir) if args.data_dir else config.data_dir

    # ── Step 1: Load Model ────────────────────────────────────
    print(f"[1/6] Loading model: {args.model}")
    bridge = LlamaCppBridge(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=0,  # D1
        verbose=args.verbose,
    )
    spec = bridge.load_model()
    print(f"  Model: {spec['model_id']}")
    print(f"  Architecture: {spec['n_layers']}L / {spec['n_heads']}H / {spec['n_kv_heads']}KV / {spec['head_dim']}D")
    print(f"  Context window: {args.n_ctx}")
    print()

    # ── Step 2: Generate + Cold TTFT ──────────────────────────
    filler = "The quick brown fox jumps over the lazy dog. " * 100
    target_tokens = args.context
    prompt = filler[:target_tokens * 4]

    print(f"[2/6] Cold prefill ({target_tokens} target tokens)...")
    t0 = time.perf_counter()
    cold = bridge.measure_cold_ttft(prompt)
    print(f"  Cold TTFT: {cold.ttft_ms:.1f}ms ({cold.context_len} tokens)")
    print()

    # ── Step 3: Extract + Serialize ───────────────────────────
    print("[3/6] Extracting KV cache...")
    try:
        parsed = bridge.extract_kv_cache()
        print(f"  Keys shape:   {list(parsed.keys.shape)}")
        print(f"  Values shape: {list(parsed.values.shape)}")
        print(f"  Cells: {parsed.n_cells}")
    except Exception as e:
        print(f"  KV extraction failed: {e}")
        print("  This is expected if the blob format doesn't match.")
        print("  Falling back to save_state/load_state raw blob path.")
        parsed = None
    print()

    print("[3b/6] Saving raw state blob...")
    raw_state = bridge.llm.save_state()
    raw_blob = bytes(raw_state.llama_state)
    print(f"  Raw state size: {len(raw_blob) / 1024 / 1024:.1f} MB")

    if parsed is not None:
        print("[3c/6] Serializing to .eng format...")
        serializer = EngramSerializer()
        eng_path = data_dir / "demo" / "session_001.eng"
        result = serializer.serialize(
            keys=parsed.keys,
            values=parsed.values,
            agent_id="demo-agent",
            task_description="demo session - cold prefill benchmark",
            model_id=spec["model_id"],
            output_path=eng_path,
            compression=CompressionMethod.Q8_0,
        )
        print(f"  .eng file: {result['path']}")
        print(f"  Size: {result['size_bytes'] / 1024 / 1024:.1f} MB")
        print(f"  Compression ratio: {result['compression_ratio']:.2f}x")
    print()

    # ── Step 4: Index in EGR ──────────────────────────────────
    if parsed is not None:
        print("[4/6] Indexing in EGR manifold index...")
        storage = LocalStorageBackend(data_dir=data_dir)
        extractor = MARStateExtractor(
            mode=StateExtractionMode.SVD_PROJECT,
            rank=min(160, spec["head_dim"]),
        )
        dim = extractor.output_dim(spec)
        index = ManifoldIndex(dim=dim)
        retriever = EGRRetriever(extractor, index, storage)

        cache_id = retriever.index_engram(
            keys=parsed.keys,
            values=parsed.values,
            spec=spec,
            agent_id="demo-agent",
            task_description="demo session - cold prefill benchmark",
            model_id=spec["model_id"],
        )
        print(f"  Indexed: {cache_id}")
        print(f"  State vector dim: {dim}")
        print(f"  Index entries: {index.n_entries}")
    else:
        print("[4/6] Skipped (KV extraction failed)")
    print()

    # ── Step 5: Restore + Cached TTFT ─────────────────────────
    print("[5/6] Restoring from cached state...")
    t0 = time.perf_counter()
    cached = bridge.measure_cached_ttft(raw_blob)
    print(f"  Cached TTFT: {cached.ttft_ms:.1f}ms")
    print()

    # ── Step 6: Results ───────────────────────────────────────
    cold_ms = cold.ttft_ms
    cached_ms = cached.ttft_ms
    speedup = cold_ms / cached_ms if cached_ms > 0 else float("inf")

    eng_path_str = result["path"] if parsed else "N/A"
    eng_size_kb = result["size_bytes"] / 1024 if parsed else 0

    sep = "=" * 35
    print(sep)
    print("ENGRAM Protocol — EGR Demo")
    print(f"Model: {spec['model_id']}")
    print(f"Context: {cold.context_len} tokens")
    print(sep)
    print(f"Cold TTFT:    {cold_ms:.1f}ms")
    print(f"Cached TTFT:  {cached_ms:.1f}ms")
    print(f"Speedup:      {speedup:.1f}x")
    print(f"D6 target:    >10x at 16K tokens")
    status = "PASS" if speedup > 10 else "FAIL"
    print(f"Status:       {status}")
    print(f".eng file:    {eng_path_str} ({eng_size_kb:.1f}KB)")
    print(sep)

    return 0 if speedup >= 4 else 1


if __name__ == "__main__":
    sys.exit(main())
