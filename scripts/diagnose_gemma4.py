"""
Diagnostic script for Gemma 4 26B-A4B GGUF compatibility with ENGRAM.

Tests:
  1. Model loading + metadata extraction
  2. Basic generation (does it produce coherent output?)
  3. State blob extraction + structure analysis
  4. ENGRAM blob parser compatibility
  5. Full fingerprint pipeline (if blob parsing works)

Usage:
    PYTHONPATH=. .venv/bin/python scripts/diagnose_gemma4.py /path/to/gemma4.gguf
"""

from __future__ import annotations

import struct
import sys
import time
from pathlib import Path


def read_u32(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<I", data, offset)[0], offset + 4


def read_i32(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<i", data, offset)[0], offset + 4


def read_u64(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<Q", data, offset)[0], offset + 8


def inspect_blob_header(blob: bytes) -> dict:
    """Parse just the header/structure of a state blob without assuming F16."""
    info = {}
    offset = 0

    # Architecture string
    str_len, offset = read_u32(blob, offset)
    info["arch"] = blob[offset:offset + str_len].decode("ascii", errors="replace")
    offset += str_len

    # KV stream
    n_stream, offset = read_u32(blob, offset)
    info["n_stream"] = n_stream
    if n_stream != 1:
        info["error"] = f"Expected 1 stream, got {n_stream}"
        return info

    cell_count, offset = read_u32(blob, offset)
    info["cell_count"] = cell_count

    # Skip cell metadata
    for _ in range(cell_count):
        _pos, offset = read_i32(blob, offset)
        n_seq, offset = read_u32(blob, offset)
        for _ in range(n_seq):
            _sid, offset = read_i32(blob, offset)

    # Data header
    v_trans, offset = read_u32(blob, offset)
    info["v_trans"] = bool(v_trans)

    n_layers, offset = read_u32(blob, offset)
    info["n_layers"] = n_layers

    # Inspect first few K layers
    info["k_layer_types"] = []
    info["k_layer_row_sizes"] = []
    for i in range(min(n_layers, 5)):
        type_k, offset = read_i32(blob, offset)
        row_size_k, offset = read_u64(blob, offset)
        info["k_layer_types"].append(type_k)
        info["k_layer_row_sizes"].append(row_size_k)
        # Skip actual data
        data_size = row_size_k * cell_count
        offset += data_size

    info["data_offset_after_k_sample"] = offset
    info["blob_total_size"] = len(blob)

    # GGML type names
    type_names = {0: "F32", 1: "F16", 2: "Q4_0", 8: "Q8_0"}
    info["k_type_names"] = [type_names.get(t, f"unknown({t})") for t in info["k_layer_types"]]

    return info


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_gemma4.py <path-to-gguf>")
        sys.exit(1)

    model_path = sys.argv[1]
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"ENGRAM × Gemma 4 Diagnostic")
    print(f"Model: {model_path}")
    print(f"{'='*60}\n")

    # ── Step 1: Load model ──────────────────────────────────────
    print("STEP 1: Loading model...")
    try:
        from llama_cpp import Llama

        t0 = time.perf_counter()
        llm = Llama(
            model_path=model_path,
            n_ctx=512,       # minimal context for diagnostics
            n_gpu_layers=0,  # CPU for safety
            verbose=False,
        )
        load_s = time.perf_counter() - t0
        print(f"  Loaded in {load_s:.1f}s")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        sys.exit(1)

    # ── Step 2: Read metadata ───────────────────────────────────
    print("\nSTEP 2: Model metadata")
    metadata = llm.metadata
    interesting_keys = [
        "general.name", "general.architecture",
        "llama.block_count", "general.block_count",
        "llama.attention.head_count", "llama.attention.head_count_kv",
        "llama.embedding_length", "llama.context_length",
        "llama.expert_count", "llama.expert_used_count",
        "gemma.block_count", "gemma.attention.head_count",
        "gemma.attention.head_count_kv", "gemma.embedding_length",
    ]
    for key in interesting_keys:
        val = metadata.get(key)
        if val is not None:
            print(f"  {key}: {val}")

    # Also dump any keys containing "expert" or "moe"
    for key, val in sorted(metadata.items()):
        if "expert" in key.lower() or "moe" in key.lower():
            print(f"  {key}: {val}")

    # Derive spec parameters
    n_layers = int(metadata.get("llama.block_count", metadata.get("gemma.block_count", metadata.get("general.block_count", "0"))))
    n_heads = int(metadata.get("llama.attention.head_count", metadata.get("gemma.attention.head_count", "0")))
    n_kv_heads = int(metadata.get("llama.attention.head_count_kv", metadata.get("gemma.attention.head_count_kv", str(n_heads))))
    embed_dim = int(metadata.get("llama.embedding_length", metadata.get("gemma.embedding_length", "0")))
    head_dim = embed_dim // n_heads if n_heads > 0 else 0

    print(f"\n  Derived spec:")
    print(f"    n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}")
    print(f"    embed_dim={embed_dim}, head_dim={head_dim}")
    print(f"    n_embd_kv = {n_kv_heads * head_dim}")

    # ── Step 3: Generate ────────────────────────────────────────
    print("\nSTEP 3: Basic generation")
    try:
        t0 = time.perf_counter()
        output = llm("Hello, my name is", max_tokens=20, temperature=0.0)
        gen_ms = (time.perf_counter() - t0) * 1000
        text = output["choices"][0]["text"]
        print(f"  Generated in {gen_ms:.0f}ms")
        print(f"  Output: {text[:200]}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        print("  Continuing anyway (bartowski warned about conversion issues)...")

    # ── Step 4: State blob extraction ───────────────────────────
    print("\nSTEP 4: State blob extraction")
    try:
        state_data = llm.save_state()
        blob = bytes(state_data.llama_state)
        print(f"  Blob size: {len(blob):,} bytes ({len(blob)/1024/1024:.1f} MB)")

        # Inspect structure without assuming F16
        info = inspect_blob_header(blob)
        print(f"  Architecture: {info.get('arch', '?')}")
        print(f"  Cell count: {info.get('cell_count', '?')}")
        print(f"  V transposed: {info.get('v_trans', '?')}")
        print(f"  N layers: {info.get('n_layers', '?')}")
        print(f"  K dtype (first 5 layers): {info.get('k_type_names', [])}")
        print(f"  K row sizes (first 5): {info.get('k_layer_row_sizes', [])}")

        if info.get("k_layer_row_sizes"):
            row = info["k_layer_row_sizes"][0]
            cells = info["cell_count"]
            elements_per_row = row // 2  # assuming F16
            expected_embd_kv = n_kv_heads * head_dim
            print(f"\n  Row analysis:")
            print(f"    row_size={row}, cells={cells}")
            print(f"    elements_per_cell (if F16) = {row // 2}")
            print(f"    expected n_embd_kv = {expected_embd_kv}")
            if elements_per_row == expected_embd_kv:
                print(f"    MATCH: row elements == n_kv_heads * head_dim")
            else:
                print(f"    MISMATCH: {elements_per_row} != {expected_embd_kv}")
                # Check if it matches with different assumptions
                for dtype_name, dtype_size in [("F32", 4.0), ("F16", 2.0), ("Q8_0", 34/32), ("Q4_0", 18/32)]:
                    if row / dtype_size == expected_embd_kv:
                        print(f"    → Would match with dtype {dtype_name}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Step 5: ENGRAM blob parser ──────────────────────────────
    print("\nSTEP 5: ENGRAM blob parser")
    if n_kv_heads == 0 or head_dim == 0:
        print("  SKIPPED: could not derive n_kv_heads/head_dim from metadata")
    else:
        try:
            from kvcos.core.blob_parser import parse_state_blob
            parsed = parse_state_blob(blob, n_kv_heads=n_kv_heads, head_dim=head_dim)
            print(f"  SUCCESS!")
            print(f"  Keys shape: {parsed.keys.shape}")
            print(f"  Values shape: {parsed.values.shape}")
            print(f"  N cells: {parsed.n_cells}")
            print(f"  N layers: {parsed.n_layers}")
            print(f"  Arch: {parsed.arch}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            print("  This is where we need to fix compatibility.")

    # ── Step 6: Fourier fingerprint ─────────────────────────────
    print("\nSTEP 6: Fourier fingerprint (if blob parsed)")
    try:
        parsed  # check it exists
        from kvcos.core.fingerprint import compute_fourier_fingerprint_v2
        layer_keys = parsed.keys.float().mean(dim=2)  # [layers, heads, dim]
        fp = compute_fourier_fingerprint_v2(layer_keys, freqs=[0, 1])
        print(f"  Fingerprint shape: {fp.shape}")
        print(f"  Norm: {fp.norm():.4f}")
        print(f"  First 5 values: {fp[:5].tolist()}")
    except NameError:
        print("  SKIPPED: blob parsing failed")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print("Diagnostic complete.")


if __name__ == "__main__":
    main()
