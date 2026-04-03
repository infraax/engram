"""
EIGENGRAM writer: text + model -> .eng file
"""

from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path

import torch
from llama_cpp import Llama

from kvcos.core.blob_parser import parse_state_blob
from .format import EigramEncoder

_encoder = EigramEncoder()

_DEFAULT_LR = (8, 24)
_DEFAULT_GATE = 6
_DEFAULT_RANK = 116


def _get_model_id(model_path: str) -> str:
    name = os.path.basename(model_path)
    if "3B" in name or "3b" in name:
        return "Llama-3.2-3B"
    if "8B" in name or "8b" in name:
        return "Llama-3.1-8B"
    return name[:15]


def _corpus_hash(basis_path: str) -> str:
    raw = Path(basis_path).read_bytes()
    return hashlib.sha256(raw).hexdigest()[:32]


def write_eigengram(
    model_path: str,
    text: str,
    output_path: str,
    cache_id: str = "",
    task_description: str = "",
    layer_range: tuple[int, int] = _DEFAULT_LR,
    gate: int = _DEFAULT_GATE,
    rank_perdoc: int = _DEFAULT_RANK,
    basis_path: str = "results/corpus_basis_fcdb_v2.pt",
) -> dict:
    """Encode a document as an EIGENGRAM file."""
    saved = torch.load(basis_path, weights_only=False)
    P_fcdb = saved["basis"]
    center = saved["joint_center"]
    n_corpus = int(saved["n_docs"])
    basis_rank = P_fcdb.shape[0]

    llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
    meta = llm.metadata
    n_kv = int(meta.get("llama.attention.head_count_kv", "8"))
    hd = int(meta.get("llama.embedding_length", "4096")) // int(
        meta.get("llama.attention.head_count", "32")
    )

    llm.reset()
    llm(text.strip(), max_tokens=1, temperature=0.0)
    state_bytes = bytes(llm.save_state().llama_state)
    del llm
    gc.collect()

    p = parse_state_blob(state_bytes, n_kv_heads=n_kv, head_dim=hd)
    l0, l1 = layer_range
    k = p.keys[l0:l1].float().reshape(-1, hd)
    mean_v = k.mean(0)
    l2_norm = float(mean_v.norm().item())

    # Per-doc SVD fingerprint
    if k.shape[0] > 8192:
        gen = torch.Generator()
        gen.manual_seed(42)
        idx = torch.randperm(k.shape[0], generator=gen)[:8192]
        svd_input = k[idx]
    else:
        svd_input = k
    _, _, Vh = torch.linalg.svd(svd_input, full_matrices=False)
    proj = (svd_input @ Vh[gate : gate + rank_perdoc].T).mean(0)
    vec_perdoc = proj / (proj.norm() + 1e-8)

    # FCDB fingerprint
    delta = mean_v - center
    delta = delta / (delta.norm() + 1e-8)
    vec_fcdb = delta @ P_fcdb.T
    vec_fcdb = vec_fcdb / (vec_fcdb.norm() + 1e-8)

    # SCS
    scs = float(
        ((delta @ P_fcdb.T @ P_fcdb) ** 2).sum().item()
        / ((delta**2).sum().item() + 1e-12)
    )

    corpus_h = _corpus_hash(basis_path)
    model_id = _get_model_id(model_path)

    cert = _encoder.encode(
        vec_perdoc=vec_perdoc,
        vec_fcdb=vec_fcdb,
        joint_center=center,
        corpus_hash=corpus_h,
        model_id=model_id,
        basis_rank=basis_rank,
        n_corpus=n_corpus,
        layer_range=layer_range,
        context_len=int(k.shape[0]),
        l2_norm=l2_norm,
        scs=scs,
        margin_proof=0.0,
        task_description=task_description or text[:100],
        cache_id=cache_id or "",
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(cert)

    return {
        "output_path": output_path,
        "model_id": model_id,
        "corpus_hash": corpus_h,
        "basis_rank": basis_rank,
        "n_corpus": n_corpus,
        "file_size_bytes": len(cert),
        "scs": round(scs, 4),
        "l2_norm": round(l2_norm, 4),
        "layer_range": layer_range,
    }
