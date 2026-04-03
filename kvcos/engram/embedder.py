"""
kvcos/engram/embedder.py — Unified text-to-fingerprint embedding.

Three strategies, tried in priority order:
  1. llama_cpp:  Native ENGRAM KV-cache Fourier pipeline (2048-dim)
  2. sbert:      Sentence-transformers all-MiniLM-L6-v2 (384-dim)
  3. hash:       Deterministic SHA256-seeded pseudo-fingerprint (2048-dim)

The chosen strategy is cached after first call. The fingerprint
source tag travels with every .eng file so retrieval knows what
comparison is valid.

Usage:
    from kvcos.engram.embedder import get_fingerprint
    fp, source = get_fingerprint("some text")
    # fp: torch.Tensor, source: "llama_cpp"|"sbert"|"hash-fallback"
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol

import numpy as np
import torch


class Embedder(Protocol):
    """Protocol for text → fingerprint embedding."""
    def embed(self, text: str) -> torch.Tensor: ...
    @property
    def source(self) -> str: ...
    @property
    def dim(self) -> int: ...


# ── Strategy 1: Native ENGRAM (llama_cpp) ────────────────────────────

class LlamaCppEmbedder:
    """KV-cache Fourier fingerprint via local GGUF model.

    Uses the full ENGRAM pipeline:
      text → LlamaCppBridge (generate → KV cache) → Fourier DFT → fingerprint

    Supports both standard and ISWA models:
      Standard (Llama):  2048-dim (8 × 128 × 2)
      ISWA (Gemma 4):    6144-dim (1024×2 + 2048×2)
    """

    def __init__(self, model_path: str) -> None:
        from integrations.llama_cpp_bridge import LlamaCppBridge
        from kvcos.core.cache_spec import is_iswa_spec
        from kvcos.core.fingerprint import compute_fourier_fingerprint_v2, compute_iswa_fingerprint

        self._bridge = LlamaCppBridge(
            model_path,
            n_ctx=2048,
            n_gpu_layers=0,
            verbose=False,
        )
        self._spec = self._bridge.load_model()
        self._is_iswa = is_iswa_spec(self._spec)
        self._compute_standard = compute_fourier_fingerprint_v2
        self._compute_iswa = compute_iswa_fingerprint

        if self._is_iswa:
            sections = self._spec["cache_sections"]
            self._dim = sum(s.n_kv_heads * s.head_dim * 2 for s in sections)
        else:
            self._dim = self._spec["n_kv_heads"] * self._spec["head_dim"] * 2

    def embed(self, text: str) -> torch.Tensor:
        """Generate text through model, extract KV keys, compute Fourier fp."""
        self._bridge.llm.reset()
        self._bridge.generate(text, max_tokens=1)

        if self._is_iswa:
            parsed = self._bridge.extract_kv_cache_iswa()
            return self._compute_iswa(parsed, freqs=[0, 1])

        parsed = self._bridge.extract_kv_cache()
        layer_keys = parsed.keys.float().mean(dim=2)
        return self._compute_standard(layer_keys, freqs=[0, 1])

    @property
    def source(self) -> str:
        return "llama_cpp"

    @property
    def dim(self) -> int:
        return self._dim


# ── Strategy 2: Sentence-transformers ────────────────────────────────

class SBertEmbedder:
    """Semantic fingerprint via sentence-transformers.

    Uses all-MiniLM-L6-v2 (80MB, 384-dim). Downloads on first use.
    Subsequent calls use the cached model (~50ms per text on CPU).
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        import logging
        import warnings
        # Suppress noisy HF/tokenizer/sbert/safetensors warnings on load
        for name in (
            "sentence_transformers",
            "transformers",
            "transformers.modeling_utils",
            "huggingface_hub",
            "huggingface_hub.utils",
            "safetensors",
        ):
            logging.getLogger(name).setLevel(logging.CRITICAL)
        # Suppress the HF_TOKEN and load-report warnings
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        warnings.filterwarnings("ignore", category=FutureWarning)
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.MODEL_NAME)
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> torch.Tensor:
        # encode returns numpy array
        vec = self._model.encode(text, normalize_embeddings=True)
        return torch.from_numpy(vec.astype(np.float32))

    @property
    def source(self) -> str:
        return "sbert"

    @property
    def dim(self) -> int:
        return self._dim


# ── Strategy 3: Hash fallback ────────────────────────────────────────

class HashEmbedder:
    """Deterministic pseudo-fingerprint from SHA256 hash.

    No semantic meaning — same text always maps to same vector,
    but unrelated texts have random cosine similarity (~0).
    """

    def __init__(self, dim: int = 2048) -> None:
        self._dim = dim

    def embed(self, text: str) -> torch.Tensor:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        fp = rng.randn(self._dim).astype("float32")
        fp /= np.linalg.norm(fp) + 1e-8
        return torch.from_numpy(fp)

    @property
    def source(self) -> str:
        return "hash-fallback"

    @property
    def dim(self) -> int:
        return self._dim


# ── Singleton factory ────────────────────────────────────────────────

_cached_embedder: Embedder | None = None


def _create_embedder() -> Embedder:
    """Try strategies in priority order, return first that works."""

    # Strategy 1: llama_cpp
    model_path = os.environ.get("ENGRAM_MODEL_PATH", "")
    if model_path and Path(model_path).exists():
        try:
            return LlamaCppEmbedder(model_path)
        except Exception:
            pass

    # Strategy 2: sentence-transformers
    try:
        embedder = SBertEmbedder()
        return embedder
    except Exception:
        pass

    # Strategy 3: hash fallback (always works)
    return HashEmbedder()


def get_embedder() -> Embedder:
    """Get the cached embedder singleton."""
    global _cached_embedder
    if _cached_embedder is None:
        _cached_embedder = _create_embedder()
    return _cached_embedder


def get_fingerprint(text: str) -> tuple[torch.Tensor, str]:
    """
    Compute fingerprint for text using best available strategy.

    Returns:
        (fingerprint_tensor, source_tag)
    """
    embedder = get_embedder()
    fp = embedder.embed(text)
    return fp, embedder.source


def reset_embedder() -> None:
    """Reset the cached embedder (for testing or strategy switching)."""
    global _cached_embedder
    _cached_embedder = None
