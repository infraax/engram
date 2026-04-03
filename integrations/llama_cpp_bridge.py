"""
ENGRAM Protocol — llama-cpp-python Bridge


D1: llama-cpp-python direct. No Ollama. n_gpu_layers=0 for Phase 1.

Provides:
  - KV cache extraction via llama_state_seq_get_data() → blob_parser
  - KV cache injection via llama_state_seq_set_data() for session restore
  - TTFT measurement for benchmarking (D6: >10x at 16K)
  - Model loading with architecture spec auto-detection

WARNING: State blob format is llama.cpp version-dependent.
Pin llama-cpp-python version in pyproject.toml.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

from kvcos.core.blob_parser import (
    GGML_TYPE_F16,
    GGML_TYPE_Q8_0,
    ParsedKVCache,
    ParsedMultiSectionCache,
    parse_multi_section_blob,
    parse_state_blob,
)
from kvcos.core.cache_spec import (
    ModelCacheSpec,
    get_model_spec,
    is_iswa_spec,
    make_spec_from_metadata,
)


# Metadata key prefixes in order of preference per architecture.
# llama.cpp uses architecture-specific keys (e.g., gemma4.block_count).
_METADATA_PREFIXES = ("llama", "gemma4", "gemma", "phi", "qwen", "mistral", "deepseek")


def _meta_get(metadata: dict, key_suffix: str, default: str = "0") -> str:
    """Get a metadata value trying architecture-specific prefixes.

    Searches: llama.{suffix}, gemma4.{suffix}, gemma.{suffix}, etc.
    Falls back to general.{suffix}, then default.

    Args:
        metadata: llama.cpp model metadata dict.
        key_suffix: Key without prefix, e.g. "block_count" or "attention.head_count".
        default: Default if no key found.
    """
    for prefix in _METADATA_PREFIXES:
        val = metadata.get(f"{prefix}.{key_suffix}")
        if val is not None:
            return val
    # Fall back to general.*
    val = metadata.get(f"general.{key_suffix}")
    return val if val is not None else default


@dataclass
class TTFTMeasurement:
    """Time-to-first-token measurement for benchmarking."""

    ttft_ms: float  # milliseconds
    context_len: int
    method: str  # "cold_prefill" or "cached_restore"
    model_id: str


class LlamaCppBridge:
    """Bridge between llama-cpp-python and ENGRAM's KV cache system.

    Handles model loading, KV cache extraction, and injection.

    Usage:
        bridge = LlamaCppBridge("/path/to/model.gguf")
        bridge.load_model()

        # Generate and extract KV state
        bridge.generate(prompt)
        parsed = bridge.extract_kv_cache()

        # Later: inject cached state
        bridge.inject_kv_cache(cached_blob, spec)
        bridge.generate("Continue from cached state:")
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 16384,
        n_gpu_layers: int = 0,  # D1: CPU-only Phase 1
        kv_cache_type: str = "f16",  # "f16" or "q8_0"
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.kv_cache_type = kv_cache_type
        self.verbose = verbose
        self._llm = None
        self._spec: ModelCacheSpec | None = None

    def load_model(self) -> ModelCacheSpec:
        """Load the GGUF model and auto-detect architecture spec.

        Returns the ModelCacheSpec for this model.
        """
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
        )

        # Auto-detect model architecture from llama.cpp metadata.
        # Uses fallback chain across architecture prefixes (llama.*, gemma4.*, etc.)
        metadata = self._llm.metadata
        model_name = metadata.get("general.name", Path(self.model_path).stem)

        # Check registry first (handles ISWA specs with cache_sections)
        registry_spec = get_model_spec(model_name)
        if registry_spec is not None:
            self._spec = registry_spec
        else:
            n_layers = int(_meta_get(metadata, "block_count", "32"))
            n_heads = int(_meta_get(metadata, "attention.head_count", "32"))
            n_kv_heads = int(_meta_get(metadata, "attention.head_count_kv", str(n_heads)))
            embed_dim = int(_meta_get(metadata, "embedding_length", "4096"))
            head_dim = embed_dim // n_heads if n_heads > 0 else 128

            self._spec = make_spec_from_metadata(
                model_id=model_name,
                n_layers=n_layers,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                rope_enabled=True,
            )

        if self.verbose:
            logger.info("Loaded model: %s", model_name)
            logger.info(
                "  Layers: %d, KV Heads: %d, Head Dim: %d",
                self._spec["n_layers"], self._spec["n_kv_heads"], self._spec["head_dim"],
            )
            logger.info("  Context: %d, GPU Layers: %d", self.n_ctx, self.n_gpu_layers)
            if is_iswa_spec(self._spec):
                sections = self._spec["cache_sections"]
                logger.info("  ISWA: %d cache sections", len(sections))
                for i, s in enumerate(sections):
                    logger.info(
                        "    Section %d: %s — %d layers, %d KV heads, head_dim=%d",
                        i, s.attention_type, s.n_layers, s.n_kv_heads, s.head_dim,
                    )

        return self._spec

    @property
    def spec(self) -> ModelCacheSpec:
        if self._spec is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._spec

    @property
    def llm(self):
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._llm

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1,
        temperature: float = 0.0,
    ) -> tuple[str, float]:
        """Generate tokens and return (output_text, ttft_ms).

        With max_tokens=1, this effectively does a prefill + one decode step,
        which is what we need for TTFT measurement.
        """
        t0 = time.perf_counter()
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        t1 = time.perf_counter()

        ttft_ms = (t1 - t0) * 1000
        text = output["choices"][0]["text"]
        return text, ttft_ms

    def extract_kv_cache(self, seq_id: int = 0) -> ParsedKVCache:
        """Extract the current KV cache as structured tensors.

        For standard models: returns ParsedKVCache.
        For ISWA models: parses only the first (global) section.
        Use extract_kv_cache_iswa() for full multi-section extraction.

        Args:
            seq_id: Sequence ID to extract (default 0 for single-sequence use)

        Returns:
            ParsedKVCache with [n_layers, n_kv_heads, seq_len, head_dim] tensors
        """
        state_data = self.llm.save_state()
        blob = bytes(state_data.llama_state)

        if is_iswa_spec(self.spec):
            # For backward compat, parse just the first section
            sections = self.spec["cache_sections"]
            first = sections[0]
            return parse_state_blob(
                blob,
                n_kv_heads=first.n_kv_heads,
                head_dim=first.head_dim,
            )

        return parse_state_blob(
            blob,
            n_kv_heads=self.spec["n_kv_heads"],
            head_dim=self.spec["head_dim"],
        )

    def extract_kv_cache_iswa(self) -> ParsedMultiSectionCache:
        """Extract all ISWA cache sections as structured tensors.

        Only valid for ISWA models (those with cache_sections in spec).

        Returns:
            ParsedMultiSectionCache with one ParsedKVCache per section.

        Raises:
            RuntimeError: If model is not ISWA.
        """
        if not is_iswa_spec(self.spec):
            raise RuntimeError(
                f"extract_kv_cache_iswa() requires an ISWA model, "
                f"but {self.spec['model_id']} has no cache_sections"
            )

        state_data = self.llm.save_state()
        blob = bytes(state_data.llama_state)

        return parse_multi_section_blob(blob, self.spec["cache_sections"])

    def inject_kv_cache(self, state_data: bytes) -> float:
        """Inject a previously saved KV cache state, returning restore time in ms.

        Args:
            state_data: Raw state blob (as returned by save_state / extracted earlier)

        Returns:
            Restore time in milliseconds
        """
        from llama_cpp import LlamaState

        t0 = time.perf_counter()

        state = LlamaState(
            input_ids=[],  # Will be overridden by the state
            scores=[],
            llama_state=list(state_data),
            llama_state_size=len(state_data),
        )
        self.llm.load_state(state)

        t1 = time.perf_counter()
        return (t1 - t0) * 1000

    def measure_cold_ttft(self, prompt: str) -> TTFTMeasurement:
        """Measure cold TTFT (full prefill from scratch).

        Resets the KV cache before generation.
        """
        self.llm.reset()

        tokens = self.llm.tokenize(prompt.encode())
        _, ttft_ms = self.generate(prompt, max_tokens=1)

        return TTFTMeasurement(
            ttft_ms=ttft_ms,
            context_len=len(tokens),
            method="cold_prefill",
            model_id=self.spec["model_id"],
        )

    def measure_cached_ttft(self, state_data: bytes, continuation: str = " ") -> TTFTMeasurement:
        """Measure cached TTFT (restore from saved state + generate).

        Args:
            state_data: Saved state blob to restore from
            continuation: Text to generate after restore

        Returns:
            TTFTMeasurement with restore + first token time
        """
        self.llm.reset()

        t0 = time.perf_counter()
        self.inject_kv_cache(state_data)
        output = self.llm(continuation, max_tokens=1, temperature=0.0)
        t1 = time.perf_counter()

        ttft_ms = (t1 - t0) * 1000

        return TTFTMeasurement(
            ttft_ms=ttft_ms,
            context_len=0,  # Not re-prefilling
            method="cached_restore",
            model_id=self.spec["model_id"],
        )

    def close(self) -> None:
        """Release model resources."""
        self._llm = None
        self._spec = None
