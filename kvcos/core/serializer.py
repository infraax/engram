"""
ENGRAM Protocol — .eng File Serializer


.eng = safetensors container with:
  - __metadata__: JSON-stringified EngramMetadata (all string values per D7)
  - Tensor keys: layer_{i}_keys, layer_{i}_values
  - Each tensor: [n_kv_heads, ctx_len, head_dim] at compressed dtype

D7: safetensors confirmed. GGUF rejected. String-only metadata values.
Reference: arXiv:2603.04428 uses identical safetensors approach.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from kvcos.core.cache_spec import infer_model_family
from kvcos.core.compression import CompressionResult, compress, decompress
from kvcos.core.types import (
    ENGRAM_VERSION,
    ENG_FILE_EXTENSION,
    CompressionMethod,
    EngramMetadata,
)


class SerializationError(Exception):
    """Raised when serialization or deserialization fails."""


class EngramSerializer:
    """Serializes/deserializes KV cache tensors to/from .eng files.

    Canonical shape for KV tensors in ENGRAM:
        keys:   [n_layers, n_kv_heads, ctx_len, head_dim]
        values: [n_layers, n_kv_heads, ctx_len, head_dim]
    """

    def serialize(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        agent_id: str,
        task_description: str,
        model_id: str,
        output_path: Path,
        compression: CompressionMethod = CompressionMethod.Q8_0,
        cache_id: str | None = None,
        parent_cache_id: str | None = None,
        input_tokens: list[int] | None = None,
        extra_metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Serialize KV cache tensors to a .eng file.

        Args:
            keys: [n_layers, n_kv_heads, ctx_len, head_dim]
            values: [n_layers, n_kv_heads, ctx_len, head_dim]
            agent_id: Identifier for the agent that produced this state
            task_description: Human-readable description (used for EGR search)
            model_id: Full model identifier
            output_path: Path to write .eng file
            compression: Compression method to apply
            cache_id: Explicit cache ID (auto-generated if None)
            parent_cache_id: ID of parent for delta chains
            input_tokens: Token IDs that generated this state (for hash)
            extra_metadata: Additional string key-value pairs

        Returns:
            Dict with cache_id, size_bytes, compression_ratio, path
        """
        if keys.shape != values.shape:
            raise SerializationError(
                f"Keys/values shape mismatch: {keys.shape} vs {values.shape}"
            )
        if keys.dim() != 4:
            raise SerializationError(
                f"Expected 4D [n_layers, n_kv_heads, ctx_len, head_dim], "
                f"got {keys.dim()}D: {keys.shape}"
            )

        n_layers, n_kv_heads, ctx_len, head_dim = keys.shape

        tensors: dict[str, torch.Tensor] = {}

        if compression == CompressionMethod.INT8:
            from kvcos.core.compression import compress_int8_tensor

            k_pair = compress_int8_tensor(keys)
            v_pair = compress_int8_tensor(values)
            for i in range(n_layers):
                tensors[f"layer_{i}_keys"] = k_pair.quantized[i].contiguous()
                tensors[f"layer_{i}_keys_scale"] = k_pair.scales[i].contiguous()
                tensors[f"layer_{i}_values"] = v_pair.quantized[i].contiguous()
                tensors[f"layer_{i}_values_scale"] = v_pair.scales[i].contiguous()
            # Reuse k_compressed for metadata only — actual INT8 data is
            # already written per-layer above via k_pair/v_pair.
            k_compressed = compress(keys, compression)
            v_compressed = k_compressed
        elif compression == CompressionMethod.LAYER_DELTA:
            from kvcos.core.compression import compress_layer_delta

            k_ld = compress_layer_delta(keys)
            v_ld = compress_layer_delta(values)
            # Layer 0: fp16 baseline
            tensors["layer_0_keys"] = k_ld.baseline.contiguous()
            tensors["layer_0_values"] = v_ld.baseline.contiguous()
            # Layers 1..N: int8 deltas + fp16 scales
            for i in range(n_layers - 1):
                tensors[f"layer_{i+1}_keys"] = k_ld.delta_quantized[i].contiguous()
                tensors[f"layer_{i+1}_keys_scale"] = k_ld.delta_scales[i].contiguous()
                tensors[f"layer_{i+1}_values"] = v_ld.delta_quantized[i].contiguous()
                tensors[f"layer_{i+1}_values_scale"] = v_ld.delta_scales[i].contiguous()
            # Reuse k_compressed for metadata only — actual layer-delta data
            # is already written above via k_ld/v_ld.
            k_compressed = compress(keys, compression)
            v_compressed = k_compressed
        else:
            k_compressed = compress(keys, compression)
            v_compressed = compress(values, compression)
            for i in range(n_layers):
                tensors[f"layer_{i}_keys"] = k_compressed.data[i].contiguous()
                tensors[f"layer_{i}_values"] = v_compressed.data[i].contiguous()

        cid = cache_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        token_hash = ""
        if input_tokens:
            token_bytes = b"".join(t.to_bytes(4, "little") for t in input_tokens)
            token_hash = f"sha256:{hashlib.sha256(token_bytes).hexdigest()}"

        metadata: EngramMetadata = {
            "engram_version": ENGRAM_VERSION,
            "cache_id": cid,
            "compression": compression.value,
            "model_id": model_id,
            "model_family": infer_model_family(model_id),
            "n_layers": str(n_layers),
            "n_heads": str(n_kv_heads),
            "n_kv_heads": str(n_kv_heads),
            "head_dim": str(head_dim),
            "context_len": str(ctx_len),
            "agent_id": agent_id,
            "task_description": task_description,
            "created_at": now,
        }

        if parent_cache_id:
            metadata["parent_cache_id"] = parent_cache_id
        if token_hash:
            metadata["token_hash"] = token_hash
        for key, val in k_compressed.metadata.items():
            metadata[f"compression_{key}"] = val  # type: ignore[literal-required]
        if extra_metadata:
            for key, val in extra_metadata.items():
                metadata[key] = val  # type: ignore[literal-required]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        str_metadata: dict[str, str] = {k: str(v) for k, v in metadata.items()}
        save_file(tensors, str(output_path), metadata=str_metadata)

        original_bytes = (keys.numel() + values.numel()) * keys.element_size()
        compressed_bytes = output_path.stat().st_size

        return {
            "cache_id": cid,
            "size_bytes": compressed_bytes,
            "compression_ratio": original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0,
            "path": str(output_path),
            "n_layers": n_layers,
            "context_len": ctx_len,
        }

    def deserialize(
        self,
        path: Path,
        target_compression: CompressionMethod | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, EngramMetadata]:
        """Deserialize a .eng file into KV cache tensors.

        Returns (keys, values, metadata) where tensors are
        [n_layers, n_kv_heads, ctx_len, head_dim].
        """
        if not path.exists():
            raise SerializationError(f"Engram file not found: {path}")

        tensors = load_file(str(path))
        metadata = self._read_metadata(path)

        n_layers = int(metadata.get("n_layers", "0"))
        if n_layers == 0:
            n_layers = (
                max(int(k.split("_")[1]) for k in tensors if k.startswith("layer_")) + 1
            )

        stored_compression = metadata.get("compression", "fp16")
        is_int8 = stored_compression == CompressionMethod.INT8.value
        is_layer_delta = stored_compression == CompressionMethod.LAYER_DELTA.value

        k_layers: list[torch.Tensor] = []
        v_layers: list[torch.Tensor] = []

        if is_layer_delta:
            from kvcos.core.compression import decompress_int8_tensor

            # Layer 0: fp16 baseline
            k_layers.append(tensors["layer_0_keys"].float())
            v_layers.append(tensors["layer_0_values"].float())
            # Layers 1..N: accumulate int8 deltas
            for i in range(1, n_layers):
                k_delta = decompress_int8_tensor(
                    tensors[f"layer_{i}_keys"], tensors[f"layer_{i}_keys_scale"]
                )
                v_delta = decompress_int8_tensor(
                    tensors[f"layer_{i}_values"], tensors[f"layer_{i}_values_scale"]
                )
                k_layers.append(k_layers[-1] + k_delta.float())
                v_layers.append(v_layers[-1] + v_delta.float())
            keys = torch.stack([l.to(torch.float16) for l in k_layers], dim=0)
            values = torch.stack([l.to(torch.float16) for l in v_layers], dim=0)
        else:
            for i in range(n_layers):
                k_key = f"layer_{i}_keys"
                v_key = f"layer_{i}_values"
                if k_key not in tensors or v_key not in tensors:
                    raise SerializationError(f"Missing tensor for layer {i}")

                if is_int8:
                    from kvcos.core.compression import decompress_int8_tensor

                    k_scale_key = f"layer_{i}_keys_scale"
                    v_scale_key = f"layer_{i}_values_scale"
                    if k_scale_key not in tensors or v_scale_key not in tensors:
                        raise SerializationError(f"Missing INT8 scale for layer {i}")
                    k_layers.append(decompress_int8_tensor(tensors[k_key], tensors[k_scale_key]))
                    v_layers.append(decompress_int8_tensor(tensors[v_key], tensors[v_scale_key]))
                else:
                    k_layers.append(tensors[k_key])
                    v_layers.append(tensors[v_key])

            keys = torch.stack(k_layers, dim=0)
            values = torch.stack(v_layers, dim=0)

        if target_compression is not None:
            stored = CompressionMethod(metadata.get("compression", "fp16"))
            keys = decompress(keys, stored)
            values = decompress(values, stored)

        return keys, values, metadata  # type: ignore[return-value]

    def _read_metadata(self, path: Path) -> dict[str, str]:
        """Read only the metadata header (no tensor data loaded)."""
        from safetensors import safe_open

        metadata: dict[str, str] = {}
        with safe_open(str(path), framework="pt") as f:
            raw_meta = f.metadata()
            if raw_meta:
                metadata = dict(raw_meta)
        return metadata

    def read_metadata_only(self, path: Path) -> EngramMetadata:
        """Read just the metadata from a .eng file. Efficient for indexing."""
        raw = self._read_metadata(path)
        return raw  # type: ignore[return-value]
