"""
ENGRAM Protocol — KV Cache Compression Layer


Implements:
  - FP16 passthrough (no compression)
  - Q8_0: group quantization matching llama.cpp GGML_TYPE_Q8_0
    Phase 1 production fallback. ~2x compression, <5% speed hit (D5).
  - PolarQuant: MSE-optimal random rotation + Lloyd-Max codebook at 3 bits.
    QJL REMOVED — confirmed harmful by 6+ independent implementations (D5).
    Softmax amplifies QJL variance, making two-stage worse than MSE-only.

Reference: TheTom/turboquant_plus (511+ tests, most mature impl)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from kvcos.core.types import CompressionMethod

# ── Q8_0 Constants ────────────────────────────────────────────────────────────
Q8_GROUP_SIZE = 32


@dataclass(frozen=True)
class CompressionResult:
    """Result of compressing a KV cache tensor."""

    data: torch.Tensor
    method: CompressionMethod
    original_dtype: torch.dtype
    compression_ratio: float
    metadata: dict[str, str]


# ── FP16 Passthrough ──────────────────────────────────────────────────────────


def compress_fp16(kv: torch.Tensor) -> CompressionResult:
    """No-op compression: ensure tensor is FP16."""
    data = kv.to(torch.float16).contiguous()
    return CompressionResult(
        data=data,
        method=CompressionMethod.FP16,
        original_dtype=kv.dtype,
        compression_ratio=1.0,
        metadata={},
    )


def decompress_fp16(data: torch.Tensor) -> torch.Tensor:
    return data.to(torch.float16)


# ── Q8_0 Quantization ────────────────────────────────────────────────────────
# Matches llama.cpp GGML_TYPE_Q8_0 layout:
#   32-element groups, 1 float16 scale per group, 32 int8 values
#   Storage: (32*1 + 2) / (32*2) = 34/64 ≈ 1.88x compression


def compress_q8_0(kv: torch.Tensor) -> CompressionResult:
    """Quantize KV cache to Q8_0 (int8 with per-group scale).

    Stores dequantized bfloat16 for safetensors compatibility —
    safetensors doesn't support int8+scale pairs natively.
    """
    original_dtype = kv.dtype
    original_bytes = kv.numel() * kv.element_size()

    kv_flat = kv.float().contiguous()
    orig_shape = kv_flat.shape

    last_dim = orig_shape[-1]
    pad_amount = (Q8_GROUP_SIZE - last_dim % Q8_GROUP_SIZE) % Q8_GROUP_SIZE
    if pad_amount > 0:
        kv_flat = torch.nn.functional.pad(kv_flat, (0, pad_amount))

    new_shape = kv_flat.shape[:-1] + (-1, Q8_GROUP_SIZE)
    grouped = kv_flat.reshape(new_shape)

    scales = grouped.abs().amax(dim=-1, keepdim=True) / 127.0
    scales = scales.clamp(min=1e-10)

    quantized = torch.clamp(torch.round(grouped / scales), -127, 127)
    dequantized = (quantized * scales).reshape(kv_flat.shape)

    if pad_amount > 0:
        dequantized = dequantized[..., :last_dim]

    dequantized = dequantized.reshape(orig_shape).to(torch.bfloat16)
    compressed_bytes = dequantized.numel() * 2

    return CompressionResult(
        data=dequantized,
        method=CompressionMethod.Q8_0,
        original_dtype=original_dtype,
        compression_ratio=original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0,
        metadata={"q8_group_size": str(Q8_GROUP_SIZE)},
    )


def decompress_q8_0(data: torch.Tensor) -> torch.Tensor:
    return data.to(torch.float16)


# ── PolarQuant (Phase 2 — TurboQuant without QJL) ────────────────────────────
# QJL is INTENTIONALLY ABSENT per D5.


class PolarQuantConfig:
    """Configuration for PolarQuant compression."""

    def __init__(self, bits: int = 3, seed: int = 42):
        self.bits = bits
        self.n_centroids = 2**bits
        self.seed = seed
        self._rotation_cache: dict[int, torch.Tensor] = {}
        self._codebook_cache: dict[int, torch.Tensor] = {}

    def get_rotation_matrix(self, dim: int, device: torch.device) -> torch.Tensor:
        """Get fixed random orthogonal rotation matrix R ∈ R^(d×d)."""
        if dim not in self._rotation_cache:
            rng = np.random.RandomState(self.seed)
            gaussian = rng.randn(dim, dim).astype(np.float32)
            q, r = np.linalg.qr(gaussian)
            d = np.diag(r)
            ph = np.sign(d)
            q *= ph[np.newaxis, :]
            self._rotation_cache[dim] = torch.from_numpy(q)
        return self._rotation_cache[dim].to(device)

    def get_lloyd_max_codebook(self, dim: int) -> torch.Tensor:
        """Lloyd-Max optimal centroids for N(0,1), 3-bit (8 levels)."""
        if dim not in self._codebook_cache:
            codebook = torch.tensor(
                [-1.748, -1.050, -0.501, -0.000, 0.000, 0.501, 1.050, 1.748],
                dtype=torch.float32,
            )
            self._codebook_cache[dim] = codebook
        return self._codebook_cache[dim]


_POLAR_CONFIG = PolarQuantConfig()


def compress_polarquant(kv: torch.Tensor) -> CompressionResult:
    """Compress using PolarQuant (3-bit Lloyd-Max after random rotation).

    Phase 2 implementation. Currently stores dequantized bfloat16.
    True 3-bit packed storage is Phase 2+.
    """
    original_dtype = kv.dtype
    original_bytes = kv.numel() * kv.element_size()
    device = kv.device

    kv_float = kv.float().contiguous()
    orig_shape = kv_float.shape

    head_dim = orig_shape[-1]
    flat = kv_float.reshape(-1, head_dim)

    R = _POLAR_CONFIG.get_rotation_matrix(head_dim, device)
    rotated = flat @ R

    dim_std = rotated.std(dim=0, keepdim=True).clamp(min=1e-10)
    normalized = rotated / dim_std

    codebook = _POLAR_CONFIG.get_lloyd_max_codebook(head_dim).to(device)
    distances = (normalized.unsqueeze(-1) - codebook.unsqueeze(0).unsqueeze(0)) ** 2
    indices = distances.argmin(dim=-1)

    dequantized = codebook[indices]
    dequantized = dequantized * dim_std
    R_inv = R.T
    dequantized = dequantized @ R_inv

    dequantized = dequantized.reshape(orig_shape).to(torch.bfloat16)
    compressed_bytes = dequantized.numel() * 2

    return CompressionResult(
        data=dequantized,
        method=CompressionMethod.POLARQUANT,
        original_dtype=original_dtype,
        compression_ratio=original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0,
        metadata={
            "polarquant_bits": "3",
            "polarquant_seed": str(_POLAR_CONFIG.seed),
            "qjl_enabled": "false",  # D5: QJL permanently disabled
        },
    )


def decompress_polarquant(data: torch.Tensor) -> torch.Tensor:
    return data.to(torch.float16)


# ── INT8 Quantization (Phase 2 — true on-disk compression) ───────────────────
# Stores actual int8 tensors in safetensors (1 byte/element vs 2 for fp16).
# Per-row symmetric quantization: scale = max(abs(row)) / 127.
# Separate scale tensor stored alongside quantized data.
# 2x on-disk compression with cos_sim > 0.999.


@dataclass(frozen=True)
class Int8CompressedPair:
    """INT8 quantized tensor + per-row scales."""

    quantized: torch.Tensor  # int8 [same shape as input]
    scales: torch.Tensor  # float16 [shape[:-1]] — one scale per row


def compress_int8_tensor(kv: torch.Tensor) -> Int8CompressedPair:
    """Quantize a KV tensor to int8 with per-row scales.

    Args:
        kv: [..., head_dim] tensor (any dtype)

    Returns:
        Int8CompressedPair with int8 data and float16 scales
    """
    orig_shape = kv.shape
    flat = kv.float().reshape(-1, orig_shape[-1])

    row_max = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scales = row_max / 127.0

    quantized = (flat / scales).round().clamp(-127, 127).to(torch.int8)
    scales_f16 = scales.squeeze(1).to(torch.float16)

    return Int8CompressedPair(
        quantized=quantized.reshape(orig_shape),
        scales=scales_f16.reshape(orig_shape[:-1]),
    )


def decompress_int8_tensor(quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize int8 tensor using per-row scales.

    Returns float16 tensor of the original shape.
    """
    return (quantized.float() * scales.float().unsqueeze(-1)).to(torch.float16)


def compress_int8(kv: torch.Tensor) -> CompressionResult:
    """INT8 compression — returns dequantized float16 for CompressionResult compat.

    The actual int8 storage is handled by the serializer which calls
    compress_int8_tensor() directly for true on-disk compression.
    This wrapper exists for the dispatcher API.
    """
    pair = compress_int8_tensor(kv)
    dequantized = decompress_int8_tensor(pair.quantized, pair.scales)

    original_bytes = kv.numel() * kv.element_size()
    # True on-disk: int8 data + float16 scales
    compressed_bytes = pair.quantized.numel() * 1 + pair.scales.numel() * 2

    return CompressionResult(
        data=dequantized,
        method=CompressionMethod.INT8,
        original_dtype=kv.dtype,
        compression_ratio=original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0,
        metadata={"int8_scale_dtype": "float16"},
    )


# ── LAYER_DELTA Compression ──────────────────────────────────────────────────
# Stores layer 0 as fp16 baseline, layers 1..N as int8 deltas from previous.
# Inter-layer residuals are typically small (adjacent layers are correlated),
# so int8 quantization of deltas achieves better fidelity than direct int8.
# On-disk: ~(1/N) fp16 + ((N-1)/N) int8 ≈ slightly better than straight INT8.


@dataclass(frozen=True)
class LayerDeltaCompressed:
    """Layer-delta compressed: fp16 baseline + int8 deltas."""

    baseline: torch.Tensor  # [n_kv_heads, n_cells, head_dim] fp16
    delta_quantized: list[torch.Tensor]  # each int8 [n_kv_heads, n_cells, head_dim]
    delta_scales: list[torch.Tensor]  # each fp16 [n_kv_heads, n_cells]
    n_layers: int


def compress_layer_delta(kv: torch.Tensor) -> LayerDeltaCompressed:
    """Compress KV tensor using inter-layer delta encoding.

    Args:
        kv: [n_layers, n_kv_heads, n_cells, head_dim]

    Returns:
        LayerDeltaCompressed with fp16 baseline + int8 deltas
    """
    n_layers = kv.shape[0]
    baseline = kv[0].to(torch.float16)

    deltas: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []

    for i in range(1, n_layers):
        delta = (kv[i].float() - kv[i - 1].float())
        flat = delta.reshape(-1, delta.shape[-1])
        row_max = flat.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        q = (flat / row_max.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
        deltas.append(q.reshape(delta.shape))
        scales.append(row_max.to(torch.float16).reshape(delta.shape[:-1]))

    return LayerDeltaCompressed(
        baseline=baseline, delta_quantized=deltas,
        delta_scales=scales, n_layers=n_layers,
    )


def decompress_layer_delta(data: LayerDeltaCompressed) -> torch.Tensor:
    """Decompress layer-delta encoded KV tensor."""
    layers = [data.baseline.float()]
    for dq, ds in zip(data.delta_quantized, data.delta_scales):
        flat = dq.float().reshape(-1, dq.shape[-1])
        delta = (flat * ds.float().reshape(-1).unsqueeze(1)).reshape(dq.shape)
        layers.append(layers[-1] + delta)
    return torch.stack(layers).to(torch.float16)


def compress_layer_delta_result(kv: torch.Tensor) -> CompressionResult:
    """Layer-delta wrapper for CompressionResult API."""
    compressed = compress_layer_delta(kv)
    decompressed = decompress_layer_delta(compressed)

    original_bytes = kv.numel() * kv.element_size()
    # On-disk: baseline fp16 + (N-1) int8 deltas + (N-1) fp16 scales
    n = compressed.n_layers
    per_layer_elements = kv[0].numel()
    scale_elements = kv.shape[1] * kv.shape[2]  # n_kv_heads * n_cells
    compressed_bytes = (
        per_layer_elements * 2  # baseline fp16
        + (n - 1) * per_layer_elements * 1  # int8 deltas
        + (n - 1) * scale_elements * 2  # fp16 scales
    )

    return CompressionResult(
        data=decompressed,
        method=CompressionMethod.LAYER_DELTA,
        original_dtype=kv.dtype,
        compression_ratio=original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0,
        metadata={"delta_n_layers": str(n)},
    )


# ── Dispatcher ────────────────────────────────────────────────────────────────


def compress(kv: torch.Tensor, method: CompressionMethod) -> CompressionResult:
    """Compress a KV cache tensor using the specified method."""
    match method:
        case CompressionMethod.FP16:
            return compress_fp16(kv)
        case CompressionMethod.Q8_0:
            return compress_q8_0(kv)
        case CompressionMethod.POLARQUANT:
            return compress_polarquant(kv)
        case CompressionMethod.INT8:
            return compress_int8(kv)
        case CompressionMethod.LAYER_DELTA:
            return compress_layer_delta_result(kv)
        case CompressionMethod.Q4_0:
            import warnings

            warnings.warn(
                "Q4_0 has 92% dequantization slowdown at 64K+ context. "
                "Using Q8_0 instead. See D5.",
                UserWarning,
                stacklevel=2,
            )
            return compress_q8_0(kv)
        case _:
            raise ValueError(f"Unknown compression method: {method}")


def decompress(data: torch.Tensor, method: CompressionMethod) -> torch.Tensor:
    """Decompress a KV cache tensor."""
    match method:
        case CompressionMethod.FP16:
            return decompress_fp16(data)
        case CompressionMethod.Q8_0 | CompressionMethod.Q4_0:
            return decompress_q8_0(data)
        case CompressionMethod.POLARQUANT:
            return decompress_polarquant(data)
        case CompressionMethod.INT8 | CompressionMethod.LAYER_DELTA:
            # Already dequantized float16 in CompressionResult
            return data.to(torch.float16)
        case _:
            raise ValueError(f"Unknown compression method: {method}")
