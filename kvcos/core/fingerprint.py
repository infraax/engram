"""
ENGRAM Protocol — Standalone State Extraction Functions

Contains the Engram Absolute fingerprint: compute_fourier_fingerprint().
This is the primary cross-model retrieval fingerprint, validated at
98% recall@1 at N=1000 with power-law decay N^-0.207.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from kvcos.core.blob_parser import ParsedMultiSectionCache


def compute_fourier_fingerprint(
    layer_keys: torch.Tensor,
    freqs: list[int] | None = None,
) -> torch.Tensor:
    """Compute the Engram Absolute fingerprint (f0+f1) from per-layer mean keys.

    Takes the real DFT over the layer dimension, extracts amplitude at
    the specified frequencies, normalizes each, and concatenates.

    Args:
        layer_keys: [n_layers, dim] where dim = n_kv_heads * head_dim.
                    Must be per-layer MEAN across token positions.
                    If shape is [n_layers, n_kv, hd], reshapes automatically.
        freqs:      DFT frequency indices to concatenate.
                    Default [0, 1] = DC (f0) + first harmonic (f1).

    Returns:
        Fingerprint tensor [dim * len(freqs)], L2-normalized, float32.

    Properties:
        - Cross-model invariant within Llama-3.x family (cos ~0.89)
        - Zero corpus dependency: no centroid, no basis, no training data
        - Recall@1 N=200: 98%  N=1000: 98%  decay: N^-0.207
    """
    if freqs is None:
        freqs = [0, 1]

    if layer_keys.dim() == 3:
        n_layers = layer_keys.shape[0]
        layer_keys = layer_keys.reshape(n_layers, -1)

    layer_keys = layer_keys.float()

    F_complex = torch.fft.rfft(layer_keys, dim=0)
    F_amp = F_complex.abs()

    components = []
    for f in freqs:
        if f >= F_amp.shape[0]:
            raise ValueError(
                f"Requested freq={f} but rfft produced only "
                f"{F_amp.shape[0]} components for {layer_keys.shape[0]} layers."
            )
        components.append(F.normalize(F_amp[f], dim=-1))

    return torch.cat(components, dim=-1)


def compute_eigenform_score(
    layer_keys: torch.Tensor,
    noise_sigma: float = 0.001,
    n_trials: int = 3,
    freqs: list | None = None,
) -> float:
    """Compute eigenform stability score via noise perturbation.

    Measures how stable the Fourier fingerprint is under small noise.
    Score near 1.0 = stable. Below 0.95 = fragile fingerprint.

    Args:
        layer_keys: [n_layers, dim] per-layer mean key vectors.
        noise_sigma: Gaussian noise standard deviation.
        n_trials: Number of perturbed copies to compare.
        freqs: DFT frequencies. Default [0, 1].

    Returns:
        float in [0, 1]. Mean pairwise cosine across noise trials.
    """
    if freqs is None:
        freqs = [0, 1]
    fps = []
    for t in range(n_trials):
        noisy = layer_keys if t == 0 else layer_keys + torch.randn_like(layer_keys) * noise_sigma
        fps.append(compute_fourier_fingerprint(noisy.float(), freqs=freqs))
    pairs = [(i, j) for i in range(n_trials) for j in range(i+1, n_trials)]
    if not pairs:
        return 1.0
    return float(sum(F.cosine_similarity(fps[a].unsqueeze(0), fps[b].unsqueeze(0)).item() for a, b in pairs) / len(pairs))


def compute_iswa_fingerprint(
    parsed: "ParsedMultiSectionCache",
    freqs: list | None = None,
    normalize_layers: bool = True,
) -> torch.Tensor:
    """Compute concatenated Fourier fingerprint for ISWA multi-section caches.

    Strategy A (per-section concatenation):
      For each cache section, compute mean over tokens, then Fourier FP.
      Concatenate section FPs into one vector.

    For Gemma 4 with freqs=[0, 1]:
      Global (5 layers, 2 heads, 512 dim) → 1024 * 2 = 2048
      SWA   (25 layers, 8 heads, 256 dim) → 2048 * 2 = 4096
      Total: 6144-dim fingerprint

    Each section's sub-fingerprint is independently L2-normalized,
    preserving the relative geometry within each attention type.

    Args:
        parsed: ParsedMultiSectionCache from parse_multi_section_blob()
        freqs: DFT frequency indices. Default [0, 1].
        normalize_layers: L2-normalize each layer before DFT (v2 behavior).

    Returns:
        Concatenated fingerprint tensor, float32.
    """
    if freqs is None:
        freqs = [0, 1]

    section_fps: list[torch.Tensor] = []
    for section in parsed.sections:
        # Mean over tokens: [n_layers, n_kv_heads, n_cells, head_dim] → [n_layers, n_kv_heads * head_dim]
        layer_keys = section.keys.float().mean(dim=2)
        fp = compute_fourier_fingerprint_v2(layer_keys, freqs=freqs, normalize_layers=normalize_layers)
        section_fps.append(fp)

    return torch.cat(section_fps, dim=-1)


def compute_fourier_fingerprint_v2(
    layer_keys: torch.Tensor,
    freqs: list | None = None,
    normalize_layers: bool = True,
) -> torch.Tensor:
    """Fourier fingerprint v2: L2-normalize each layer before DFT.

    Removes absolute magnitude scale (which differs by KV head count
    across model families), preserves layer-progression shape.

    Within-family: same recall as v1 (98%).
    Cross-family: f0+f1 cross-sim expected >>0.26 (v1 baseline).
    """
    if freqs is None:
        freqs = [0, 1]
    if layer_keys.dim() == 3:
        layer_keys = layer_keys.reshape(layer_keys.shape[0], -1)
    layer_keys = layer_keys.float()
    if normalize_layers:
        layer_keys = F.normalize(layer_keys, dim=-1)
    F_complex = torch.fft.rfft(layer_keys, dim=0)
    F_amp = F_complex.abs()
    components = []
    for f in freqs:
        if f >= F_amp.shape[0]:
            raise ValueError(f"freq={f} out of range for {layer_keys.shape[0]} layers")
        components.append(F.normalize(F_amp[f], dim=-1))
    return torch.cat(components, dim=-1)
