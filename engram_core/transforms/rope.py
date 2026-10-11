"""S1 — exact inverse of rotary position embedding (RoPE) on cached keys.

HF convention:   k_rot = k * cos + rotate_half(k) * sin,  rotate_half(x) = cat(-x2, x1)
This is a 2-D rotation R(θ) per frequency pair, scaled by s = sqrt(cos²+sin²)
(s == attention_scaling; 1.0 for default/llama3 rope, != 1 for e.g. YaRN).
Inverse:         k = (k_rot * cos - rotate_half(k_rot) * sin) / (cos² + sin²)
"""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """k: [..., T, D]; cos/sin: [T, D]."""
    return k * cos + rotate_half(k) * sin


def inverse_rope(k_rot: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Exact inverse of apply_rope. k_rot: [..., T, D]; cos/sin: [T, D]."""
    denom = (cos * cos + sin * sin).clamp_min(1e-12)
    return (k_rot * cos - rotate_half(k_rot) * sin) / denom


def rerotate(k_pre: torch.Tensor, cos_new: torch.Tensor, sin_new: torch.Tensor) -> torch.Tensor:
    """Place position-free keys at new positions (restore at a different offset)."""
    return apply_rope(k_pre, cos_new, sin_new)
