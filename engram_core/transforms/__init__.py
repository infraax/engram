"""Fingerprint transform stack (docs/v2/PLAN.md §2).

S0 capture  → capture.py   (HF transformers: post-RoPE cache keys + pre-RoPE hook keys)
S1 derope   → rope.py      (exact inverse rotation of cached keys)
S2..S8      → ops.py       (sinks, pooling, layer axis, center, gate, project, normalize)
pipeline.py composes stages from a config dict and yields a SpaceDescriptor.
"""

from .capture import HFCapturer, KVCapture  # noqa: F401
from .pipeline import FrozenMean, Pipeline  # noqa: F401
