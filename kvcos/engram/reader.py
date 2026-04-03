"""
EIGENGRAM reader: .eng file -> IndexEntry + fingerprint vectors
"""

from __future__ import annotations

from pathlib import Path

from .format import EigramDecoder
from kvcos.core.manifold_index import IndexEntry

_decoder = EigramDecoder()


def read_eigengram(path: str) -> dict:
    """Read a .eng file and return decoded fields."""
    if not Path(path).exists():
        raise FileNotFoundError(f"EIGENGRAM not found: {path}")
    data = Path(path).read_bytes()
    return _decoder.decode(data)


def load_eigengram_index(
    paths: list[str],
    fingerprint: str = "perdoc",
) -> tuple[list, list]:
    """Load multiple .eng files for ManifoldIndex.

    fingerprint: 'perdoc' (same-model) | 'fcdb' (cross-model)

    Returns (vecs, entries) ready for ManifoldIndex.add().
    """
    if fingerprint not in ("perdoc", "fcdb", "fourier"):
        raise ValueError(f"fingerprint must be 'perdoc', 'fcdb', or 'fourier', got '{fingerprint}'")

    vecs = []
    entries = []
    key = f"vec_{fingerprint}"

    for path in paths:
        rec = read_eigengram(path)
        vecs.append(rec[key])
        entries.append(
            IndexEntry(
                cache_id=rec["cache_id"],
                task_description=rec["task_description"],
                model_id=rec["model_id"],
                created_at=rec["created_at"],
                context_len=rec["context_len"],
                l2_norm=rec["l2_norm"],
            )
        )

    return vecs, entries
