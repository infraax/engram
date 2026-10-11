"""SpaceDescriptor and space_key.

A *space* is the exact recipe that produced a fingerprint: sense, model, every
transform stage with its parameters, and any frozen statistics (centering mean).
Two fingerprints may only be compared if their space_key is identical
(non-negotiable #3 in docs/v2/PLAN.md).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

try:  # BLAKE3 per the EGR2 spec; blake2b fallback keeps the core dependency-free
    import blake3 as _blake3

    def _digest(data: bytes) -> bytes:
        return _blake3.blake3(data).digest()

    HASH_NAME = "blake3"
except ImportError:  # pragma: no cover
    def _digest(data: bytes) -> bytes:
        return hashlib.blake2b(data, digest_size=32).digest()

    HASH_NAME = "blake2b"


def canonical(obj: Any) -> bytes:
    """Deterministic JSON encoding (sorted keys, no whitespace, UTF-8)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest_hex(obj: Any, domain: str, n_bytes: int = 32) -> str:
    return _digest(domain.encode() + b"\x00" + canonical(obj))[:n_bytes].hex()


SENSES = ("reflex_kv", "meaning_text")


@dataclass(frozen=True)
class SpaceDescriptor:
    sense: str                       # "reflex_kv" | "meaning_text"
    model_id: str                    # HF repo id (+ revision if pinned) or embedder id
    pipeline: dict = field(default_factory=dict)   # full transform config
    dims: int = 0
    center_hash: str = ""            # hash of the frozen mean vector, "" if no centering
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sense not in SENSES:
            raise ValueError(f"unknown sense {self.sense!r}; expected one of {SENSES}")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def key(self) -> str:
        """8-byte space key (16 hex chars), domain-separated."""
        return digest_hex(self.to_dict(), "engram.space.v2", n_bytes=8)


def assert_same_space(a: SpaceDescriptor, b: SpaceDescriptor) -> None:
    if a.key != b.key:
        raise TypeError(f"cross-space comparison: {a.key} vs {b.key}")
