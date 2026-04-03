"""
EIGENGRAM binary format codec (EGR1 v1.0).

An EIGENGRAM (.eng) file is a self-contained semantic certificate
for a KV-cache document. It encodes two fingerprint vectors, the
shared coordinate system they live in, and enough metadata to
reproduce the query fold-in without access to the original text
or model.

Design goals:
  - Portable: pure binary, no JSON, no pickle, no protobuf.
  - Versioned: magic bytes + version field.
  - Self-contained: joint_center embedded for query fold-in.
  - Compact: float16 vectors, ~800 bytes total per document.

Dual-fingerprint architecture:
  vec_perdoc  - per-document SVD projection (same-model, margin ~0.37)
  vec_fcdb    - FCDB projection (cross-model, margin ~0.013)

Binary layout (little-endian, 99-byte fixed header):
  Offset  Size  Type     Field
     0     4    bytes    magic = "EGR1"
     4     1    uint8    version (currently 1)
     5    32    ascii    corpus_hash
    37    20    ascii    created_at
    57    16    ascii    model_id (null-padded)
    73     2    uint16   basis_rank R
    75     2    uint16   n_corpus
    77     2    int8x2   layer_range
    79     4    uint32   context_len
    83     4    float32  l2_norm
    87     4    float32  scs
    91     4    float32  margin_proof
    95     2    uint16   task_desc_len
    97     2    uint16   cache_id_len
  Variable:
    99     R*2  float16  vec_perdoc
   +R*2   R*2  float16  vec_fcdb
  +2R*2   256  float16  joint_center (128 x float16)
   +256   var  utf-8    task_description
   +var   var  utf-8    cache_id

Total for R=116: ~800 bytes.

Compatibility: readers MUST reject magic != "EGR1" or version mismatch.
"""

from __future__ import annotations

import struct

import numpy as np
import torch

EIGENGRAM_MAGIC = b"EGR1"
EIGENGRAM_VERSION = 1


class EigramEncoder:
    """Encode and decode EIGENGRAM binary certificates.

    A single instance handles both directions. EigramDecoder is an alias.
    Float16 storage preserves cosine similarity to > 0.999.
    """

    def encode(
        self,
        vec_perdoc: torch.Tensor,
        vec_fcdb: torch.Tensor,
        joint_center: torch.Tensor,
        corpus_hash: str,
        model_id: str,
        basis_rank: int,
        n_corpus: int,
        layer_range: tuple[int, int],
        context_len: int,
        l2_norm: float,
        scs: float,
        margin_proof: float,
        task_description: str,
        cache_id: str,
        vec_fourier: torch.Tensor | None = None,
        local_density: int = 0,
        eigenform_score: float = 1.0,
        confusion_flag: bool = False,
        vec_fourier_v2: torch.Tensor | None = None,
    ) -> bytes:
        """Serialise all fields into an EIGENGRAM binary blob."""
        from datetime import datetime, timezone

        td_b = task_description.encode("utf-8")[:256]
        ci_b = cache_id.encode("utf-8")[:64]
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

        buf = bytearray()
        buf += EIGENGRAM_MAGIC
        buf += struct.pack("<B", EIGENGRAM_VERSION)
        buf += corpus_hash.encode("ascii")[:32].ljust(32, b"\x00")
        buf += now.encode("ascii")[:20].ljust(20, b"\x00")
        buf += model_id.encode("ascii")[:16].ljust(16, b"\x00")
        buf += struct.pack("<H", basis_rank)
        buf += struct.pack("<H", n_corpus)
        buf += struct.pack("<bb", layer_range[0], layer_range[1])
        buf += struct.pack("<I", context_len)
        buf += struct.pack("<f", l2_norm)
        buf += struct.pack("<f", scs)
        buf += struct.pack("<f", margin_proof)
        buf += struct.pack("<H", len(td_b))
        buf += struct.pack("<H", len(ci_b))
        # fourier_dim: 0 if no vec_fourier, else len(vec_fourier)
        fourier_dim = len(vec_fourier) if vec_fourier is not None else 0
        buf += struct.pack("<H", fourier_dim)
        buf += struct.pack("<H", local_density)
        buf += struct.pack("<f", eigenform_score)

        buf += vec_perdoc.to(torch.float16).numpy().tobytes()
        buf += vec_fcdb.to(torch.float16).numpy().tobytes()
        buf += joint_center[:128].to(torch.float16).numpy().tobytes()

        buf += td_b
        buf += ci_b

        # Append vec_fourier if present (backward-compatible extension)
        if vec_fourier is not None:
            buf += vec_fourier.to(torch.float16).numpy().tobytes()

        # v1.2 extension: confusion_flag + vec_fourier_v2
        # Written only when at least one is non-default, preserving
        # backward compat with readers that stop after vec_fourier.
        if confusion_flag or vec_fourier_v2 is not None:
            buf += struct.pack("<B", 1 if confusion_flag else 0)
            v2_dim = len(vec_fourier_v2) if vec_fourier_v2 is not None else 0
            buf += struct.pack("<H", v2_dim)
            if vec_fourier_v2 is not None:
                buf += vec_fourier_v2.to(torch.float16).numpy().tobytes()

        return bytes(buf)

    def decode(self, data: bytes) -> dict:
        """Deserialise an EIGENGRAM binary blob into a dict.

        Returns dict with all fields. Vectors upcast to float32.
        Raises ValueError on magic/version mismatch.
        """
        if len(data) < 4 or data[:4] != EIGENGRAM_MAGIC:
            raise ValueError(
                f"Invalid EIGENGRAM magic: {data[:4]!r} (expected {EIGENGRAM_MAGIC!r})"
            )

        off = 4
        version = struct.unpack_from("<B", data, off)[0]; off += 1
        if version != EIGENGRAM_VERSION:
            raise ValueError(
                f"Unsupported EIGENGRAM version {version} "
                f"(this reader supports v{EIGENGRAM_VERSION})"
            )

        corpus_hash = data[off : off + 32].rstrip(b"\x00").decode("ascii"); off += 32
        created_at = data[off : off + 20].rstrip(b"\x00").decode("ascii"); off += 20
        model_id = data[off : off + 16].rstrip(b"\x00").decode("ascii"); off += 16

        basis_rank = struct.unpack_from("<H", data, off)[0]; off += 2
        n_corpus = struct.unpack_from("<H", data, off)[0]; off += 2
        lr0, lr1 = struct.unpack_from("<bb", data, off); off += 2
        context_len = struct.unpack_from("<I", data, off)[0]; off += 4
        l2_norm = struct.unpack_from("<f", data, off)[0]; off += 4
        scs = struct.unpack_from("<f", data, off)[0]; off += 4
        margin_proof = struct.unpack_from("<f", data, off)[0]; off += 4
        td_len = struct.unpack_from("<H", data, off)[0]; off += 2
        ci_len = struct.unpack_from("<H", data, off)[0]; off += 2

        # v1.1 extension fields: fourier_dim + local_density
        # Detect by checking if file has extra bytes beyond v1.0 layout
        fourier_dim = 0
        local_density = 0
        expected_old_size = off + basis_rank * 4 + 256 + td_len + ci_len
        eigenform_score = 1.0
        if len(data) > expected_old_size + 4:
            fourier_dim = struct.unpack_from("<H", data, off)[0]; off += 2
            local_density = struct.unpack_from("<H", data, off)[0]; off += 2
            eigenform_score = struct.unpack_from("<f", data, off)[0]; off += 4
        # If the file was written with fourier_dim field but is old format,
        # we already consumed 2 bytes. This is safe because old files
        # won't have extra bytes.

        R = basis_rank
        vec_perdoc = torch.from_numpy(
            np.frombuffer(data, dtype=np.float16, count=R, offset=off).copy()
        ).float(); off += R * 2

        vec_fcdb = torch.from_numpy(
            np.frombuffer(data, dtype=np.float16, count=R, offset=off).copy()
        ).float(); off += R * 2

        joint_center = torch.from_numpy(
            np.frombuffer(data, dtype=np.float16, count=128, offset=off).copy()
        ).float(); off += 128 * 2

        task_description = data[off : off + td_len].decode("utf-8", errors="replace"); off += td_len
        cache_id = data[off : off + ci_len].decode("utf-8", errors="replace"); off += ci_len

        # Read vec_fourier if present
        vec_fourier = None
        if fourier_dim > 0 and off + fourier_dim * 2 <= len(data):
            vec_fourier = torch.from_numpy(
                np.frombuffer(data, dtype=np.float16, count=fourier_dim, offset=off).copy()
            ).float()
            off += fourier_dim * 2

        # v1.2 extension: confusion_flag + vec_fourier_v2
        confusion_flag = False
        vec_fourier_v2 = None
        if off + 3 <= len(data):  # 1 byte flag + 2 byte dim minimum
            confusion_flag = bool(struct.unpack_from("<B", data, off)[0])
            off += 1
            v2_dim = struct.unpack_from("<H", data, off)[0]
            off += 2
            if v2_dim > 0 and off + v2_dim * 2 <= len(data):
                vec_fourier_v2 = torch.from_numpy(
                    np.frombuffer(data, dtype=np.float16, count=v2_dim, offset=off).copy()
                ).float()

        result = {
            "version": version,
            "corpus_hash": corpus_hash,
            "created_at": created_at,
            "model_id": model_id,
            "basis_rank": basis_rank,
            "n_corpus": n_corpus,
            "layer_range": (lr0, lr1),
            "context_len": context_len,
            "l2_norm": l2_norm,
            "scs": scs,
            "margin_proof": margin_proof,
            "vec_perdoc": vec_perdoc,
            "vec_fcdb": vec_fcdb,
            "joint_center": joint_center,
            "task_description": task_description,
            "cache_id": cache_id,
        }
        if vec_fourier is not None:
            result["vec_fourier"] = vec_fourier
        if vec_fourier_v2 is not None:
            result["vec_fourier_v2"] = vec_fourier_v2
        result["local_density"] = local_density
        result["eigenform_score"] = eigenform_score
        result["confusion_flag"] = confusion_flag
        return result


EigramDecoder = EigramEncoder
