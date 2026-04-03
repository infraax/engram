"""
EIGENGRAM test suite — no model calls, pure format verification.
"""

from __future__ import annotations

import os
import struct

import pytest
import torch

from kvcos.engram.format import (
    EigramDecoder,
    EigramEncoder,
    EIGENGRAM_MAGIC,
    EIGENGRAM_VERSION,
)

BASIS_PATH = "results/corpus_basis_fcdb_v2.pt"


@pytest.fixture(scope="module")
def basis():
    if not os.path.exists(BASIS_PATH):
        pytest.skip("FCDB v2 basis not built yet")
    return torch.load(BASIS_PATH, weights_only=False)


@pytest.fixture(scope="module")
def sample_cert(basis):
    enc = EigramEncoder()
    R = basis["basis"].shape[0]
    return enc.encode(
        vec_perdoc=torch.randn(R),
        vec_fcdb=torch.randn(R),
        joint_center=basis["joint_center"],
        corpus_hash="a" * 32,
        model_id="Llama-3.1-8B",
        basis_rank=R,
        n_corpus=200,
        layer_range=(8, 24),
        context_len=512,
        l2_norm=1.234,
        scs=0.42,
        margin_proof=0.013,
        task_description="Test document for transformer attention.",
        cache_id="test-doc-001",
    )


class TestFormat:
    def test_magic_present(self, sample_cert: bytes) -> None:
        assert sample_cert[:4] == EIGENGRAM_MAGIC

    def test_version_byte(self, sample_cert: bytes) -> None:
        assert struct.unpack_from("<B", sample_cert, 4)[0] == EIGENGRAM_VERSION

    def test_minimum_size(self, sample_cert: bytes, basis) -> None:
        R = basis["basis"].shape[0]
        min_size = 99 + R * 2 + R * 2 + 128 * 2
        assert len(sample_cert) >= min_size

    def test_file_size_reasonable(self, sample_cert: bytes) -> None:
        assert len(sample_cert) < 2048


class TestRoundTrip:
    def test_model_id(self, sample_cert: bytes) -> None:
        rec = EigramDecoder().decode(sample_cert)
        assert rec["model_id"] == "Llama-3.1-8B"

    def test_basis_rank(self, sample_cert: bytes, basis) -> None:
        rec = EigramDecoder().decode(sample_cert)
        assert rec["basis_rank"] == basis["basis"].shape[0]

    def test_vec_perdoc_shape(self, sample_cert: bytes, basis) -> None:
        rec = EigramDecoder().decode(sample_cert)
        assert rec["vec_perdoc"].shape == (basis["basis"].shape[0],)

    def test_vec_fcdb_shape(self, sample_cert: bytes, basis) -> None:
        rec = EigramDecoder().decode(sample_cert)
        assert rec["vec_fcdb"].shape == (basis["basis"].shape[0],)

    def test_joint_center_shape(self, sample_cert: bytes) -> None:
        rec = EigramDecoder().decode(sample_cert)
        assert rec["joint_center"].shape == (128,)

    def test_scs(self, sample_cert: bytes) -> None:
        rec = EigramDecoder().decode(sample_cert)
        assert abs(rec["scs"] - 0.42) < 0.01

    def test_margin_proof(self, sample_cert: bytes) -> None:
        rec = EigramDecoder().decode(sample_cert)
        assert abs(rec["margin_proof"] - 0.013) < 0.001

    def test_task_description(self, sample_cert: bytes) -> None:
        rec = EigramDecoder().decode(sample_cert)
        assert "transformer" in rec["task_description"]

    def test_cache_id(self, sample_cert: bytes) -> None:
        rec = EigramDecoder().decode(sample_cert)
        assert rec["cache_id"] == "test-doc-001"

    def test_layer_range(self, sample_cert: bytes) -> None:
        rec = EigramDecoder().decode(sample_cert)
        assert rec["layer_range"] == (8, 24)

    def test_n_corpus(self, sample_cert: bytes) -> None:
        rec = EigramDecoder().decode(sample_cert)
        assert rec["n_corpus"] == 200

    def test_context_len(self, sample_cert: bytes) -> None:
        rec = EigramDecoder().decode(sample_cert)
        assert rec["context_len"] == 512

    def test_float16_cosine_preserved(self, basis) -> None:
        enc = EigramEncoder()
        R = basis["basis"].shape[0]
        v = torch.randn(R)
        v = v / v.norm()
        cert = enc.encode(
            vec_perdoc=v, vec_fcdb=v,
            joint_center=basis["joint_center"],
            corpus_hash="a" * 32, model_id="test",
            basis_rank=R, n_corpus=200,
            layer_range=(8, 24), context_len=0,
            l2_norm=1.0, scs=0.5, margin_proof=0.0,
            task_description="cosine test", cache_id="cos",
        )
        rec = EigramDecoder().decode(cert)
        cos = torch.nn.functional.cosine_similarity(
            v.unsqueeze(0), rec["vec_perdoc"].unsqueeze(0)
        ).item()
        assert cos > 0.999, f"Cosine after round-trip: {cos:.5f}"


class TestErrorHandling:
    def test_bad_magic_raises(self) -> None:
        bad = b"XXXX" + b"\x00" * 200
        with pytest.raises(ValueError, match="magic"):
            EigramDecoder().decode(bad)

    def test_wrong_version_raises(self, sample_cert: bytes) -> None:
        data = bytearray(sample_cert)
        data[4] = 99
        with pytest.raises(ValueError, match="version"):
            EigramDecoder().decode(bytes(data))

    def test_truncated_raises(self, sample_cert: bytes) -> None:
        with pytest.raises(Exception):
            EigramDecoder().decode(sample_cert[:20])
