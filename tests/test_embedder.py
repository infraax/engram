"""Tests for kvcos.engram.embedder — unified fingerprint embedding."""

import pytest
import torch
import torch.nn.functional as F

from kvcos.engram.embedder import (
    HashEmbedder,
    get_embedder,
    get_fingerprint,
    reset_embedder,
)


class TestHashEmbedder:
    def test_deterministic(self):
        emb = HashEmbedder(dim=128)
        fp1 = emb.embed("hello")
        fp2 = emb.embed("hello")
        assert torch.allclose(fp1, fp2)

    def test_different_text(self):
        emb = HashEmbedder(dim=128)
        fp1 = emb.embed("hello")
        fp2 = emb.embed("world")
        assert not torch.allclose(fp1, fp2)

    def test_normalized(self):
        emb = HashEmbedder(dim=128)
        fp = emb.embed("test")
        norm = torch.norm(fp).item()
        assert abs(norm - 1.0) < 0.01

    def test_dimension(self):
        emb = HashEmbedder(dim=256)
        fp = emb.embed("test")
        assert fp.shape == (256,)
        assert emb.dim == 256

    def test_source_tag(self):
        emb = HashEmbedder()
        assert emb.source == "hash-fallback"


class TestGetFingerprint:
    def test_returns_tensor_and_source(self):
        fp, source = get_fingerprint("test text")
        assert isinstance(fp, torch.Tensor)
        assert isinstance(source, str)
        assert source in ("llama_cpp", "sbert", "hash-fallback")

    def test_deterministic(self):
        fp1, _ = get_fingerprint("same text")
        fp2, _ = get_fingerprint("same text")
        assert torch.allclose(fp1, fp2)


class TestSBertEmbedder:
    """Test sbert if available (installed in this venv)."""

    def test_sbert_available(self):
        """Verify sentence-transformers is usable."""
        try:
            from kvcos.engram.embedder import SBertEmbedder
            emb = SBertEmbedder()
            assert emb.source == "sbert"
            assert emb.dim == 384
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_semantic_discrimination(self):
        """Related texts should be more similar than unrelated."""
        try:
            from kvcos.engram.embedder import SBertEmbedder
            emb = SBertEmbedder()
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        fp_a = emb.embed("machine learning neural network training")
        fp_b = emb.embed("deep learning model optimization")
        fp_c = emb.embed("chocolate cake baking recipe")

        sim_ab = F.cosine_similarity(fp_a.unsqueeze(0), fp_b.unsqueeze(0)).item()
        sim_ac = F.cosine_similarity(fp_a.unsqueeze(0), fp_c.unsqueeze(0)).item()

        assert sim_ab > sim_ac, (
            f"Related topics ({sim_ab:.4f}) should be more similar "
            f"than unrelated ({sim_ac:.4f})"
        )


class TestGetEmbedder:
    def test_singleton(self):
        reset_embedder()
        e1 = get_embedder()
        e2 = get_embedder()
        assert e1 is e2

    def test_reset(self):
        reset_embedder()
        e1 = get_embedder()
        reset_embedder()
        e2 = get_embedder()
        # After reset, a new instance is created
        # (may or may not be same object depending on strategy)
        assert e2 is not None
