"""Tests for kvcos.engram.hnsw_index — HNSW nearest-neighbor index."""

import torch
import pytest

from kvcos.engram.hnsw_index import EngramIndex, HNSWResult


@pytest.fixture
def small_index():
    """Build a small 8-dim HNSW index with 5 documents."""
    idx = EngramIndex(dim=8)
    ids = [f"doc_{i}" for i in range(5)]
    # deterministic orthogonal-ish vectors
    vecs = torch.eye(5, 8)
    idx.add_batch(ids, vecs)
    return idx


class TestEngramIndexBuild:
    def test_add_batch_len(self, small_index):
        assert len(small_index) == 5

    def test_add_batch_ids_stored(self, small_index):
        assert small_index._ids == [f"doc_{i}" for i in range(5)]

    def test_repr(self, small_index):
        r = repr(small_index)
        assert "n=5" in r
        assert "dim=8" in r


class TestEngramIndexSearch:
    def test_search_returns_results(self, small_index):
        query = torch.eye(5, 8)[0]  # matches doc_0
        results = small_index.search(query, top_k=3)
        assert len(results) == 3
        assert results[0].doc_id == "doc_0"

    def test_search_scores_descending(self, small_index):
        query = torch.eye(5, 8)[2]
        results = small_index.search(query, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_margin(self, small_index):
        query = torch.eye(5, 8)[0]
        results = small_index.search(query, top_k=3)
        assert results[0].margin >= 0

    def test_search_raises_before_build(self):
        idx = EngramIndex(dim=8)
        with pytest.raises(RuntimeError, match="not built"):
            idx.search(torch.randn(8), top_k=1)


class TestEngramIndexGetVector:
    def test_get_vector_returns_tensor(self, small_index):
        vec = small_index.get_vector("doc_0")
        assert vec is not None
        assert isinstance(vec, torch.Tensor)
        assert vec.shape == (8,)

    def test_get_vector_none_for_missing(self, small_index):
        vec = small_index.get_vector("nonexistent")
        assert vec is None

    def test_get_vector_reconstructs_normalized(self, small_index):
        """Vectors are L2-normalized on add, so reconstruction should be unit-length."""
        vec = small_index.get_vector("doc_0")
        norm = torch.norm(vec).item()
        assert abs(norm - 1.0) < 0.01

    def test_get_vector_matches_original_direction(self, small_index):
        """Reconstructed vector should point in the same direction as the original."""
        original = torch.nn.functional.normalize(torch.eye(5, 8)[3:4], dim=-1)[0]
        reconstructed = small_index.get_vector("doc_3")
        cosine = torch.dot(original, reconstructed).item()
        assert cosine > 0.99


class TestEngramIndexPersistence:
    def test_save_and_load(self, small_index, tmp_path):
        path = str(tmp_path / "test_hnsw")
        small_index.save(path)

        loaded = EngramIndex.load(path)
        assert len(loaded) == 5
        assert loaded._ids == small_index._ids

    def test_loaded_search_matches_original(self, small_index, tmp_path):
        path = str(tmp_path / "test_hnsw")
        small_index.save(path)
        loaded = EngramIndex.load(path)

        query = torch.eye(5, 8)[1]
        orig_results = small_index.search(query, top_k=3)
        load_results = loaded.search(query, top_k=3)
        assert [r.doc_id for r in orig_results] == [r.doc_id for r in load_results]

    def test_loaded_get_vector(self, small_index, tmp_path):
        path = str(tmp_path / "test_hnsw")
        small_index.save(path)
        loaded = EngramIndex.load(path)

        vec = loaded.get_vector("doc_2")
        assert vec is not None
        assert vec.shape == (8,)
