"""
ENGRAM Protocol — Manifold Index Tests
Tests for FAISS IndexFlatIP add/search/remove/persist (D2, D4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from kvcos.core.manifold_index import IndexEntry, ManifoldIndex


def _entry(cid: str = "c1", model: str = "llama") -> IndexEntry:
    return IndexEntry(
        cache_id=cid, task_description="test",
        model_id=model, created_at="2026-01-01T00:00:00Z",
        context_len=256, l2_norm=1.0,
    )


class TestAddAndSearch:
    """Add vectors, search via MIPS."""

    def test_add_increments(self) -> None:
        idx = ManifoldIndex(dim=8)
        idx.add(torch.randn(8), _entry("a"))
        idx.add(torch.randn(8), _entry("b"))
        assert idx.n_entries == 2

    def test_search_returns_correct_order(self) -> None:
        idx = ManifoldIndex(dim=4)
        v1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        v2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
        idx.add(v1, _entry("close"))
        idx.add(v2, _entry("far"))

        query = torch.tensor([1.0, 0.0, 0.0, 0.0])
        results = idx.search(query, top_k=2)
        assert results[0]["cache_id"] == "close"
        assert results[0]["similarity"] > results[1]["similarity"]

    def test_search_empty_returns_empty(self) -> None:
        idx = ManifoldIndex(dim=4)
        results = idx.search(torch.randn(4), top_k=5)
        assert results == []

    def test_model_filter(self) -> None:
        idx = ManifoldIndex(dim=4)
        idx.add(torch.randn(4), _entry("a", model="llama"))
        idx.add(torch.randn(4), _entry("b", model="phi"))
        results = idx.search(torch.randn(4), top_k=10, model_id="phi")
        assert all(r["model_id"] == "phi" for r in results)


class TestRemoveAndRebuild:
    """Remove entries and rebuild index."""

    def test_remove_hides_from_search(self) -> None:
        idx = ManifoldIndex(dim=4)
        v = torch.tensor([1.0, 0.0, 0.0, 0.0])
        idx.add(v, _entry("target"))
        assert idx.remove("target")
        results = idx.search(v, top_k=1)
        assert len(results) == 0

    def test_rebuild_compacts(self) -> None:
        idx = ManifoldIndex(dim=4)
        for i in range(5):
            idx.add(torch.randn(4), _entry(f"c{i}"))
        idx.remove("c1")
        idx.remove("c3")
        active = idx.rebuild()
        assert active == 3


class TestPersistence:
    """Save/load round-trip (D2: serialize_index/deserialize_index)."""

    def test_save_load_round_trip(self, tmp_index_dir: Path) -> None:
        idx = ManifoldIndex(dim=4)
        v1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        idx.add(v1, _entry("persisted"))
        idx.save(tmp_index_dir / "test.faiss")

        idx2 = ManifoldIndex(dim=4, index_path=tmp_index_dir / "test.faiss")
        assert idx2.n_entries == 1
        results = idx2.search(v1, top_k=1)
        assert results[0]["cache_id"] == "persisted"

    def test_dim_mismatch_raises(self) -> None:
        idx = ManifoldIndex(dim=4)
        with pytest.raises(ValueError, match="dim"):
            idx.add(torch.randn(8), _entry("wrong"))
