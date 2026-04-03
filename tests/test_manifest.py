"""Tests for kvcos.engram.manifest — knowledge index registry."""

import json
import tempfile
from pathlib import Path

import pytest

from kvcos.engram.manifest import ChunkRecord, Manifest, SourceRecord, _content_hash


@pytest.fixture
def tmp_manifest(tmp_path):
    """Create a Manifest with a temporary path."""
    return Manifest.load(tmp_path / "manifest.json")


class TestContentHash:
    def test_deterministic(self):
        assert _content_hash("hello") == _content_hash("hello")

    def test_different_content(self):
        assert _content_hash("hello") != _content_hash("world")


class TestManifestLoad:
    def test_load_nonexistent_creates_empty(self, tmp_path):
        m = Manifest.load(tmp_path / "does_not_exist.json")
        assert m.total_sources == 0
        assert m.total_chunks == 0

    def test_load_existing(self, tmp_path):
        # Write a manifest, then load it
        m = Manifest.load(tmp_path / "manifest.json")
        m = m.register(
            source_path="/test/file.md",
            content_hash="abc123",
            project="test",
            file_size=100,
            chunks=[ChunkRecord(
                eng_path="/test/file.eng",
                chunk_index=0,
                chunk_total=1,
                char_start=0,
                char_end=100,
                indexed_at=1000.0,
            )],
        )

        # Load again from disk
        m2 = Manifest.load(tmp_path / "manifest.json")
        assert m2.total_sources == 1
        assert m2.total_chunks == 1


class TestManifestRegister:
    def test_register_new(self, tmp_manifest):
        chunks = [ChunkRecord(
            eng_path="/out/test.eng",
            chunk_index=0,
            chunk_total=1,
            char_start=0,
            char_end=50,
            indexed_at=1000.0,
        )]
        m = tmp_manifest.register(
            source_path="/src/test.md",
            content_hash="hash1",
            project="myproject",
            file_size=50,
            chunks=chunks,
        )
        assert m.total_sources == 1
        assert m.total_chunks == 1
        assert "myproject" in m.projects

    def test_register_overwrites_existing(self, tmp_manifest):
        chunks1 = [ChunkRecord(
            eng_path="/out/v1.eng", chunk_index=0, chunk_total=1,
            char_start=0, char_end=50, indexed_at=1000.0,
        )]
        m = tmp_manifest.register(
            "/src/test.md", "hash1", "proj", 50, chunks1,
        )
        assert m.total_chunks == 1

        chunks2 = [
            ChunkRecord("/out/v2_1.eng", 0, 2, 0, 25, 2000.0),
            ChunkRecord("/out/v2_2.eng", 1, 2, 25, 50, 2000.0),
        ]
        m = m.register("/src/test.md", "hash2", "proj", 50, chunks2)
        assert m.total_sources == 1  # still 1 source
        assert m.total_chunks == 2   # now 2 chunks

    def test_register_returns_new_manifest(self, tmp_manifest):
        """Register returns a new Manifest (immutability)."""
        m1 = tmp_manifest
        m2 = m1.register("/src/a.md", "h", "p", 10, [])
        assert m1.total_sources == 0  # original unchanged
        assert m2.total_sources == 1


class TestManifestNeedsReindex:
    def test_unknown_file_needs_index(self, tmp_manifest):
        assert tmp_manifest.needs_reindex("/new/file.md", "any_hash")

    def test_same_hash_no_reindex(self, tmp_manifest):
        m = tmp_manifest.register("/src/a.md", "hash1", "p", 10, [])
        assert not m.needs_reindex("/src/a.md", "hash1")

    def test_different_hash_needs_reindex(self, tmp_manifest):
        m = tmp_manifest.register("/src/a.md", "hash1", "p", 10, [])
        assert m.needs_reindex("/src/a.md", "hash2")


class TestManifestUnregister:
    def test_unregister_existing(self, tmp_manifest):
        m = tmp_manifest.register("/src/a.md", "h", "p", 10, [])
        m = m.unregister("/src/a.md")
        assert m.total_sources == 0

    def test_unregister_nonexistent(self, tmp_manifest):
        m = tmp_manifest.unregister("/not/here.md")
        assert m.total_sources == 0


class TestManifestQueries:
    def test_get_project_records(self, tmp_manifest):
        m = tmp_manifest
        m = m.register("/a.md", "h1", "proj_a", 10, [])
        m = m.register("/b.md", "h2", "proj_b", 20, [])
        m = m.register("/c.md", "h3", "proj_a", 30, [])

        a_recs = m.get_project_records("proj_a")
        assert len(a_recs) == 2

    def test_summary(self, tmp_manifest):
        m = tmp_manifest.register("/a.md", "h", "p", 10, [
            ChunkRecord("/a.eng", 0, 1, 0, 10, 1000.0),
        ])
        s = m.summary()
        assert s["total_sources"] == 1
        assert s["total_chunks"] == 1
        assert "p" in s["projects"]

    def test_contains(self, tmp_manifest):
        m = tmp_manifest.register("/a.md", "h", "p", 10, [])
        assert "/a.md" in m
        assert "/b.md" not in m

    def test_len(self, tmp_manifest):
        m = tmp_manifest.register("/a.md", "h", "p", 10, [])
        assert len(m) == 1


class TestManifestPersistence:
    def test_atomic_write(self, tmp_path):
        m = Manifest.load(tmp_path / "manifest.json")
        m = m.register("/a.md", "h", "p", 10, [])

        # File should exist
        assert (tmp_path / "manifest.json").exists()

        # Content should be valid JSON
        data = json.loads((tmp_path / "manifest.json").read_text())
        assert data["version"] == 1
        assert len(data["sources"]) == 1
