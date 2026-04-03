"""Tests for kvcos.engram.knowledge_index — HNSW knowledge search."""

import json
from pathlib import Path

import pytest
import torch

from kvcos.engram.embedder import get_fingerprint
from kvcos.engram.format import EigramEncoder
from kvcos.engram.knowledge_index import KnowledgeIndex


@pytest.fixture
def knowledge_dir(tmp_path):
    """Create a temporary knowledge directory with test .eng files."""
    encoder = EigramEncoder()
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    docs = [
        ("doc_ml", "Machine learning model training and optimization"),
        ("doc_db", "PostgreSQL database schema migration tools"),
        ("doc_api", "REST API endpoint authentication and authorization"),
        ("doc_test", "Unit testing with pytest fixtures and mocking"),
        ("doc_deploy", "Docker container deployment to Kubernetes cluster"),
    ]

    for doc_id, text in docs:
        fp, source = get_fingerprint(text)
        dim = fp.shape[0]

        blob = encoder.encode(
            vec_perdoc=torch.zeros(116),
            vec_fcdb=torch.zeros(116),
            joint_center=torch.zeros(128),
            corpus_hash="test" * 8,
            model_id=source[:16],
            basis_rank=116,
            n_corpus=0,
            layer_range=(0, 0),
            context_len=len(text),
            l2_norm=float(torch.norm(fp).item()),
            scs=0.0,
            margin_proof=0.0,
            task_description=text[:256],
            cache_id=doc_id,
            vec_fourier=fp if dim == 2048 else None,
            vec_fourier_v2=fp,
            confusion_flag=False,
        )

        eng_path = project_dir / f"{doc_id}.eng"
        eng_path.write_bytes(blob)

        meta = {
            "cache_id": doc_id,
            "task_description": text,
            "source_path": f"/test/{doc_id}.md",
            "project": "test_project",
            "fp_source": source,
            "chunk_index": 0,
            "chunk_total": 1,
            "headers": [],
        }
        meta_path = Path(str(eng_path) + ".meta.json")
        meta_path.write_text(json.dumps(meta))

    return tmp_path


class TestKnowledgeIndexBuild:
    def test_build_from_directory(self, knowledge_dir):
        kidx = KnowledgeIndex.build_from_knowledge_dir(
            knowledge_dir, verbose=False
        )
        assert len(kidx) == 5

    def test_build_empty_directory(self, tmp_path):
        with pytest.raises(ValueError, match="No .eng files"):
            KnowledgeIndex.build_from_knowledge_dir(tmp_path, verbose=False)


class TestKnowledgeIndexSearch:
    def test_search_returns_results(self, knowledge_dir):
        kidx = KnowledgeIndex.build_from_knowledge_dir(
            knowledge_dir, verbose=False
        )
        results = kidx.search("database query optimization", k=3)
        assert len(results) == 3
        assert all(r.score > 0 for r in results)

    def test_search_result_fields(self, knowledge_dir):
        kidx = KnowledgeIndex.build_from_knowledge_dir(
            knowledge_dir, verbose=False
        )
        results = kidx.search("testing", k=1)
        r = results[0]
        assert r.doc_id
        assert isinstance(r.score, float)
        assert r.rank == 0
        assert r.project == "test_project"

    def test_search_with_tensor(self, knowledge_dir):
        kidx = KnowledgeIndex.build_from_knowledge_dir(
            knowledge_dir, verbose=False
        )
        query_fp, _ = get_fingerprint("unit tests")
        results = kidx.search(query_fp, k=2)
        assert len(results) == 2

    def test_search_margin(self, knowledge_dir):
        kidx = KnowledgeIndex.build_from_knowledge_dir(
            knowledge_dir, verbose=False
        )
        results = kidx.search("testing", k=3)
        # Top result should have a margin
        assert results[0].margin >= 0


class TestKnowledgeIndexPersistence:
    def test_save_and_load(self, knowledge_dir, tmp_path):
        kidx = KnowledgeIndex.build_from_knowledge_dir(
            knowledge_dir, verbose=False
        )
        index_dir = tmp_path / "index"
        kidx.save(index_dir)

        loaded = KnowledgeIndex.load(index_dir)
        assert len(loaded) == len(kidx)

        # Search should work on loaded index
        results = loaded.search("database", k=2)
        assert len(results) == 2

    def test_load_nonexistent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            KnowledgeIndex.load(tmp_path / "nonexistent")
