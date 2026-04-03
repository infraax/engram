#!/usr/bin/env python3
"""
mcp/engram_memory.py — ENGRAM Session Memory MCP Server

Three tools for Claude Code to persist and retrieve session memory
using the ENGRAM fingerprint protocol.

Install:
  claude mcp add --global engram-memory \
    -e ENGRAM_SESSIONS_DIR=~/.engram/sessions \
    -- python3 /path/to/mcp/engram_memory.py

Tools:
  write_session_engram    Encode + store terminal session state
  get_last_session        Fast-path: newest session terminal state
  retrieve_relevant_sessions  Semantic search over stored sessions

Session summary format (enforce in prompts):
  VALIDATED: <confirmed results, metrics>
  CURRENT:   <current system state, file locations>
  NEXT:      <next session priorities, in order>
  OPEN:      <unresolved items, known failures>
"""

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "mcp package required: pip install mcp"
    )

SESSIONS_DIR = Path(
    os.environ.get("ENGRAM_SESSIONS_DIR", "~/.engram/sessions")
).expanduser()
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

ENGRAM_PROJECT = Path(
    os.environ.get("ENGRAM_PROJECT_DIR",
                   Path(__file__).parent.parent)
)

# Eager imports — load torch/numpy/faiss at startup so the first tool call
# doesn't hang for 3-5 seconds while Claude Code shows "connecting..."
sys.path.insert(0, str(ENGRAM_PROJECT))
import numpy as np          # noqa: E402
import torch                # noqa: E402
import torch.nn.functional as F  # noqa: E402
from kvcos.engram.format import EigramEncoder  # noqa: E402

_encoder = EigramEncoder()

mcp = FastMCP("engram-memory")


# ── Encoding helpers ──────────────────────────────────────────────────

from kvcos.engram.embedder import get_fingerprint as _get_fingerprint  # noqa: E402


def _write_eng(fp_tensor: torch.Tensor, summary: str, session_id: str,
               domain: str, fp_source: str) -> Path:
    """Write a real EIGENGRAM .eng binary using the format codec."""
    dim = fp_tensor.shape[0]

    # Placeholder vectors for corpus-specific fields not relevant to sessions
    basis_rank = 116
    vec_perdoc = torch.zeros(basis_rank)
    vec_fcdb = torch.zeros(basis_rank)
    joint_center = torch.zeros(128)

    blob = _encoder.encode(
        vec_perdoc=vec_perdoc,
        vec_fcdb=vec_fcdb,
        joint_center=joint_center,
        corpus_hash=hashlib.sha256(session_id.encode()).hexdigest()[:32],
        model_id=fp_source[:16],
        basis_rank=basis_rank,
        n_corpus=0,
        layer_range=(0, 0),
        context_len=len(summary),
        l2_norm=float(torch.norm(fp_tensor).item()),
        scs=0.0,
        margin_proof=0.0,
        task_description=summary[:256],
        cache_id=session_id,
        vec_fourier=fp_tensor if dim == 2048 else None,
        vec_fourier_v2=fp_tensor,
        confusion_flag=False,
    )

    eng_path = SESSIONS_DIR / f"{session_id}.eng"
    with open(eng_path, "wb") as f:
        f.write(blob)

    # Write a small JSON sidecar for fields the binary doesn't carry
    # (domain, fp_source, full summary beyond 256 chars, timestamp)
    meta_path = SESSIONS_DIR / f"{session_id}.eng.meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "cache_id": session_id,
            "task_description": summary[:500],
            "domain": domain,
            "fp_source": fp_source,
            "ts": time.time(),
        }, f)

    return eng_path


def _load_sessions() -> list[dict]:
    """Load all stored session .eng files using the EIGENGRAM codec."""
    records = []

    for p in sorted(SESSIONS_DIR.glob("*.eng"), key=os.path.getmtime):
        if p.suffix != ".eng":
            continue
        try:
            data = _encoder.decode(p.read_bytes())
            # Merge metadata sidecar if it exists (domain, fp_source, full summary, ts)
            meta_path = Path(str(p) + ".meta.json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                data["domain"] = meta.get("domain", "")
                data["fp_source"] = meta.get("fp_source", "unknown")
                data["ts"] = meta.get("ts", 0.0)
                # Sidecar may have longer task_description than the 256-char binary limit
                if len(meta.get("task_description", "")) > len(data.get("task_description", "")):
                    data["task_description"] = meta["task_description"]
            records.append(data)
        except Exception as exc:
            logger.debug("Skipping session %s: %s", p, exc)

    return records


def _cosine(a, b) -> float:
    """Cosine similarity between two vectors (list or torch.Tensor)."""
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float32)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, dtype=torch.float32)
    return float(F.cosine_similarity(a.float().flatten().unsqueeze(0),
                                     b.float().flatten().unsqueeze(0)).item())


# ── MCP Tools ──────────────────────────────────────────────────────────

@mcp.tool()
def write_session_engram(
    session_summary: str,
    session_id:      str  = "",
    domain:          str  = "engram",
) -> str:
    """
    Encode the terminal session state and store as a session memory file.

    Call at the END of every Claude Code session.

    The session_summary should follow this format for best retrieval:
        VALIDATED: <confirmed results, accuracy metrics>
        CURRENT:   <current file locations, system state>
        NEXT:      <prioritised next steps>
        OPEN:      <unresolved items, known failures>

    Args:
        session_summary: Terminal session state (use format above).
        session_id:      Unique ID, e.g. "s6_2026-04-02".
                         Auto-generated from timestamp if empty.
        domain:          Domain tag for density hinting (default: "engram").

    Returns:
        Path to stored .eng file (EIGENGRAM binary format).
    """
    if not session_id:
        session_id = f"session_{int(time.time())}"

    fp_list, fp_source = _get_fingerprint(session_summary)
    eng_path = _write_eng(fp_list, session_summary, session_id,
                          domain, fp_source)

    return json.dumps({
        "stored":     str(eng_path),
        "session_id": session_id,
        "fp_source":  fp_source,
        "chars":      len(session_summary),
    })


@mcp.tool()
def get_last_session() -> str:
    """
    Return the terminal state of the most recent stored session.

    Call at the START of every Claude Code session before doing anything.
    This is the fast path — no semantic search, just the newest file.

    Returns:
        JSON with session_id and task_description (terminal state summary).
        Returns empty JSON if no sessions are stored yet.
    """
    records = _load_sessions()
    if not records:
        return json.dumps({"status": "no sessions stored"})

    latest = records[-1]
    return json.dumps({
        "session_id":   latest.get("cache_id"),
        "terminal_state": latest.get("task_description"),
        "stored_at":    latest.get("ts"),
        "fp_source":    latest.get("fp_source"),
    })


@mcp.tool()
def retrieve_relevant_sessions(
    query: str,
    k:     int = 3,
) -> str:
    """
    Semantic search over all stored session memories.

    Call when starting a complex task that may have relevant prior work.
    Returns k most semantically similar prior sessions to the query.

    Args:
        query: Description of the current task.
        k:     Number of sessions to return (default 3).

    Returns:
        JSON list of k most relevant sessions with their terminal states.
    """
    records = _load_sessions()
    if not records:
        return json.dumps([])

    query_fp, _ = _get_fingerprint(query)

    scored = []
    for rec in records:
        # Decoded .eng files have vec_fourier_v2 as torch.Tensor
        fp = rec.get("vec_fourier_v2")
        if fp is None:
            fp = rec.get("vec_fourier")
        if fp is None:
            continue
        sim = _cosine(query_fp, fp)
        scored.append({
            "session_id":    rec.get("cache_id"),
            "terminal_state": rec.get("task_description"),
            "similarity":    round(sim, 4),
            "fp_source":     rec.get("fp_source", "unknown"),
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return json.dumps(scored[:k], indent=2)


# ── Knowledge Index Tools ─────────────────────────────────────────────

KNOWLEDGE_DIR = Path(
    os.environ.get("ENGRAM_KNOWLEDGE_DIR", "~/.engram/knowledge")
).expanduser()


def _load_knowledge(project: str = "") -> list[dict]:
    """Load all .eng files from the knowledge index."""
    records = []

    if project:
        search_dir = KNOWLEDGE_DIR / project
        if not search_dir.exists():
            return records
        eng_files = sorted(search_dir.glob("*.eng"), key=os.path.getmtime)
    else:
        eng_files = sorted(KNOWLEDGE_DIR.rglob("*.eng"), key=os.path.getmtime)

    for p in eng_files:
        if p.suffix != ".eng":
            continue
        try:
            data = _encoder.decode(p.read_bytes())
            meta_path = Path(str(p) + ".meta.json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                data["source_path"] = meta.get("source_path", "")
                data["project"] = meta.get("project", "")
                data["fp_source"] = meta.get("fp_source", "unknown")
                data["chunk_index"] = meta.get("chunk_index", 0)
                data["chunk_total"] = meta.get("chunk_total", 1)
                data["headers"] = meta.get("headers", [])
                data["type"] = meta.get("type", "knowledge")
                if len(meta.get("task_description", "")) > len(
                    data.get("task_description", "")
                ):
                    data["task_description"] = meta["task_description"]
            records.append(data)
        except Exception as exc:
            logger.debug("Skipping knowledge file %s: %s", p, exc)

    return records


_knowledge_index = None
_knowledge_index_mtime = 0.0

INDEX_DIR = Path(
    os.environ.get("ENGRAM_INDEX_DIR", "~/.engram/index")
).expanduser()


def _get_knowledge_index():
    """Load or rebuild the HNSW knowledge index (cached)."""
    global _knowledge_index, _knowledge_index_mtime

    faiss_path = INDEX_DIR / "knowledge.faiss"
    if faiss_path.exists():
        current_mtime = faiss_path.stat().st_mtime
        if _knowledge_index is not None and current_mtime <= _knowledge_index_mtime:
            return _knowledge_index
        try:
            from kvcos.engram.knowledge_index import KnowledgeIndex
            _knowledge_index = KnowledgeIndex.load(INDEX_DIR)
            _knowledge_index_mtime = current_mtime
            return _knowledge_index
        except Exception as exc:
            logger.warning("Failed to load knowledge index: %s", exc)

    # No pre-built index — build on demand
    try:
        from kvcos.engram.knowledge_index import KnowledgeIndex
        kidx = KnowledgeIndex.build_from_knowledge_dir(verbose=False)
        kidx.save(INDEX_DIR)
        _knowledge_index = kidx
        _knowledge_index_mtime = time.time()
        return kidx
    except Exception as exc:
        logger.warning("Failed to build knowledge index: %s", exc)
        return None


@mcp.tool()
def get_relevant_context(
    query: str,
    k:     int = 5,
    project: str = "",
) -> str:
    """
    Semantic search over the ENGRAM knowledge index.

    Searches all indexed markdown files (rules, docs, geodesics, etc.)
    for chunks most relevant to the query. Uses HNSW for sub-ms search.

    Args:
        query:   Description of what you're looking for.
        k:       Number of results to return (default 5).
        project: Filter by project namespace (empty = search all).

    Returns:
        JSON list of k most relevant knowledge chunks with source info.
    """
    kidx = _get_knowledge_index()

    if kidx is not None:
        # Fast path: HNSW search
        results = kidx.search(query, k=k * 2 if project else k)
        scored = []
        for r in results:
            if project and r.project != project:
                continue
            scored.append({
                "content":      r.content,
                "source_path":  r.source_path,
                "project":      r.project,
                "chunk":        r.chunk_info,
                "headers":      r.headers,
                "similarity":   round(r.score, 4),
                "fp_source":    r.doc_id,
            })
            if len(scored) >= k:
                break
        return json.dumps(scored[:k], indent=2)

    # Fallback: brute-force scan (no HNSW index available)
    records = _load_knowledge(project)
    if not records:
        return json.dumps({"status": "no knowledge indexed",
                           "hint": "Run: python scripts/index_knowledge.py"})

    query_fp, _ = _get_fingerprint(query)

    scored = []
    for rec in records:
        fp = rec.get("vec_fourier_v2")
        if fp is None:
            fp = rec.get("vec_fourier")
        if fp is None:
            continue
        sim = _cosine(query_fp, fp)
        scored.append({
            "content":      rec.get("task_description", ""),
            "source_path":  rec.get("source_path", ""),
            "project":      rec.get("project", ""),
            "chunk":        f"{rec.get('chunk_index', 0)+1}/{rec.get('chunk_total', 1)}",
            "headers":      rec.get("headers", []),
            "similarity":   round(sim, 4),
            "fp_source":    rec.get("fp_source", "unknown"),
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return json.dumps(scored[:k], indent=2)


@mcp.tool()
def list_indexed(
    project: str = "",
) -> str:
    """
    List all indexed knowledge files and their chunk counts.

    Args:
        project: Filter by project namespace (empty = list all).

    Returns:
        JSON summary of the knowledge index.
    """
    manifest_path = Path(
        os.environ.get("ENGRAM_MANIFEST_PATH", "~/.engram/manifest.json")
    ).expanduser()

    if not manifest_path.exists():
        return json.dumps({"status": "no manifest found",
                           "hint": "Run: python scripts/index_knowledge.py"})

    data = json.loads(manifest_path.read_text())
    sources = data.get("sources", {})

    if project:
        sources = {
            k: v for k, v in sources.items()
            if v.get("project") == project
        }

    summary = {
        "total_sources": len(sources),
        "total_chunks": sum(len(s.get("chunks", [])) for s in sources.values()),
        "projects": sorted({s.get("project", "") for s in sources.values()}),
        "files": [
            {
                "path": s.get("source_path", k).split("/")[-1],
                "project": s.get("project", ""),
                "chunks": len(s.get("chunks", [])),
                "size": s.get("file_size", 0),
            }
            for k, s in sorted(sources.items())
        ],
    }

    return json.dumps(summary, indent=2)


@mcp.tool()
def index_knowledge(
    source_path: str,
    project:     str = "engram",
    force:       bool = False,
) -> str:
    """
    Index a markdown file or directory into the ENGRAM knowledge index.

    Processes markdown files into fingerprinted .eng chunks that
    are searchable via get_relevant_context().

    Args:
        source_path: Path to a .md file or directory of .md files.
        project:     Project namespace (default: "engram").
        force:       Re-index even if content unchanged (default: false).

    Returns:
        JSON summary of indexing results.
    """
    from pathlib import Path as P
    source = P(source_path).expanduser().resolve()

    if not source.exists():
        return json.dumps({"error": f"Path not found: {source_path}"})

    try:
        # Import indexer (avoid circular imports)
        sys.path.insert(0, str(ENGRAM_PROJECT / "scripts"))
        from index_knowledge import index_batch

        stats = index_batch(
            source=source,
            project=project,
            incremental=not force,
            dry_run=False,
            force=force,
        )
        return json.dumps(stats, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run()
