#!/usr/bin/env python3
"""
scripts/index_knowledge.py — Batch index markdown files into .eng binaries.

Processes markdown files from a directory (or single file), chunks them,
fingerprints each chunk, and writes .eng files to the knowledge index.

Usage:
    # Index a single file
    python scripts/index_knowledge.py --source path/to/file.md --project engram

    # Index a directory recursively
    python scripts/index_knowledge.py --source path/to/docs/ --project engram

    # Re-index changed files only (incremental)
    python scripts/index_knowledge.py --source path/to/docs/ --project engram --incremental

    # Dry run — show what would be indexed
    python scripts/index_knowledge.py --source path/to/docs/ --project engram --dry-run

    # Force re-index everything
    python scripts/index_knowledge.py --source path/to/docs/ --project engram --force

Environment:
    ENGRAM_SESSIONS_DIR   Base sessions dir (default: ~/.engram/sessions)
    ENGRAM_KNOWLEDGE_DIR  Knowledge index dir (default: ~/.engram/knowledge)
    ENGRAM_MODEL_PATH     Path to GGUF model for real fingerprints (optional)
    PYTHONPATH=.          Must include project root for kvcos imports
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from kvcos.engram.chunker import Chunk, chunk_markdown, eng_filename, slug_from_path
from kvcos.engram.format import EigramEncoder
from kvcos.engram.manifest import ChunkRecord, Manifest, _content_hash, _file_hash


# ── Configuration ────────────────────────────────────────────────────

KNOWLEDGE_DIR = Path(
    os.environ.get("ENGRAM_KNOWLEDGE_DIR", "~/.engram/knowledge")
).expanduser()

SKIP_PATTERNS = {
    "node_modules",
    ".venv",
    "__pycache__",
    ".git",
    ".eng",
    "site-packages",
}

SKIP_FILES = {
    "LICENSE.md",
    "CHANGELOG.md",
    "SECURITY.md",
}


# ── Fingerprinting ──────────────────────────────────────────────────

from kvcos.engram.embedder import get_fingerprint as _get_fingerprint


# ── .eng Writer ──────────────────────────────────────────────────────

_encoder = EigramEncoder()


def _write_knowledge_eng(
    fp_tensor: torch.Tensor,
    chunk: Chunk,
    eng_path: Path,
    session_id: str,
    fp_source: str,
    source_path: str,
    project: str,
    chunk_index: int,
    chunk_total: int,
) -> Path:
    """Write a .eng binary for a knowledge chunk."""
    dim = fp_tensor.shape[0]
    basis_rank = 116
    vec_perdoc = torch.zeros(basis_rank)
    vec_fcdb = torch.zeros(basis_rank)
    joint_center = torch.zeros(128)

    # Truncate description to 256 chars for binary
    description = chunk.text[:256]

    blob = _encoder.encode(
        vec_perdoc=vec_perdoc,
        vec_fcdb=vec_fcdb,
        joint_center=joint_center,
        corpus_hash=hashlib.sha256(source_path.encode()).hexdigest()[:32],
        model_id=fp_source[:16],
        basis_rank=basis_rank,
        n_corpus=0,
        layer_range=(0, 0),
        context_len=len(chunk.text),
        l2_norm=float(torch.norm(fp_tensor).item()),
        scs=0.0,
        margin_proof=0.0,
        task_description=description,
        cache_id=session_id,
        vec_fourier=fp_tensor if dim == 2048 else None,
        vec_fourier_v2=fp_tensor,
        confusion_flag=False,
    )

    eng_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eng_path, "wb") as f:
        f.write(blob)

    # Write extended sidecar with full metadata
    meta = {
        "cache_id": session_id,
        "task_description": chunk.text[:500],
        "source_path": source_path,
        "project": project,
        "fp_source": fp_source,
        "chunk_index": chunk_index,
        "chunk_total": chunk_total,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "headers": list(chunk.headers),
        "ts": time.time(),
        "type": "knowledge",
    }
    meta_path = Path(str(eng_path) + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return eng_path


# ── Discovery ────────────────────────────────────────────────────────

def discover_markdown_files(source: Path) -> list[Path]:
    """Find all indexable .md files under source path."""
    if source.is_file():
        return [source] if source.suffix == ".md" else []

    files: list[Path] = []
    for p in sorted(source.rglob("*.md")):
        # Skip files in excluded directories
        if any(skip in p.parts for skip in SKIP_PATTERNS):
            continue
        # Skip excluded filenames
        if p.name in SKIP_FILES:
            continue
        # Skip empty files
        if p.stat().st_size == 0:
            continue
        files.append(p)

    return files


# ── Main Pipeline ────────────────────────────────────────────────────

def index_file(
    source_path: Path,
    project: str,
    manifest: Manifest,
    date_str: str,
    dry_run: bool = False,
    force: bool = False,
) -> tuple[Manifest, int]:
    """
    Index a single markdown file into .eng chunks.

    Returns:
        (updated_manifest, chunks_written)
    """
    content = source_path.read_text(encoding="utf-8", errors="replace")
    content_hash = _content_hash(content)

    # Incremental: skip if unchanged
    if not force and not manifest.needs_reindex(str(source_path), content_hash):
        return manifest, 0

    slug = slug_from_path(str(source_path))
    context = f"Source: {source_path.name} | Project: {project}"

    # Chunk the content
    chunks = chunk_markdown(
        content,
        max_chars=2000,
        min_chars=100,
        context_prefix=context,
    )

    if dry_run:
        print(f"  [DRY RUN] {source_path.name}: {len(chunks)} chunks, "
              f"{len(content)} chars")
        return manifest, len(chunks)

    # Write .eng for each chunk
    chunk_records: list[ChunkRecord] = []
    project_dir = KNOWLEDGE_DIR / project
    project_dir.mkdir(parents=True, exist_ok=True)

    for chunk in chunks:
        filename = eng_filename(
            project=project,
            slug=slug,
            date=date_str,
            chunk_index=chunk.index,
            chunk_total=len(chunks),
        )
        eng_path = project_dir / filename

        # Fingerprint the chunk text (with context)
        fp_tensor, fp_source = _get_fingerprint(chunk.text)

        session_id = f"{project}/{slug}"
        if len(chunks) > 1:
            session_id += f"_c{chunk.index + 1:03d}"

        _write_knowledge_eng(
            fp_tensor=fp_tensor,
            chunk=chunk,
            eng_path=eng_path,
            session_id=session_id,
            fp_source=fp_source,
            source_path=str(source_path),
            project=project,
            chunk_index=chunk.index,
            chunk_total=len(chunks),
        )

        chunk_records.append(ChunkRecord(
            eng_path=str(eng_path),
            chunk_index=chunk.index,
            chunk_total=len(chunks),
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            indexed_at=time.time(),
        ))

    # Register in manifest
    manifest = manifest.register(
        source_path=str(source_path),
        content_hash=content_hash,
        project=project,
        file_size=len(content.encode("utf-8")),
        chunks=chunk_records,
    )

    return manifest, len(chunks)


def index_batch(
    source: Path,
    project: str,
    incremental: bool = True,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """
    Index all markdown files under source path.

    Returns summary dict with stats.
    """
    manifest = Manifest.load()
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    files = discover_markdown_files(source)
    if not files:
        return {"error": f"No .md files found under {source}"}

    stats = {
        "source": str(source),
        "project": project,
        "files_found": len(files),
        "files_indexed": 0,
        "files_skipped": 0,
        "chunks_written": 0,
        "dry_run": dry_run,
        "incremental": incremental,
        "date": date_str,
    }

    print(f"\nENGRAM Knowledge Indexer")
    print(f"{'=' * 50}")
    print(f"Source:      {source}")
    print(f"Project:     {project}")
    print(f"Files found: {len(files)}")
    print(f"Mode:        {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'=' * 50}\n")

    for i, fpath in enumerate(files, 1):
        prev_chunks = manifest.total_chunks
        manifest, n_chunks = index_file(
            source_path=fpath,
            project=project,
            manifest=manifest,
            date_str=date_str,
            dry_run=dry_run,
            force=force,
        )

        if n_chunks > 0:
            stats["files_indexed"] += 1
            stats["chunks_written"] += n_chunks
            status = "INDEXED" if not dry_run else "DRY RUN"
            print(f"  [{i}/{len(files)}] {status}: {fpath.name} "
                  f"→ {n_chunks} chunks")
        else:
            stats["files_skipped"] += 1
            print(f"  [{i}/{len(files)}] SKIP (unchanged): {fpath.name}")

    print(f"\n{'=' * 50}")
    print(f"Done. {stats['files_indexed']} files → "
          f"{stats['chunks_written']} chunks")
    if stats["files_skipped"]:
        print(f"Skipped {stats['files_skipped']} unchanged files")
    print(f"Manifest: {manifest.summary()}")
    print(f"{'=' * 50}\n")

    return stats


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index markdown files into ENGRAM .eng knowledge files"
    )
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Path to file or directory to index",
    )
    parser.add_argument(
        "--project", "-p",
        default="engram",
        help="Project namespace (default: engram)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be indexed without writing",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-index all files regardless of content hash",
    )
    parser.add_argument(
        "--incremental", "-i",
        action="store_true",
        default=True,
        help="Skip unchanged files (default: true)",
    )

    args = parser.parse_args()
    source = Path(args.source).resolve()

    if not source.exists():
        print(f"Error: {source} does not exist", file=sys.stderr)
        sys.exit(1)

    stats = index_batch(
        source=source,
        project=args.project,
        incremental=args.incremental,
        dry_run=args.dry_run,
        force=args.force,
    )

    if "error" in stats:
        print(f"Error: {stats['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
