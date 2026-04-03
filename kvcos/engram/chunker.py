"""
kvcos/engram/chunker.py — Markdown-aware semantic chunker.

Splits markdown files into chunks suitable for .eng indexing.
Each chunk gets its own fingerprint and becomes independently
retrievable via HNSW.

Strategy:
  1. Split on H1/H2 headers first (natural semantic boundaries)
  2. If a section exceeds max_chars, split on H3/H4
  3. If still too large, split on paragraph boundaries
  4. Never break mid-paragraph (preserve semantic coherence)

Each chunk carries context: the file's title + parent headers
are prepended so the fingerprint captures the full meaning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Chunk:
    """One semantic chunk from a markdown file."""
    text: str              # Chunk content (with context header prepended)
    raw_text: str          # Original content without context header
    char_start: int        # Start offset in original file
    char_end: int          # End offset in original file
    index: int             # 0-based chunk index
    headers: tuple[str, ...]  # Header hierarchy (e.g., ("# Title", "## Section"))

    @property
    def char_count(self) -> int:
        return len(self.text)


# Regex for markdown headers (ATX style: # through ######)
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _header_level(line: str) -> int:
    """Return header level (1-6) or 0 if not a header."""
    m = re.match(r"^(#{1,6})\s+", line)
    return len(m.group(1)) if m else 0


def _split_by_headers(
    content: str,
    max_level: int = 2,
) -> list[tuple[int, int, list[str]]]:
    """
    Split content into sections by header level.

    Returns list of (start_offset, end_offset, header_stack) tuples.
    max_level: split on headers of this level or lower (1=H1, 2=H2, etc.)
    """
    sections: list[tuple[int, int, list[str]]] = []
    header_stack: list[str] = []
    current_start = 0

    for m in _HEADER_RE.finditer(content):
        level = len(m.group(1))
        header_text = m.group(0).strip()

        if level <= max_level and m.start() > current_start:
            # Close previous section
            section_text = content[current_start:m.start()].strip()
            if section_text:
                sections.append((
                    current_start,
                    m.start(),
                    list(header_stack),
                ))
            current_start = m.start()

        # Update header stack
        if level <= max_level:
            # Trim stack to parent level and push current
            header_stack = [
                h for h in header_stack
                if _header_level(h) < level
            ]
            header_stack.append(header_text)

    # Final section
    if current_start < len(content):
        final_text = content[current_start:].strip()
        if final_text:
            sections.append((
                current_start,
                len(content),
                list(header_stack),
            ))

    return sections


def _split_paragraphs(
    text: str,
    max_chars: int,
    base_offset: int = 0,
) -> list[tuple[int, int]]:
    """
    Split text into chunks at paragraph boundaries.

    Returns list of (start_offset, end_offset) tuples relative
    to the original file (offset by base_offset).
    """
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[tuple[int, int]] = []
    current_start = 0
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # +2 for the \n\n separator

        if current_len + para_len > max_chars and current_len > 0:
            # Close current chunk
            chunks.append((
                base_offset + current_start,
                base_offset + current_start + current_len,
            ))
            current_start = current_start + current_len
            current_len = 0

        current_len += para_len

    # Final chunk
    if current_len > 0:
        chunks.append((
            base_offset + current_start,
            base_offset + current_start + current_len,
        ))

    return chunks


def chunk_markdown(
    content: str,
    max_chars: int = 2000,
    min_chars: int = 100,
    context_prefix: str = "",
) -> list[Chunk]:
    """
    Split a markdown file into semantic chunks.

    Args:
        content: Full markdown file content.
        max_chars: Target maximum chars per chunk (soft limit).
        min_chars: Minimum chars — smaller sections merge with next.
        context_prefix: Prepended to each chunk for context
                       (e.g., "Source: geodesic3.md | Project: engram").

    Returns:
        List of Chunk objects, ordered by position in file.
    """
    if not content.strip():
        return []

    # If the whole file fits in one chunk, return it directly
    if len(content) <= max_chars:
        full_text = f"{context_prefix}\n\n{content}" if context_prefix else content
        return [Chunk(
            text=full_text,
            raw_text=content,
            char_start=0,
            char_end=len(content),
            index=0,
            headers=(),
        )]

    # Phase 1: Split on H1/H2 boundaries
    sections = _split_by_headers(content, max_level=2)

    # If no headers found, treat as single block
    if not sections:
        sections = [(0, len(content), [])]

    # Phase 2: Sub-split large sections on H3/H4
    refined: list[tuple[int, int, list[str]]] = []
    for start, end, headers in sections:
        section_text = content[start:end]
        if len(section_text) > max_chars:
            subsections = _split_by_headers(section_text, max_level=4)
            if len(subsections) > 1:
                for sub_start, sub_end, sub_headers in subsections:
                    refined.append((
                        start + sub_start,
                        start + sub_end,
                        headers + sub_headers,
                    ))
            else:
                refined.append((start, end, headers))
        else:
            refined.append((start, end, headers))

    # Phase 3: Paragraph-split anything still over max_chars
    final_ranges: list[tuple[int, int, list[str]]] = []
    for start, end, headers in refined:
        section_text = content[start:end]
        if len(section_text) > max_chars:
            para_ranges = _split_paragraphs(section_text, max_chars, start)
            for p_start, p_end in para_ranges:
                final_ranges.append((p_start, p_end, headers))
        else:
            final_ranges.append((start, end, headers))

    # Phase 4: Greedily pack sections into chunks up to max_chars.
    # Keep merging consecutive sections while their combined size
    # stays under max_chars. This prevents over-fragmentation of
    # files with many small header sections.
    merged: list[tuple[int, int, list[str]]] = []
    for start, end, headers in final_ranges:
        chunk_text = content[start:end].strip()
        if not chunk_text:
            continue

        if merged:
            prev_start, prev_end, prev_headers = merged[-1]
            prev_len = prev_end - prev_start
            curr_len = end - start

            # Merge if combined chunk stays under max_chars
            if (prev_len + curr_len) <= max_chars:
                merged[-1] = (prev_start, end, prev_headers)
                continue

        merged.append((start, end, headers))

    # Phase 5: Build Chunk objects with context
    chunks: list[Chunk] = []
    for idx, (start, end, headers) in enumerate(merged):
        raw = content[start:end].strip()
        if not raw:
            continue

        # Build context header
        parts = []
        if context_prefix:
            parts.append(context_prefix)
        if headers:
            parts.append(" > ".join(headers))

        prefix = "\n".join(parts)
        text = f"{prefix}\n\n{raw}" if prefix else raw

        chunks.append(Chunk(
            text=text,
            raw_text=raw,
            char_start=start,
            char_end=end,
            index=idx,
            headers=tuple(headers),
        ))

    # Re-index after merging
    return [
        Chunk(
            text=c.text,
            raw_text=c.raw_text,
            char_start=c.char_start,
            char_end=c.char_end,
            index=i,
            headers=c.headers,
        )
        for i, c in enumerate(chunks)
    ]


def slug_from_path(path: str) -> str:
    """
    Generate a kebab-case slug from a file path.

    Examples:
        "geodesic3.md" → "geodesic3"
        "EIGENGRAM_SPEC.md" → "eigengram-spec"
        "coding-style.md" → "coding-style"
    """
    name = path.rsplit("/", 1)[-1]  # filename only
    name = name.rsplit(".", 1)[0]   # strip extension
    # Convert underscores and spaces to hyphens, lowercase
    slug = re.sub(r"[_\s]+", "-", name).lower()
    # Strip non-alphanumeric except hyphens
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    # Collapse multiple hyphens
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def eng_filename(
    project: str,
    slug: str,
    date: str,
    chunk_index: int | None = None,
    chunk_total: int | None = None,
    time_str: str = "",
) -> str:
    """
    Generate .eng filename following the naming convention.

    Format: <slug>[_<chunk>]_<date>[_<time>].eng

    Args:
        project: Project namespace (used for directory, not filename)
        slug: Kebab-case file identifier
        date: ISO date string (YYYY-MM-DD)
        chunk_index: 0-based chunk index (None if single chunk)
        chunk_total: Total chunks (None if single chunk)
        time_str: Optional HHmm time string

    Returns:
        Filename (not full path) like "geodesic3_001_2026-04-02.eng"
    """
    parts = [slug]

    if chunk_index is not None and chunk_total is not None and chunk_total > 1:
        parts.append(f"{chunk_index + 1:03d}")

    parts.append(date)

    if time_str:
        parts.append(time_str)

    return "_".join(parts) + ".eng"
