"""Tests for kvcos.engram.chunker — markdown-aware semantic chunker."""

import pytest

from kvcos.engram.chunker import Chunk, chunk_markdown, eng_filename, slug_from_path


class TestChunkMarkdown:
    def test_empty_content(self):
        assert chunk_markdown("") == []
        assert chunk_markdown("   ") == []

    def test_small_file_single_chunk(self):
        content = "# Title\n\nSome short content."
        chunks = chunk_markdown(content, max_chars=2000)
        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == len(content)

    def test_large_file_splits(self):
        # Create content that exceeds max_chars
        content = "# Section 1\n\n" + "A" * 1500 + "\n\n# Section 2\n\n" + "B" * 1500
        chunks = chunk_markdown(content, max_chars=2000)
        assert len(chunks) >= 2

    def test_chunks_cover_full_content(self):
        content = "# A\n\nText A.\n\n# B\n\nText B.\n\n# C\n\nText C."
        chunks = chunk_markdown(content, max_chars=15)
        # All original content should be present across chunks
        combined = " ".join(c.raw_text for c in chunks)
        for word in ["Text A", "Text B", "Text C"]:
            assert word in combined

    def test_context_prefix(self):
        content = "Hello world"
        chunks = chunk_markdown(content, context_prefix="Source: test.md")
        assert len(chunks) == 1
        assert chunks[0].text.startswith("Source: test.md")

    def test_indices_sequential(self):
        content = "# A\n\n" + "X" * 3000 + "\n\n# B\n\n" + "Y" * 3000
        chunks = chunk_markdown(content, max_chars=2000)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_merge_small_sections(self):
        """Small consecutive sections should merge into one chunk."""
        content = "# A\n\nShort.\n\n# B\n\nAlso short.\n\n# C\n\nStill short."
        chunks = chunk_markdown(content, max_chars=2000, min_chars=100)
        # All three small sections should merge into 1 chunk
        assert len(chunks) == 1

    def test_paragraph_split_fallback(self):
        """Content without headers should split on paragraphs."""
        paragraphs = ["Paragraph " + str(i) + ". " + "X" * 500
                       for i in range(6)]
        content = "\n\n".join(paragraphs)
        chunks = chunk_markdown(content, max_chars=1500)
        assert len(chunks) >= 2


class TestSlugFromPath:
    def test_simple_filename(self):
        assert slug_from_path("readme.md") == "readme"

    def test_uppercase_underscores(self):
        assert slug_from_path("EIGENGRAM_SPEC.md") == "eigengram-spec"

    def test_already_kebab(self):
        assert slug_from_path("coding-style.md") == "coding-style"

    def test_full_path(self):
        assert slug_from_path("/Users/test/docs/my_doc.md") == "my-doc"

    def test_special_chars(self):
        assert slug_from_path("file (copy).md") == "file-copy"


class TestEngFilename:
    def test_single_chunk(self):
        name = eng_filename("engram", "readme", "2026-04-02")
        assert name == "readme_2026-04-02.eng"

    def test_multi_chunk(self):
        name = eng_filename("engram", "geodesic3", "2026-04-02",
                           chunk_index=0, chunk_total=5)
        assert name == "geodesic3_001_2026-04-02.eng"

    def test_with_time(self):
        name = eng_filename("engram", "session", "2026-04-02",
                           time_str="1430")
        assert name == "session_2026-04-02_1430.eng"

    def test_single_chunk_no_index(self):
        """Single-chunk files should not have chunk number."""
        name = eng_filename("engram", "small", "2026-04-02",
                           chunk_index=0, chunk_total=1)
        assert name == "small_2026-04-02.eng"
