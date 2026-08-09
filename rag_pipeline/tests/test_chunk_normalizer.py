"""
tests/test_chunk_normalizer.py

Unit tests for normalizers/chunk_normalizer.py — the pipeline-stage
wrapper around table_normalizer.py, including its fallback/error
handling behaviour (which table_normalizer's own tests don't cover
since that module only raises, never catches).
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_pipeline.normalizers.chunk_normalizer import (
    TableChunkNormalizer,
    TableNormalizerConfig,
)
from rag_pipeline.normalizers.table_normalizer import TableNormalizationError
from rag_pipeline.splitters.semantic_chunker import SemanticChunk


def make_chunk(content: str, section_type: str = "table") -> SemanticChunk:
    return SemanticChunk(
        content=content,
        section_type=section_type,
        breadcrumb="### Test",
        header_levels=[3],
        header_texts=["Test"],
        source_path="test.md",
        normalized_text=content,  # matches SemanticChunker._wrap default
    )


VALID_TABLE = (
    "<table border='1'><thead><tr><th>Action</th><th>Entry</th></tr></thead>"
    "<tbody><tr><td>Create SOI</td><td>CMI track</td></tr></tbody></table>"
)

MALFORMED_TABLE = "<div>not a table</div>"


class TestTableChunkNormalizerBasics(unittest.TestCase):

    def setUp(self):
        self.normalizer = TableChunkNormalizer()

    def test_table_chunk_gets_normalized(self):
        chunk = make_chunk(VALID_TABLE, "table")
        result = self.normalizer.normalize_chunks([chunk])
        self.assertNotEqual(result[0].normalized_text, VALID_TABLE)
        self.assertIn("Create SOI", result[0].normalized_text)

    def test_normalized_text_has_no_html(self):
        chunk = make_chunk(VALID_TABLE, "table")
        result = self.normalizer.normalize_chunks([chunk])
        self.assertNotIn("<table", result[0].normalized_text)

    def test_content_unchanged(self):
        chunk = make_chunk(VALID_TABLE, "table")
        result = self.normalizer.normalize_chunks([chunk])
        self.assertEqual(result[0].content, VALID_TABLE)

    def test_text_chunk_untouched(self):
        chunk = make_chunk("Plain text content.", "text")
        result = self.normalizer.normalize_chunks([chunk])
        self.assertEqual(result[0].normalized_text, "Plain text content.")

    def test_image_description_chunk_untouched(self):
        chunk = make_chunk("An image description.", "image_description")
        result = self.normalizer.normalize_chunks([chunk])
        self.assertEqual(result[0].normalized_text, "An image description.")

    def test_returns_same_list_length(self):
        chunks = [
            make_chunk(VALID_TABLE, "table"),
            make_chunk("Text.", "text"),
            make_chunk("Image.", "image_description"),
        ]
        result = self.normalizer.normalize_chunks(chunks)
        self.assertEqual(len(result), 3)

    def test_empty_list_returns_empty(self):
        result = self.normalizer.normalize_chunks([])
        self.assertEqual(result, [])

    def test_mixed_chunk_types_only_tables_changed(self):
        text_chunk = make_chunk("Some text.", "text")
        table_chunk = make_chunk(VALID_TABLE, "table")
        result = self.normalizer.normalize_chunks([text_chunk, table_chunk])
        self.assertEqual(result[0].normalized_text, "Some text.")
        self.assertNotEqual(result[1].normalized_text, VALID_TABLE)


class TestTableChunkNormalizerFallback(unittest.TestCase):

    def test_malformed_table_falls_back_to_raw_content(self):
        cfg = TableNormalizerConfig(fallback_to_raw_on_error=True, warn_on_fallback=False)
        normalizer = TableChunkNormalizer(cfg)
        chunk = make_chunk(MALFORMED_TABLE, "table")
        result = normalizer.normalize_chunks([chunk])
        self.assertEqual(result[0].normalized_text, MALFORMED_TABLE)

    def test_malformed_table_raises_when_fallback_disabled(self):
        cfg = TableNormalizerConfig(fallback_to_raw_on_error=False)
        normalizer = TableChunkNormalizer(cfg)
        chunk = make_chunk(MALFORMED_TABLE, "table")
        with self.assertRaises(TableNormalizationError):
            normalizer.normalize_chunks([chunk])

    def test_fallback_count_tracked(self):
        cfg = TableNormalizerConfig(fallback_to_raw_on_error=True, warn_on_fallback=False)
        normalizer = TableChunkNormalizer(cfg)
        chunk = make_chunk(MALFORMED_TABLE, "table")
        normalizer.normalize_chunks([chunk])
        self.assertEqual(normalizer._fallback_count, 1)

    def test_warn_on_fallback_prints_message(self):
        cfg = TableNormalizerConfig(fallback_to_raw_on_error=True, warn_on_fallback=True)
        normalizer = TableChunkNormalizer(cfg)
        chunk = make_chunk(MALFORMED_TABLE, "table")
        with patch("builtins.print") as mock_print:
            normalizer.normalize_chunks([chunk])
            mock_print.assert_called()

    def test_warn_on_fallback_false_suppresses_message(self):
        cfg = TableNormalizerConfig(fallback_to_raw_on_error=True, warn_on_fallback=False)
        normalizer = TableChunkNormalizer(cfg)
        chunk = make_chunk(MALFORMED_TABLE, "table")
        with patch("builtins.print") as mock_print:
            normalizer.normalize_chunks([chunk])
            mock_print.assert_not_called()

    def test_unexpected_exception_also_falls_back(self):
        """Any exception type — not just TableNormalizationError — must be caught."""
        cfg = TableNormalizerConfig(fallback_to_raw_on_error=True, warn_on_fallback=False)
        normalizer = TableChunkNormalizer(cfg)
        chunk = make_chunk(VALID_TABLE, "table")

        with patch(
            "rag_pipeline.normalizers.chunk_normalizer.normalize_table_html",
            side_effect=RuntimeError("boom"),
        ):
            result = normalizer.normalize_chunks([chunk])
            self.assertEqual(result[0].normalized_text, VALID_TABLE)

    def test_empty_normalization_result_falls_back(self):
        """If normalization succeeds but returns empty string, still fall back."""
        cfg = TableNormalizerConfig(fallback_to_raw_on_error=True, warn_on_fallback=False)
        normalizer = TableChunkNormalizer(cfg)
        chunk = make_chunk(VALID_TABLE, "table")

        with patch(
            "rag_pipeline.normalizers.chunk_normalizer.normalize_table_html",
            return_value="   ",
        ):
            result = normalizer.normalize_chunks([chunk])
            self.assertEqual(result[0].normalized_text, VALID_TABLE)


class TestTableChunkNormalizerDisabled(unittest.TestCase):

    def test_enabled_false_leaves_all_chunks_untouched(self):
        cfg = TableNormalizerConfig(enabled=False)
        normalizer = TableChunkNormalizer(cfg)
        chunk = make_chunk(VALID_TABLE, "table")
        result = normalizer.normalize_chunks([chunk])
        # normalized_text stays == content (the default from SemanticChunker)
        self.assertEqual(result[0].normalized_text, VALID_TABLE)

    def test_enabled_false_skips_processing_entirely(self):
        """When disabled, normalize_table_html should never even be called."""
        cfg = TableNormalizerConfig(enabled=False)
        normalizer = TableChunkNormalizer(cfg)
        chunk = make_chunk(VALID_TABLE, "table")

        with patch(
            "rag_pipeline.normalizers.chunk_normalizer.normalize_table_html"
        ) as mock_normalize:
            normalizer.normalize_chunks([chunk])
            mock_normalize.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
