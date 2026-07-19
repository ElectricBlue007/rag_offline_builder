"""
tests/test_semantic_chunker.py

Unit tests for SemanticChunker and its helpers.
Uses a mock LangChain chunker — no embedding model download required.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Chunker.config import SemanticChunkerConfig
from Chunker.splitters.semantic_chunker import (
    SemanticChunker,
    SemanticChunk,
    _hard_split,
)
from Chunker.splitters.structure_splitter import StructuralSection


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def make_section(
    content: str,
    section_type: str = "text",
    breadcrumb: str = "### Test",
    header_levels=None,
    header_texts=None,
    source_path: str = "test.md",
) -> StructuralSection:
    return StructuralSection(
        content=content,
        section_type=section_type,
        breadcrumb=breadcrumb,
        header_levels=header_levels or [3],
        header_texts=header_texts or ["Test"],
        source_path=source_path,
    )


def make_mock_lc_chunker(split_texts: list[str]):
    """Return a mock that simulates LangChain's create_documents() output."""
    class MockDoc:
        def __init__(self, text):
            self.page_content = text

    mock = MagicMock()
    mock.create_documents.return_value = [MockDoc(t) for t in split_texts]
    return mock


# ════════════════════════════════════════════════════════════════════
# Tests for _hard_split
# ════════════════════════════════════════════════════════════════════

class TestHardSplit(unittest.TestCase):

    def test_text_within_limit_returned_as_is(self):
        text = "Short text."
        result = _hard_split(text, max_units=100)
        self.assertEqual(result, [text])

    def test_text_exactly_at_limit(self):
        text = "A" * 100
        result = _hard_split(text, max_units=100)
        self.assertEqual(result, [text])

    def test_splits_at_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence."
        result = _hard_split(text, max_units=30)
        self.assertGreater(len(result), 1)
        # Reassembled must equal original (modulo spacing)
        joined = " ".join(result)
        self.assertEqual(joined.replace("  ", " "), text)

    def test_single_long_sentence_kept_whole(self):
        """If one sentence exceeds max_chars, it must not be cut mid-sentence."""
        long_sent = "A" * 200
        result = _hard_split(long_sent, max_units=50)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], long_sent)

    def test_empty_string_returns_single_empty(self):
        result = _hard_split("", max_units=100)
        self.assertEqual(result, [""])

    def test_multiple_sentences_split_correctly(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        result = _hard_split(text, max_units=25)
        for chunk in result:
            # Each chunk must be a complete sentence or combination
            self.assertTrue(len(chunk) > 0)

    def test_exclamation_and_question_marks_as_boundaries(self):
        text = "Is this right? Yes it is! And another sentence follows."
        result = _hard_split(text, max_units=20)
        self.assertGreater(len(result), 1)

    def test_no_content_lost(self):
        text = "Part one. Part two. Part three."
        result = _hard_split(text, max_units=15)
        reassembled = " ".join(result)
        # All words present
        for word in ["Part", "one", "two", "three"]:
            self.assertIn(word, reassembled)


# ════════════════════════════════════════════════════════════════════
# Tests for SemanticChunker (using mock LangChain)
# ════════════════════════════════════════════════════════════════════

class TestSemanticChunkerAtomicSections(unittest.TestCase):
    """Tables and image_description sections must always be single atomic chunks."""

    def setUp(self):
        self.chunker = SemanticChunker()

    def test_table_section_always_single_chunk(self):
        table_html = (
            "<table border='1'>"
            "<thead><tr><th>Action</th><th>Entry</th></tr></thead>"
            "<tbody><tr><td>Create</td><td>Enter NT</td></tr></tbody>"
            "</table>"
        )
        section = make_section(table_html, section_type="table")
        chunks = self.chunker.chunk_section(section)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, table_html)

    def test_table_never_calls_lc_chunker(self):
        """LangChain must NOT be called for table sections."""
        section = make_section("<table><tr><td>Data</td></tr></table>", section_type="table")
        mock_lc = make_mock_lc_chunker(["irrelevant"])
        self.chunker._lc_chunker = mock_lc
        self.chunker.chunk_section(section)
        mock_lc.create_documents.assert_not_called()

    def test_image_description_always_single_chunk(self):
        alt = "Screenshot of the MAC Great Circle menu interface."
        section = make_section(alt, section_type="image_description")
        chunks = self.chunker.chunk_section(section)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, alt)

    def test_image_description_never_calls_lc_chunker(self):
        section = make_section("An image alt text.", section_type="image_description")
        mock_lc = make_mock_lc_chunker(["irrelevant"])
        self.chunker._lc_chunker = mock_lc
        self.chunker.chunk_section(section)
        mock_lc.create_documents.assert_not_called()

    def test_table_chunk_type_preserved(self):
        section = make_section("<table><tr><td>X</td></tr></table>", section_type="table")
        chunks = self.chunker.chunk_section(section)
        self.assertEqual(chunks[0].section_type, "table")

    def test_image_chunk_type_preserved(self):
        section = make_section("Alt text content.", section_type="image_description")
        chunks = self.chunker.chunk_section(section)
        self.assertEqual(chunks[0].section_type, "image_description")


class TestSemanticChunkerSmallSections(unittest.TestCase):
    """Text sections below min_chunk_chars skip semantic splitting."""

    def test_small_text_returns_single_chunk_no_lc_call(self):
        cfg = SemanticChunkerConfig(min_chunk_chars=200, skip_semantic_if_small=True, use_token_based_sizing=False)
        chunker = SemanticChunker(cfg)
        mock_lc = make_mock_lc_chunker(["irrelevant"])
        chunker._lc_chunker = mock_lc

        section = make_section("Short text.", section_type="text")
        chunks = chunker.chunk_section(section)

        mock_lc.create_documents.assert_not_called()
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, "Short text.")

    def test_small_text_flag_false_calls_lc(self):
        """When skip_semantic_if_small=False, even small text goes through LangChain."""
        cfg = SemanticChunkerConfig(min_chunk_chars=200, skip_semantic_if_small=False, use_token_based_sizing=False)
        chunker = SemanticChunker(cfg)
        mock_lc = make_mock_lc_chunker(["Short text."])
        chunker._lc_chunker = mock_lc

        section = make_section("Short text.", section_type="text")
        chunker.chunk_section(section)

        mock_lc.create_documents.assert_called_once()

    def test_empty_content_returns_empty_list(self):
        cfg = SemanticChunkerConfig(min_chunk_chars=5, skip_semantic_if_small=True, use_token_based_sizing=False)
        chunker = SemanticChunker(cfg)
        section = make_section("", section_type="text")
        chunks = chunker.chunk_section(section)
        self.assertEqual(chunks, [])

    def test_whitespace_only_content_returns_empty(self):
        cfg = SemanticChunkerConfig(min_chunk_chars=5, skip_semantic_if_small=True, use_token_based_sizing=False)
        chunker = SemanticChunker(cfg)
        section = make_section("   \n\n  ", section_type="text")
        chunks = chunker.chunk_section(section)
        self.assertEqual(chunks, [])


class TestSemanticChunkerWithMockLangChain(unittest.TestCase):
    """Tests that inject a mock LangChain chunker to test the full flow."""

    def _make_chunker(self, lc_return_texts: list[str], **cfg_kwargs) -> SemanticChunker:
        cfg = SemanticChunkerConfig(
            min_chunk_chars=10,
            max_chunk_chars=2000,
            skip_semantic_if_small=False,
            use_token_based_sizing=False,
            **cfg_kwargs
        )
        chunker = SemanticChunker(cfg)
        chunker._lc_chunker = make_mock_lc_chunker(lc_return_texts)
        return chunker

    def test_lc_returns_one_chunk(self):
        text = "A complete paragraph about prompt engineering and LLMs."
        chunker = self._make_chunker([text])
        section = make_section(text)
        chunks = chunker.chunk_section(section)
        self.assertEqual(len(chunks), 1)
        self.assertIn("prompt engineering", chunks[0].content)

    def test_lc_returns_multiple_chunks(self):
        part1 = "First semantic unit about operators and CMI tracks."
        part2 = "Second semantic unit about system reports and parameters."
        chunker = self._make_chunker([part1, part2])
        section = make_section(part1 + " " + part2)
        chunks = chunker.chunk_section(section)
        self.assertEqual(len(chunks), 2)

    def test_metadata_inherited_from_section(self):
        text = "Some content about the system service."
        chunker = self._make_chunker([text])
        section = make_section(
            text,
            breadcrumb="# Root > ### Section",
            header_levels=[1, 3],
            header_texts=["Root", "Section"],
            source_path="outputs/MyFile/MyFile.md",
        )
        chunks = chunker.chunk_section(section)
        self.assertEqual(chunks[0].breadcrumb, "# Root > ### Section")
        self.assertEqual(chunks[0].header_levels, [1, 3])
        self.assertEqual(chunks[0].header_texts, ["Root", "Section"])
        self.assertEqual(chunks[0].source_path, "outputs/MyFile/MyFile.md")
        self.assertEqual(chunks[0].section_type, "text")

    def test_fallback_to_hard_split_on_lc_error(self):
        """If LangChain raises, must fall back to hard split gracefully."""
        cfg = SemanticChunkerConfig(min_chunk_chars=5, max_chunk_chars=50,
                                    skip_semantic_if_small=False,
                                    use_token_based_sizing=False)
        chunker = SemanticChunker(cfg)
        mock_lc = MagicMock()
        mock_lc.create_documents.side_effect = RuntimeError("model error")
        chunker._lc_chunker = mock_lc

        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        section = make_section(text)
        # Should NOT raise — must fall back to hard split
        chunks = chunker.chunk_section(section)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, SemanticChunk)

    def test_chunk_content_stripped(self):
        """Whitespace around chunk content must be stripped."""
        chunker = self._make_chunker(["  Content with spaces  "])
        section = make_section("Content with spaces")
        chunks = chunker.chunk_section(section)
        self.assertEqual(chunks[0].content, "Content with spaces")


class TestChunkSections(unittest.TestCase):
    """Tests for chunk_sections() — the batch method."""

    def setUp(self):
        cfg = SemanticChunkerConfig(min_chunk_chars=5, skip_semantic_if_small=True, use_token_based_sizing=False)
        self.chunker = SemanticChunker(cfg)

    def test_chunk_index_sequential(self):
        sections = [
            make_section("Text one.", section_type="text"),
            make_section("<table><tr><td>T</td></tr></table>", section_type="table"),
            make_section("Alt text.", section_type="image_description"),
        ]
        chunks = self.chunker.chunk_sections(sections)
        indices = [c.chunk_index for c in chunks]
        self.assertEqual(indices, list(range(len(chunks))))

    def test_chunk_index_starts_at_zero(self):
        sections = [make_section("A sufficiently long text content.", section_type="text")]
        chunks = self.chunker.chunk_sections(sections)
        self.assertEqual(chunks[0].chunk_index, 0)

    def test_empty_sections_list(self):
        chunks = self.chunker.chunk_sections([])
        self.assertEqual(chunks, [])

    def test_mixed_section_types_all_processed(self):
        sections = [
            make_section("<table><tr><td>Row</td></tr></table>", section_type="table"),
            make_section("An image alt text description here.", section_type="image_description"),
            make_section("Normal text content that is long enough.", section_type="text"),
        ]
        chunks = self.chunker.chunk_sections(sections)
        types = [c.section_type for c in chunks]
        self.assertIn("table", types)
        self.assertIn("image_description", types)
        self.assertIn("text", types)


class TestEnforceSizeLimits(unittest.TestCase):
    """Tests for _enforce_size_limits() internal method."""

    def setUp(self):
        cfg = SemanticChunkerConfig(min_chunk_chars=20, max_chunk_chars=100, use_token_based_sizing=False)
        self.chunker = SemanticChunker(cfg)

    def test_normal_chunks_returned_unchanged(self):
        texts = [
            "This chunk is exactly the right size.",
            "This is another properly sized chunk here."
        ]
        result = self.chunker._enforce_size_limits(texts)
        self.assertEqual(len(result), 2)

    def test_oversized_chunk_is_split(self):
        long_text = "Sentence one. " * 20  # ~280 chars, over limit of 100
        result = self.chunker._enforce_size_limits([long_text])
        self.assertGreater(len(result), 1)
        for chunk in result:
            self.assertLessEqual(len(chunk), 150)  # allow some tolerance for long sentences

    def test_small_chunks_merged(self):
        texts = ["Hi", "there", "A long enough chunk that meets the minimum size requirement."]
        result = self.chunker._enforce_size_limits(texts)
        # "Hi" and "there" should be merged into the next chunk or each other
        self.assertLess(len(result), len(texts))

    def test_all_small_chunks_merged_together(self):
        texts = ["Hi", "there", "world"]  # all below min_chunk_chars=20
        result = self.chunker._enforce_size_limits(texts)
        self.assertEqual(len(result), 1)
        self.assertIn("Hi", result[0])
        self.assertIn("there", result[0])
        self.assertIn("world", result[0])

    def test_empty_texts_produce_no_output(self):
        texts = ["", "  ", "\n"]
        # All-empty inputs: after stripping these become empty
        # _enforce_size_limits doesn't strip, but chunk_section filters empty afterwards
        # Test that it doesn't crash
        result = self.chunker._enforce_size_limits(texts)
        self.assertIsInstance(result, list)


class TestSemanticChunkDataclass(unittest.TestCase):
    """Tests for the SemanticChunk dataclass itself."""

    def test_chunk_default_index(self):
        chunk = SemanticChunk(
            content="Content",
            section_type="text",
            breadcrumb="### H",
            header_levels=[3],
            header_texts=["H"],
            source_path="f.md",
        )
        self.assertEqual(chunk.chunk_index, 0)

    def test_chunk_fields_accessible(self):
        chunk = SemanticChunk(
            content="My content",
            section_type="table",
            breadcrumb="# Root > ## Sub",
            header_levels=[1, 2],
            header_texts=["Root", "Sub"],
            source_path="path/file.md",
            chunk_index=42,
        )
        self.assertEqual(chunk.content, "My content")
        self.assertEqual(chunk.section_type, "table")
        self.assertEqual(chunk.chunk_index, 42)
        self.assertEqual(chunk.header_texts, ["Root", "Sub"])


class TestTokenBasedSizing(unittest.TestCase):
    """
    Tests for the NEW default sizing mode (use_token_based_sizing=True).
    `transformers` isn't installed in this environment, so these tests
    exercise TokenCounter's heuristic fallback path — deterministic and
    fully offline, exactly like a real fully-offline deployment.
    """

    def test_default_config_uses_token_based_sizing(self):
        cfg = SemanticChunkerConfig()
        self.assertTrue(cfg.use_token_based_sizing)

    def test_count_fn_is_token_counter_by_default(self):
        """
        Verifies _count_fn delegates to TokenCounter by default (as
        opposed to len()). Forces attempt_real_tokenizer=False so the
        assertion is checking against the heuristic's specific math
        deterministically — without this, an environment that happens
        to have `transformers` installed (common as a transitive
        dependency of sentence-transformers, langchain, etc.) would
        silently use the real tokenizer instead and produce a different,
        equally-valid-but-different count, failing this assertion for
        reasons unrelated to what the test is actually checking.
        """
        cfg = SemanticChunkerConfig(attempt_real_tokenizer=False)
        chunker = SemanticChunker(cfg)
        # heuristic: word_count * 1.4, rounded
        result = chunker._count_fn("one two three four five")
        self.assertEqual(result, round(5 * 1.4))

    def test_count_fn_is_len_when_disabled(self):
        cfg = SemanticChunkerConfig(use_token_based_sizing=False)
        chunker = SemanticChunker(cfg)
        self.assertIs(chunker._count_fn, len)

    def test_min_units_resolves_to_token_config(self):
        cfg = SemanticChunkerConfig(min_chunk_tokens=42, min_chunk_chars=999)
        chunker = SemanticChunker(cfg)
        self.assertEqual(chunker._min_units, 42)

    def test_min_units_resolves_to_char_config_when_disabled(self):
        cfg = SemanticChunkerConfig(
            use_token_based_sizing=False, min_chunk_tokens=42, min_chunk_chars=999
        )
        chunker = SemanticChunker(cfg)
        self.assertEqual(chunker._min_units, 999)

    def test_max_units_resolves_to_token_config(self):
        cfg = SemanticChunkerConfig(max_chunk_tokens=123, max_chunk_chars=9999)
        chunker = SemanticChunker(cfg)
        self.assertEqual(chunker._max_units, 123)

    def test_small_text_skips_semantic_split_using_tokens(self):
        """A short section should skip splitting based on TOKEN count, not chars."""
        cfg = SemanticChunkerConfig(
            min_chunk_tokens=100,  # high threshold in tokens
            skip_semantic_if_small=True,
        )
        chunker = SemanticChunker(cfg)
        section = make_section("Only a few words here.", section_type="text")
        chunks = chunker.chunk_section(section)
        # Should skip splitting (content is well under 100 tokens) and
        # return as a single chunk without ever calling LangChain.
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, "Only a few words here.")

    def test_oversized_text_hard_split_respects_token_limit(self):
        cfg = SemanticChunkerConfig(
            max_chunk_tokens=10,   # ~7 words at 1.4 tokens/word heuristic
            min_chunk_tokens=1,
            skip_semantic_if_small=False,
        )
        chunker = SemanticChunker(cfg)
        # Force fallback to hard split (no real LangChain chunker available)
        text = (
            "First sentence here. Second sentence follows here. "
            "Third sentence also appears. Fourth sentence ends it."
        )
        section = make_section(text, section_type="text")
        chunks = chunker.chunk_section(section)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            token_count = chunker._count_fn(chunk.content)
            # Allow some slack since hard_split works sentence-by-sentence
            # and a single sentence may exceed the limit if it must be
            # kept whole — but no chunk should be wildly over.
            self.assertLessEqual(token_count, 30)

    def test_token_counter_shared_across_calls(self):
        """The TokenCounter instance should be created once and reused."""
        chunker = SemanticChunker()
        counter1 = chunker._token_counter
        _ = chunker._count_fn
        counter2 = chunker._token_counter
        self.assertIsNotNone(counter2)
        if counter1 is not None:
            self.assertIs(counter1, counter2)

    def test_attempt_real_tokenizer_propagated_to_token_counter(self):
        cfg = SemanticChunkerConfig(attempt_real_tokenizer=False)
        chunker = SemanticChunker(cfg)
        _ = chunker._count_fn  # triggers TokenCounter creation
        self.assertFalse(chunker._token_counter.attempt_real_tokenizer)

    def test_tokenizer_hf_id_propagated_to_token_counter(self):
        cfg = SemanticChunkerConfig(tokenizer_hf_id="BAAI/bge-m3")
        chunker = SemanticChunker(cfg)
        _ = chunker._count_fn
        self.assertEqual(chunker._token_counter.model_name, "BAAI/bge-m3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
