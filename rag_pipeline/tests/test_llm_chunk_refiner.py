"""
tests/test_llm_chunk_refiner.py

Unit tests for LlmChunkRefiner — covering the disabled-by-default no-op
path, section grouping, the merge/no-merge decision via a mocked Ollama
client (no live service required), strict response validation and its
fallback behavior, batching for large sections, and the oversized-merge
safety net.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_pipeline.config import LlmChunkRefinerConfig
from rag_pipeline.splitters.llm_chunk_refiner import LlmChunkRefiner
from rag_pipeline.splitters.semantic_chunker import SemanticChunk


def make_chunk(
    content: str,
    breadcrumb: str = "### Test Section",
    section_type: str = "text",
    source_path: str = "test.md",
) -> SemanticChunk:
    return SemanticChunk(
        content=content,
        section_type=section_type,
        breadcrumb=breadcrumb,
        header_levels=[3],
        header_texts=["Test Section"],
        source_path=source_path,
        normalized_text=content,
        contextualized_text=content,
        embed_text=content,
    )


def mock_ollama_client(reply_json: str):
    """A mock Ollama client whose .chat() always returns the given JSON
    string as the assistant message content."""
    client = MagicMock()
    client.chat.return_value = {"message": {"content": reply_json}}
    return client


class TestDisabledIsNoOp(unittest.TestCase):

    def test_disabled_by_default(self):
        cfg = LlmChunkRefinerConfig()
        self.assertFalse(cfg.enabled)

    def test_disabled_returns_chunks_unchanged(self):
        refiner = LlmChunkRefiner(LlmChunkRefinerConfig(enabled=False))
        chunks = [make_chunk("a"), make_chunk("b")]
        result = refiner.refine(chunks)
        self.assertIs(result, chunks)

    def test_empty_list_returns_empty(self):
        refiner = LlmChunkRefiner(LlmChunkRefinerConfig(enabled=True))
        self.assertEqual(refiner.refine([]), [])

    def test_ollama_never_called_when_disabled(self):
        refiner = LlmChunkRefiner(LlmChunkRefinerConfig(enabled=False))
        refiner._ollama_client = MagicMock()
        refiner.refine([make_chunk("a"), make_chunk("b")])
        refiner._ollama_client.chat.assert_not_called()


class TestGroupingBySection(unittest.TestCase):

    def test_groups_by_breadcrumb(self):
        chunks = [
            make_chunk("a", breadcrumb="### Sec 1"),
            make_chunk("b", breadcrumb="### Sec 1"),
            make_chunk("c", breadcrumb="### Sec 2"),
        ]
        groups = LlmChunkRefiner._group_by_section(chunks)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 2)
        self.assertEqual(len(groups[1]), 1)

    def test_groups_by_section_type_even_with_same_breadcrumb(self):
        chunks = [
            make_chunk("a", breadcrumb="### Sec 1", section_type="text"),
            make_chunk("b", breadcrumb="### Sec 1", section_type="table"),
        ]
        groups = LlmChunkRefiner._group_by_section(chunks)
        self.assertEqual(len(groups), 2)

    def test_non_contiguous_same_breadcrumb_treated_separately(self):
        """Two separate sections that happen to share a breadcrumb string
        (e.g. a repeated header) but aren't adjacent must NOT be merged
        into one group — grouping is purely about contiguous runs."""
        chunks = [
            make_chunk("a", breadcrumb="### Repeated"),
            make_chunk("b", breadcrumb="### Other"),
            make_chunk("c", breadcrumb="### Repeated"),
        ]
        groups = LlmChunkRefiner._group_by_section(chunks)
        self.assertEqual(len(groups), 3)


class TestTablesAndImagesNeverTouched(unittest.TestCase):

    def test_table_chunks_skip_llm_even_when_enabled(self):
        cfg = LlmChunkRefinerConfig(enabled=True, min_chunks_to_consider=1)
        refiner = LlmChunkRefiner(cfg)
        refiner._ollama_client = MagicMock()
        chunks = [
            make_chunk("<table>...</table>", section_type="table"),
            make_chunk("<table>...</table>", section_type="table"),
        ]
        result = refiner.refine(chunks)
        self.assertEqual(result, chunks)
        refiner._ollama_client.chat.assert_not_called()

    def test_image_description_chunks_skip_llm(self):
        cfg = LlmChunkRefinerConfig(enabled=True, min_chunks_to_consider=1)
        refiner = LlmChunkRefiner(cfg)
        refiner._ollama_client = MagicMock()
        chunks = [make_chunk("An image of a dog.", section_type="image_description")]
        result = refiner.refine(chunks)
        self.assertEqual(result, chunks)
        refiner._ollama_client.chat.assert_not_called()


class TestMinChunksToConsider(unittest.TestCase):

    def test_single_chunk_section_skips_llm(self):
        cfg = LlmChunkRefinerConfig(enabled=True, min_chunks_to_consider=2)
        refiner = LlmChunkRefiner(cfg)
        refiner._ollama_client = MagicMock()
        result = refiner.refine([make_chunk("only one")])
        self.assertEqual(len(result), 1)
        refiner._ollama_client.chat.assert_not_called()


class TestMergeApplication(unittest.TestCase):

    def test_merge_concatenates_original_content_verbatim(self):
        """The core safety property: merged content is the EXACT original
        text joined, never anything the LLM wrote itself."""
        cfg = LlmChunkRefinerConfig(enabled=True, min_chunks_to_consider=2)
        refiner = LlmChunkRefiner(cfg)
        refiner._ollama_client = mock_ollama_client('{"merge_groups": [[0, 1]]}')

        chunks = [
            make_chunk("Step 1: open the panel."),
            make_chunk("Step 2: press the button."),
        ]
        result = refiner.refine(chunks)

        self.assertEqual(len(result), 1)
        self.assertIn("Step 1: open the panel.", result[0].content)
        self.assertIn("Step 2: press the button.", result[0].content)

    def test_no_merge_when_all_groups_are_singletons(self):
        cfg = LlmChunkRefinerConfig(enabled=True, min_chunks_to_consider=2)
        refiner = LlmChunkRefiner(cfg)
        refiner._ollama_client = mock_ollama_client('{"merge_groups": [[0], [1]]}')

        chunks = [make_chunk("Complete thought one."), make_chunk("Complete thought two.")]
        result = refiner.refine(chunks)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].content, "Complete thought one.")
        self.assertEqual(result[1].content, "Complete thought two.")

    def test_three_way_merge(self):
        cfg = LlmChunkRefinerConfig(enabled=True, min_chunks_to_consider=2)
        refiner = LlmChunkRefiner(cfg)
        refiner._ollama_client = mock_ollama_client('{"merge_groups": [[0, 1, 2]]}')

        chunks = [make_chunk("A."), make_chunk("B."), make_chunk("C.")]
        result = refiner.refine(chunks)

        self.assertEqual(len(result), 1)
        for piece in ("A.", "B.", "C."):
            self.assertIn(piece, result[0].content)

    def test_merged_chunk_inherits_metadata_from_first_piece(self):
        cfg = LlmChunkRefinerConfig(enabled=True, min_chunks_to_consider=2)
        refiner = LlmChunkRefiner(cfg)
        refiner._ollama_client = mock_ollama_client('{"merge_groups": [[0, 1]]}')

        chunks = [
            make_chunk("A.", breadcrumb="### My Section"),
            make_chunk("B.", breadcrumb="### My Section"),
        ]
        result = refiner.refine(chunks)
        self.assertEqual(result[0].breadcrumb, "### My Section")
        self.assertEqual(result[0].section_type, "text")

    def test_llm_response_text_never_appears_in_output(self):
        """Even if the model's raw reply contains extra prose around the
        JSON, none of that prose should leak into chunk content."""
        cfg = LlmChunkRefinerConfig(enabled=True, min_chunks_to_consider=2)
        refiner = LlmChunkRefiner(cfg)
        refiner._ollama_client = mock_ollama_client(
            'Sure, here is my analysis: {"merge_groups": [[0, 1]]} Hope that helps!'
        )
        chunks = [make_chunk("Original A."), make_chunk("Original B.")]
        result = refiner.refine(chunks)
        self.assertNotIn("Sure, here is my analysis", result[0].content)
        self.assertNotIn("Hope that helps", result[0].content)


class TestResponseValidationFallback(unittest.TestCase):
    """Malformed or untrustworthy LLM output must fall back to the
    original (unmerged) chunks, never crash and never be trusted blindly."""

    def _refiner_with_reply(self, reply_json: str) -> LlmChunkRefiner:
        cfg = LlmChunkRefinerConfig(
            enabled=True, min_chunks_to_consider=2, warn_on_fallback=False
        )
        refiner = LlmChunkRefiner(cfg)
        refiner._ollama_client = mock_ollama_client(reply_json)
        return refiner

    def test_no_json_in_response_falls_back(self):
        refiner = self._refiner_with_reply("I don't know how to help with that.")
        chunks = [make_chunk("A."), make_chunk("B.")]
        result = refiner.refine(chunks)
        self.assertEqual([c.content for c in result], ["A.", "B."])

    def test_missing_index_falls_back(self):
        # Only covers index 0, index 1 is missing entirely
        refiner = self._refiner_with_reply('{"merge_groups": [[0]]}')
        chunks = [make_chunk("A."), make_chunk("B.")]
        result = refiner.refine(chunks)
        self.assertEqual([c.content for c in result], ["A.", "B."])

    def test_duplicate_index_falls_back(self):
        refiner = self._refiner_with_reply('{"merge_groups": [[0, 1], [1]]}')
        chunks = [make_chunk("A."), make_chunk("B.")]
        result = refiner.refine(chunks)
        self.assertEqual([c.content for c in result], ["A.", "B."])

    def test_out_of_range_index_falls_back(self):
        refiner = self._refiner_with_reply('{"merge_groups": [[0, 1, 5]]}')
        chunks = [make_chunk("A."), make_chunk("B.")]
        result = refiner.refine(chunks)
        self.assertEqual([c.content for c in result], ["A.", "B."])

    def test_non_adjacent_group_falls_back(self):
        """Prompt explicitly requires contiguous adjacency — [0, 2] skips
        index 1 and must be rejected, not silently reordered."""
        refiner = self._refiner_with_reply('{"merge_groups": [[0, 2], [1]]}')
        chunks = [make_chunk("A."), make_chunk("B."), make_chunk("C.")]
        result = refiner.refine(chunks)
        self.assertEqual([c.content for c in result], ["A.", "B.", "C."])

    def test_empty_group_falls_back(self):
        refiner = self._refiner_with_reply('{"merge_groups": [[], [0, 1]]}')
        chunks = [make_chunk("A."), make_chunk("B.")]
        result = refiner.refine(chunks)
        self.assertEqual([c.content for c in result], ["A.", "B."])

    def test_malformed_json_falls_back(self):
        refiner = self._refiner_with_reply('{"merge_groups": [[0, 1]')  # truncated
        chunks = [make_chunk("A."), make_chunk("B.")]
        result = refiner.refine(chunks)
        self.assertEqual([c.content for c in result], ["A.", "B."])

    def test_fallback_count_incremented(self):
        refiner = self._refiner_with_reply("not json at all")
        refiner.refine([make_chunk("A."), make_chunk("B.")])
        self.assertEqual(refiner._fallback_count, 1)

    def test_ollama_exception_falls_back(self):
        cfg = LlmChunkRefinerConfig(
            enabled=True, min_chunks_to_consider=2, warn_on_fallback=False
        )
        refiner = LlmChunkRefiner(cfg)
        refiner._ollama_client = MagicMock()
        refiner._ollama_client.chat.side_effect = ConnectionError("Ollama not running")
        chunks = [make_chunk("A."), make_chunk("B.")]
        result = refiner.refine(chunks)
        self.assertEqual([c.content for c in result], ["A.", "B."])

    def test_fallback_disabled_raises(self):
        cfg = LlmChunkRefinerConfig(
            enabled=True, min_chunks_to_consider=2, fallback_on_error=False
        )
        refiner = LlmChunkRefiner(cfg)
        refiner._ollama_client = mock_ollama_client("garbage")
        with self.assertRaises(RuntimeError):
            refiner.refine([make_chunk("A."), make_chunk("B.")])

    def test_warn_on_fallback_prints(self):
        cfg = LlmChunkRefinerConfig(
            enabled=True, min_chunks_to_consider=2, warn_on_fallback=True
        )
        refiner = LlmChunkRefiner(cfg)
        refiner._ollama_client = mock_ollama_client("garbage")
        with patch("builtins.print") as mock_print:
            refiner.refine([make_chunk("A."), make_chunk("B.")])
        mock_print.assert_called_once()
        self.assertIn("WARNING", mock_print.call_args[0][0])


class TestBatching(unittest.TestCase):

    def test_large_section_split_into_multiple_calls(self):
        cfg = LlmChunkRefinerConfig(
            enabled=True, min_chunks_to_consider=2, max_chunks_per_call=2
        )
        refiner = LlmChunkRefiner(cfg)
        # Every batch reply says "keep everything separate" for simplicity
        refiner._ollama_client = mock_ollama_client('{"merge_groups": [[0], [1]]}')

        chunks = [make_chunk(f"chunk {i}") for i in range(6)]  # 6 chunks, batches of 2 → 3 calls
        result = refiner.refine(chunks)

        self.assertEqual(refiner._ollama_client.chat.call_count, 3)
        self.assertEqual(len(result), 6)

    def test_merges_do_not_cross_batch_boundary(self):
        cfg = LlmChunkRefinerConfig(
            enabled=True, min_chunks_to_consider=2, max_chunks_per_call=2
        )
        refiner = LlmChunkRefiner(cfg)
        # Always tries to merge everything in whatever batch it sees
        refiner._ollama_client = MagicMock()
        refiner._ollama_client.chat.side_effect = [
            {"message": {"content": '{"merge_groups": [[0, 1]]}'}},
            {"message": {"content": '{"merge_groups": [[0, 1]]}'}},
            {"message": {"content": '{"merge_groups": [[0, 1]]}'}},
        ]

        chunks = [make_chunk(f"chunk {i}") for i in range(6)]
        result = refiner.refine(chunks)

        # 3 batches of 2, each fully merged → 3 resulting chunks total
        self.assertEqual(len(result), 3)


class TestOversizedMergeSafetyNet(unittest.TestCase):

    def test_merge_exceeding_char_limit_is_resplit(self):
        cfg = LlmChunkRefinerConfig(enabled=True, min_chunks_to_consider=2)
        refiner = LlmChunkRefiner(cfg)
        refiner._ollama_client = mock_ollama_client('{"merge_groups": [[0, 1]]}')

        # Two large chunks whose combined length exceeds the 4000-char safety net
        big_a = "Sentence A. " * 300   # ~3600 chars
        big_b = "Sentence B. " * 300   # ~3600 chars
        chunks = [make_chunk(big_a), make_chunk(big_b)]
        result = refiner.refine(chunks)

        self.assertGreater(len(result), 1)  # got re-split, not left as one giant blob
        # No content lost across the pieces
        combined = " ".join(c.content for c in result)
        self.assertIn("Sentence A.", combined)
        self.assertIn("Sentence B.", combined)

    def test_normal_sized_merge_not_split(self):
        cfg = LlmChunkRefinerConfig(enabled=True, min_chunks_to_consider=2)
        refiner = LlmChunkRefiner(cfg)
        refiner._ollama_client = mock_ollama_client('{"merge_groups": [[0, 1]]}')
        chunks = [make_chunk("Short A."), make_chunk("Short B.")]
        result = refiner.refine(chunks)
        self.assertEqual(len(result), 1)


class TestResponseTextExtraction(unittest.TestCase):
    """Mirrors the same dict-vs-attribute defensive parsing tested for
    LlmClient and ContextualEnricher — ollama-python's response shape has
    varied across versions."""

    def test_dict_style_response(self):
        response = {"message": {"content": "hello"}}
        self.assertEqual(LlmChunkRefiner._extract_response_text(response), "hello")

    def test_attribute_style_response(self):
        class FakeMessage:
            content = "hello"
        class FakeResponse:
            message = FakeMessage()
        self.assertEqual(LlmChunkRefiner._extract_response_text(FakeResponse()), "hello")


class TestImportErrorWhenOllamaMissing(unittest.TestCase):

    def test_import_error_surfaces_clearly(self):
        cfg = LlmChunkRefinerConfig(
            enabled=True, min_chunks_to_consider=2, fallback_on_error=False
        )
        refiner = LlmChunkRefiner(cfg)
        with patch.dict(sys.modules, {"ollama": None}):
            with self.assertRaises(RuntimeError):
                refiner.refine([make_chunk("A."), make_chunk("B.")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
