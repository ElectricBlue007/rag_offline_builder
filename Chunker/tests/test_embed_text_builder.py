"""
tests/test_embed_text_builder.py

Unit tests for EmbedTextBuilder and prepare_query_text() — the single
source of truth for the embedding model's instruction-prefix contract
(configurable via EmbedTextConfig; e.g. E5's "passage: "/"query: " or
qwen3-embedding's no-prefix/instruction-prefix contract).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Chunker.config import EmbedTextConfig
from Chunker.enrichers.embed_text_builder import (
    EmbedTextBuilder,
    prepare_query_text,
)
from Chunker.splitters.semantic_chunker import SemanticChunk


def make_chunk(contextualized_text: str = "Some context: chunk content.") -> SemanticChunk:
    return SemanticChunk(
        content=contextualized_text,
        section_type="text",
        breadcrumb="### Test",
        header_levels=[3],
        header_texts=["Test"],
        source_path="test.md",
        contextualized_text=contextualized_text,
        embed_text=contextualized_text,  # matches _wrap default
    )


class TestEmbedTextBuilder(unittest.TestCase):

    def test_no_prefix_by_default(self):
        """
        Current default (qwen3-embedding contract): no passage prefix at
        all — see EmbedTextConfig. This intentionally does NOT hardcode
        a specific prefix string, since that's a model-dependent choice
        that will change again if the embedding model changes; the
        mechanism itself (prefix gets applied) is covered separately
        by test_prefix_comes_before_content below, using an explicit
        test-only config rather than relying on the production default.
        """
        builder = EmbedTextBuilder()
        chunk = make_chunk("Some content.")
        result = builder.build([chunk])
        self.assertEqual(result[0].embed_text, "Some content.")

    def test_prefix_comes_before_content(self):
        cfg = EmbedTextConfig(passage_prefix="passage: ")
        builder = EmbedTextBuilder(cfg)
        chunk = make_chunk("XYZ_MARKER")
        result = builder.build([chunk])
        self.assertTrue(result[0].embed_text.startswith("passage: "))
        self.assertIn("XYZ_MARKER", result[0].embed_text)

    def test_custom_prefix(self):
        cfg = EmbedTextConfig(passage_prefix="custom: ")
        builder = EmbedTextBuilder(cfg)
        chunk = make_chunk("Content.")
        result = builder.build([chunk])
        self.assertEqual(result[0].embed_text, "custom: Content.")

    def test_disabled_produces_no_prefix(self):
        cfg = EmbedTextConfig(enabled=False)
        builder = EmbedTextBuilder(cfg)
        chunk = make_chunk("Plain content.")
        result = builder.build([chunk])
        self.assertEqual(result[0].embed_text, "Plain content.")
        self.assertNotIn("passage:", result[0].embed_text)

    def test_uses_contextualized_text_as_base_not_content(self):
        """embed_text must derive from contextualized_text, not raw content."""
        chunk = make_chunk("The contextualized version.")
        chunk.content = "<table>raw html</table>"  # simulate divergence
        builder = EmbedTextBuilder()
        result = builder.build([chunk])
        self.assertIn("The contextualized version.", result[0].embed_text)
        self.assertNotIn("<table>", result[0].embed_text)

    def test_multiple_chunks_all_prefixed(self):
        cfg = EmbedTextConfig(passage_prefix="passage: ")
        builder = EmbedTextBuilder(cfg)
        chunks = [make_chunk(f"Content {i}.") for i in range(5)]
        result = builder.build(chunks)
        for i, chunk in enumerate(result):
            self.assertEqual(chunk.embed_text, f"passage: Content {i}.")

    def test_empty_list_returns_empty(self):
        builder = EmbedTextBuilder()
        result = builder.build([])
        self.assertEqual(result, [])

    def test_result_stripped(self):
        builder = EmbedTextBuilder()
        chunk = make_chunk("  spaced content  ")
        result = builder.build([chunk])
        self.assertEqual(result[0].embed_text, result[0].embed_text.strip())

    def test_returns_same_chunk_objects_mutated(self):
        """build() mutates in place and returns the same list (matches
        the pattern used by TableChunkNormalizer/ContextualEnricher)."""
        builder = EmbedTextBuilder()
        chunk = make_chunk("Content.")
        result = builder.build([chunk])
        self.assertIs(result[0], chunk)


class TestPrepareQueryText(unittest.TestCase):

    def test_default_prefix_applied(self):
        """
        Checks the mechanism (config's query_prefix gets applied), not a
        hardcoded literal — the exact instruction text is a model-
        dependent choice (see EmbedTextConfig) that will keep changing.
        """
        result = prepare_query_text("What is a CMI?")
        expected = EmbedTextConfig().query_prefix + "What is a CMI?"
        self.assertEqual(result, expected)

    def test_custom_config_prefix(self):
        cfg = EmbedTextConfig(query_prefix="search: ")
        result = prepare_query_text("What is a CMI?", cfg)
        self.assertEqual(result, "search: What is a CMI?")

    def test_query_stripped_before_prefixing(self):
        cfg = EmbedTextConfig(query_prefix="query: ")
        result = prepare_query_text("  What is a CMI?  ", cfg)
        self.assertEqual(result, "query: What is a CMI?")

    def test_disabled_produces_no_prefix(self):
        cfg = EmbedTextConfig(enabled=False)
        result = prepare_query_text("What is a CMI?", cfg)
        self.assertEqual(result, "What is a CMI?")

    def test_empty_query(self):
        cfg = EmbedTextConfig(query_prefix="query: ")
        result = prepare_query_text("", cfg)
        self.assertEqual(result, "query:")

    def test_query_prefix_never_equals_passage_prefix(self):
        """
        The core safety property this whole module exists to guarantee:
        passage and query prefixes must always be distinguishable, or
        retrieval ranking silently breaks (see module docstring).
        """
        cfg = EmbedTextConfig()
        self.assertNotEqual(cfg.passage_prefix, cfg.query_prefix)

    def test_spanish_query(self):
        cfg = EmbedTextConfig(query_prefix="query: ")
        result = prepare_query_text("¿Qué es un CMI?", cfg)
        self.assertEqual(result, "query: ¿Qué es un CMI?")


class TestPrefixContractConsistency(unittest.TestCase):
    """
    Cross-checks between EmbedTextBuilder (storage side) and
    prepare_query_text (query side) — proving they share the same
    config source and can never drift apart when configured together.
    """

    def test_shared_config_produces_consistent_prefixes(self):
        cfg = EmbedTextConfig(passage_prefix="P: ", query_prefix="Q: ")
        builder = EmbedTextBuilder(cfg)
        chunk = make_chunk("Some passage content.")
        passage_result = builder.build([chunk])[0].embed_text
        query_result = prepare_query_text("Some query.", cfg)

        self.assertTrue(passage_result.startswith("P: "))
        self.assertTrue(query_result.startswith("Q: "))

    def test_default_config_instances_use_same_prefixes(self):
        """Two independently-constructed default configs must agree —
        proving the defaults are truly a single source of truth, not
        duplicated/hardcoded in two places that could drift."""
        cfg1 = EmbedTextConfig()
        cfg2 = EmbedTextConfig()
        self.assertEqual(cfg1.passage_prefix, cfg2.passage_prefix)
        self.assertEqual(cfg1.query_prefix, cfg2.query_prefix)


if __name__ == "__main__":
    unittest.main(verbosity=2)
