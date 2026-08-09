"""
tests/test_contextual_enricher.py

Unit tests for ContextualEnricher — covering all three modes
(deterministic, llm, off), the LLM fallback chain (mocked Ollama, no
live service required), and the "always fresh from normalized_text"
invariant that prevents stale raw-HTML leaking through for table chunks.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_pipeline.config import ContextualEnricherConfig
from rag_pipeline.enrichers.contextual_enricher import ContextualEnricher
from rag_pipeline.splitters.semantic_chunker import SemanticChunk


def make_chunk(
    normalized_text: str = "Some chunk content.",
    section_type: str = "text",
    breadcrumb: str = "### Test Section",
    content: str = None,
) -> SemanticChunk:
    return SemanticChunk(
        content=content if content is not None else normalized_text,
        section_type=section_type,
        breadcrumb=breadcrumb,
        header_levels=[3],
        header_texts=["Test Section"],
        source_path="test.md",
        normalized_text=normalized_text,
        contextualized_text=normalized_text,  # matches _wrap default
    )


class TestModeValidation(unittest.TestCase):

    def test_valid_modes_accepted(self):
        for mode in ["deterministic", "llm", "off"]:
            ContextualEnricher(ContextualEnricherConfig(mode=mode))  # must not raise

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            ContextualEnricher(ContextualEnricherConfig(mode="bogus"))

    def test_default_mode_is_deterministic(self):
        cfg = ContextualEnricherConfig()
        self.assertEqual(cfg.mode, "deterministic")


class TestOffMode(unittest.TestCase):

    def setUp(self):
        self.enricher = ContextualEnricher(ContextualEnricherConfig(mode="off"))

    def test_contextualized_text_equals_normalized_text(self):
        chunk = make_chunk(normalized_text="Plain content here.")
        result = self.enricher.enrich_chunks([chunk], "doc text", "file.md")
        self.assertEqual(result[0].contextualized_text, "Plain content here.")

    def test_no_prefix_added(self):
        chunk = make_chunk(normalized_text="Content.", breadcrumb="### Some Section")
        result = self.enricher.enrich_chunks([chunk], "doc text", "file.md")
        self.assertNotIn("Some Section", result[0].contextualized_text)
        self.assertNotIn("From", result[0].contextualized_text)

    def test_stale_content_never_leaks_through_for_tables(self):
        """
        The key invariant: even in 'off' mode, contextualized_text must be
        freshly derived from normalized_text (not left at its _wrap-time
        default), so a table chunk whose normalized_text was overwritten
        by TableChunkNormalizer doesn't leak stale raw HTML.
        """
        chunk = make_chunk(
            content="<table><tr><td>raw html</td></tr></table>",
            normalized_text="Create SOI: clean prose version.",  # simulates
            section_type="table",                                 # post-normalization
        )
        # Manually simulate the staleness scenario: contextualized_text
        # still holds the OLD value from _wrap() time (raw HTML), while
        # normalized_text has since been updated by TableChunkNormalizer.
        chunk.contextualized_text = "<table><tr><td>raw html</td></tr></table>"

        result = self.enricher.enrich_chunks([chunk], "doc text", "file.md")
        self.assertEqual(result[0].contextualized_text, "Create SOI: clean prose version.")
        self.assertNotIn("<table", result[0].contextualized_text)


class TestDeterministicMode(unittest.TestCase):

    def setUp(self):
        self.enricher = ContextualEnricher(ContextualEnricherConfig(mode="deterministic"))

    def test_filename_included(self):
        chunk = make_chunk(normalized_text="Content.", breadcrumb="(root)")
        result = self.enricher.enrich_chunks([chunk], "doc text", "Binder2.md")
        self.assertIn("Binder2.md", result[0].contextualized_text)

    def test_breadcrumb_included(self):
        chunk = make_chunk(normalized_text="Content.", breadcrumb="### System reports")
        result = self.enricher.enrich_chunks([chunk], "doc text", "file.md")
        self.assertIn("System reports", result[0].contextualized_text)

    def test_hashes_stripped_from_breadcrumb(self):
        chunk = make_chunk(normalized_text="Content.", breadcrumb="### System reports")
        result = self.enricher.enrich_chunks([chunk], "doc text", "file.md")
        self.assertNotIn("###", result[0].contextualized_text)

    def test_multi_level_breadcrumb_hashes_all_stripped(self):
        chunk = make_chunk(
            normalized_text="Content.",
            breadcrumb="# Prompt Engineering > ## Introduction",
        )
        result = self.enricher.enrich_chunks([chunk], "doc text", "file.md")
        self.assertNotIn("#", result[0].contextualized_text.split(":")[0])
        self.assertIn("Prompt Engineering", result[0].contextualized_text)
        self.assertIn("Introduction", result[0].contextualized_text)

    def test_root_breadcrumb_excluded(self):
        """breadcrumb=='(root)' means no header context — must not appear literally."""
        chunk = make_chunk(normalized_text="Content.", breadcrumb="(root)")
        result = self.enricher.enrich_chunks([chunk], "doc text", "file.md")
        self.assertNotIn("(root)", result[0].contextualized_text)

    def test_original_content_preserved_after_prefix(self):
        chunk = make_chunk(normalized_text="The actual chunk content.", breadcrumb="### X")
        result = self.enricher.enrich_chunks([chunk], "doc text", "file.md")
        self.assertIn("The actual chunk content.", result[0].contextualized_text)

    def test_prefix_comes_before_content(self):
        chunk = make_chunk(normalized_text="ACTUALCONTENT", breadcrumb="### MySection")
        result = self.enricher.enrich_chunks([chunk], "doc text", "file.md")
        text = result[0].contextualized_text
        self.assertLess(text.index("MySection"), text.index("ACTUALCONTENT"))

    def test_include_breadcrumb_false(self):
        cfg = ContextualEnricherConfig(mode="deterministic", include_breadcrumb=False)
        enricher = ContextualEnricher(cfg)
        chunk = make_chunk(normalized_text="Content.", breadcrumb="### System reports")
        result = enricher.enrich_chunks([chunk], "doc text", "file.md")
        self.assertNotIn("System reports", result[0].contextualized_text)

    def test_include_source_filename_false(self):
        cfg = ContextualEnricherConfig(mode="deterministic", include_source_filename=False)
        enricher = ContextualEnricher(cfg)
        chunk = make_chunk(normalized_text="Content.", breadcrumb="### System reports")
        result = enricher.enrich_chunks([chunk], "doc text", "Binder2.md")
        self.assertNotIn("Binder2.md", result[0].contextualized_text)

    def test_both_disabled_produces_no_prefix(self):
        cfg = ContextualEnricherConfig(
            mode="deterministic", include_breadcrumb=False, include_source_filename=False
        )
        enricher = ContextualEnricher(cfg)
        chunk = make_chunk(normalized_text="Just content.", breadcrumb="### Section")
        result = enricher.enrich_chunks([chunk], "doc text", "file.md")
        self.assertEqual(result[0].contextualized_text, "Just content.")

    def test_applies_to_table_chunks_too(self):
        chunk = make_chunk(
            normalized_text="Create SOI: logical entry is CMI track.",
            section_type="table",
            breadcrumb="### Operator actions",
        )
        result = enricher_result = self.enricher.enrich_chunks(
            [chunk], "doc text", "Binder2.md"
        )
        self.assertIn("Binder2.md", result[0].contextualized_text)
        self.assertIn("Create SOI", result[0].contextualized_text)

    def test_applies_to_image_description_chunks_too(self):
        chunk = make_chunk(
            normalized_text="A screenshot of the interface.",
            section_type="image_description",
            breadcrumb="### MMI Disposition",
        )
        result = self.enricher.enrich_chunks([chunk], "doc text", "file.md")
        self.assertIn("MMI Disposition", result[0].contextualized_text)

    def test_multiple_chunks_processed_independently(self):
        chunks = [
            make_chunk(normalized_text="First.", breadcrumb="### A"),
            make_chunk(normalized_text="Second.", breadcrumb="### B"),
        ]
        result = self.enricher.enrich_chunks(chunks, "doc text", "file.md")
        self.assertIn("A", result[0].contextualized_text)
        self.assertIn("B", result[1].contextualized_text)
        self.assertNotIn("First.", result[1].contextualized_text)


class TestLlmMode(unittest.TestCase):
    """Tests using a mocked Ollama client — no live service required."""

    def _make_enricher(self, **cfg_kwargs) -> ContextualEnricher:
        cfg = ContextualEnricherConfig(mode="llm", warn_on_fallback=False, **cfg_kwargs)
        return ContextualEnricher(cfg)

    def test_llm_response_used_as_prefix(self):
        enricher = self._make_enricher()
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "This passage describes the CMI creation action."}
        }
        enricher._ollama_client = mock_client

        chunk = make_chunk(normalized_text="Create SOI: enter NT and Vel.MAC.")
        result = enricher.enrich_chunks([chunk], "Full document text here.", "file.md")

        self.assertIn("This passage describes the CMI creation action.", result[0].contextualized_text)
        self.assertIn("Create SOI", result[0].contextualized_text)

    def test_chat_called_with_correct_model(self):
        enricher = self._make_enricher(llm_model="llama3.2")
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "Context."}}
        enricher._ollama_client = mock_client

        chunk = make_chunk()
        enricher.enrich_chunks([chunk], "doc", "file.md")

        call_kwargs = mock_client.chat.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "llama3.2")

    def test_document_excerpt_truncated_to_config_limit(self):
        enricher = self._make_enricher(max_document_chars_for_llm=50)
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "Context."}}
        enricher._ollama_client = mock_client

        long_doc = "A" * 5000
        chunk = make_chunk()
        enricher.enrich_chunks([chunk], long_doc, "file.md")

        prompt = mock_client.chat.call_args.kwargs["messages"][0]["content"]
        # The excerpt embedded in the prompt should be capped at 50 chars
        # of the original 5000-char document.
        self.assertNotIn("A" * 100, prompt)

    def test_prompt_includes_chunk_text(self):
        enricher = self._make_enricher()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "Context."}}
        enricher._ollama_client = mock_client

        chunk = make_chunk(normalized_text="UNIQUE_CHUNK_MARKER_XYZ")
        enricher.enrich_chunks([chunk], "doc", "file.md")

        prompt = mock_client.chat.call_args.kwargs["messages"][0]["content"]
        self.assertIn("UNIQUE_CHUNK_MARKER_XYZ", prompt)

    def test_response_object_style_extraction(self):
        """Some ollama-python versions return attribute-style objects."""
        enricher = self._make_enricher()

        class FakeMessage:
            content = "Attribute-style context."

        class FakeResponse:
            message = FakeMessage()

        mock_client = MagicMock()
        mock_client.chat.return_value = FakeResponse()
        enricher._ollama_client = mock_client

        chunk = make_chunk(normalized_text="Content.")
        result = enricher.enrich_chunks([chunk], "doc", "file.md")
        self.assertIn("Attribute-style context.", result[0].contextualized_text)

    def test_import_error_falls_back_to_deterministic(self):
        """If the ollama package isn't installed, must fall back gracefully."""
        enricher = self._make_enricher()
        # Force the lazy import inside _call_ollama to fail
        chunk = make_chunk(normalized_text="Content.", breadcrumb="### Fallback Section")

        with patch.dict(sys.modules, {"ollama": None}):
            result = enricher.enrich_chunks([chunk], "doc", "Binder2.md")

        # Should fall back to deterministic prefix, not crash
        self.assertIn("Fallback Section", result[0].contextualized_text)
        self.assertIn("Content.", result[0].contextualized_text)

    def test_ollama_exception_falls_back_to_deterministic(self):
        enricher = self._make_enricher()
        mock_client = MagicMock()
        mock_client.chat.side_effect = ConnectionError("Ollama not running")
        enricher._ollama_client = mock_client

        chunk = make_chunk(normalized_text="Content.", breadcrumb="### Fallback Section")
        result = enricher.enrich_chunks([chunk], "doc", "file.md")

        self.assertIn("Fallback Section", result[0].contextualized_text)

    def test_empty_llm_response_falls_back(self):
        enricher = self._make_enricher()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "   "}}
        enricher._ollama_client = mock_client

        chunk = make_chunk(normalized_text="Content.", breadcrumb="### Fallback Section")
        result = enricher.enrich_chunks([chunk], "doc", "file.md")

        self.assertIn("Fallback Section", result[0].contextualized_text)

    def test_fallback_disabled_raises(self):
        enricher = self._make_enricher(fallback_to_deterministic_on_error=False)
        mock_client = MagicMock()
        mock_client.chat.side_effect = ConnectionError("Ollama not running")
        enricher._ollama_client = mock_client

        chunk = make_chunk(normalized_text="Content.")
        with self.assertRaises(RuntimeError):
            enricher.enrich_chunks([chunk], "doc", "file.md")

    def test_fallback_count_tracked(self):
        enricher = self._make_enricher()
        mock_client = MagicMock()
        mock_client.chat.side_effect = ConnectionError("down")
        enricher._ollama_client = mock_client

        chunks = [make_chunk(normalized_text=f"Content {i}.") for i in range(3)]
        enricher.enrich_chunks(chunks, "doc", "file.md")
        self.assertEqual(enricher._llm_fallback_count, 3)

    def test_warn_on_fallback_prints(self):
        cfg = ContextualEnricherConfig(mode="llm", warn_on_fallback=True)
        enricher = ContextualEnricher(cfg)
        mock_client = MagicMock()
        mock_client.chat.side_effect = ConnectionError("down")
        enricher._ollama_client = mock_client

        chunk = make_chunk(normalized_text="Content.")
        with patch("builtins.print") as mock_print:
            enricher.enrich_chunks([chunk], "doc", "file.md")
            mock_print.assert_called()

    def test_client_created_lazily_once(self):
        """The Ollama client should only be constructed once, then reused."""
        enricher = self._make_enricher()
        mock_ollama_module = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.chat.return_value = {"message": {"content": "Ctx."}}
        mock_ollama_module.Client.return_value = mock_client_instance

        chunks = [make_chunk(normalized_text=f"Content {i}.") for i in range(3)]
        with patch.dict(sys.modules, {"ollama": mock_ollama_module}):
            enricher.enrich_chunks(chunks, "doc", "file.md")

        mock_ollama_module.Client.assert_called_once()


class TestEnrichChunksGeneral(unittest.TestCase):
    """Tests independent of specific mode behaviour."""

    def test_returns_same_list_length(self):
        enricher = ContextualEnricher(ContextualEnricherConfig(mode="deterministic"))
        chunks = [make_chunk(normalized_text=f"C{i}") for i in range(5)]
        result = enricher.enrich_chunks(chunks, "doc", "file.md")
        self.assertEqual(len(result), 5)

    def test_empty_chunks_list(self):
        enricher = ContextualEnricher(ContextualEnricherConfig(mode="deterministic"))
        result = enricher.enrich_chunks([], "doc", "file.md")
        self.assertEqual(result, [])

    def test_result_is_stripped(self):
        enricher = ContextualEnricher(ContextualEnricherConfig(mode="off"))
        chunk = make_chunk(normalized_text="  spaced content  ")
        result = enricher.enrich_chunks([chunk], "doc", "file.md")
        self.assertEqual(result[0].contextualized_text.strip(), result[0].contextualized_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
