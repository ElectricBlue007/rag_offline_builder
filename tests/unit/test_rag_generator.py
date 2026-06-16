"""Unit tests for RAG-optimized Markdown generator."""
import unittest
from unittest.mock import MagicMock, patch
from core.contracts.stream import DocumentStream, OrderedDocumentBlock
from core.contracts.markdown import MarkdownGenerationResult
from markdown.generators.generator import RAGMarkdownGenerator

class TestRAGMarkdownGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = RAGMarkdownGenerator()
        self.mock_stream = DocumentStream(
            source_path="/tmp/test.docx",
            blocks=[
                OrderedDocumentBlock(block_id="b1", block_type="heading", text="Introduction", level=1, order_index=0),
                OrderedDocumentBlock(block_id="b2", block_type="paragraph", text="This is a test paragraph.", order_index=1),
            ]
        )
        self.mock_metadata = {"title": "Test Doc", "author": "AI", "source_hash": "abc123"}

    def test_generate_with_frontmatter(self):
        result = self.generator.generate(self.mock_stream, metadata=self.mock_metadata)
        self.assertIsInstance(result, MarkdownGenerationResult)
        self.assertTrue(result.markdown.startswith("---"))
        self.assertIn("title: Test Doc", result.markdown)
        self.assertEqual(len(result.sections), 1)
        result.validate()

    def test_generate_normalizes_headings(self):
        # Simulate a skipped heading level (H1 -> H3)
        bad_stream = DocumentStream(
            source_path="/tmp/test.docx",
            blocks=[
                OrderedDocumentBlock(block_id="b1", block_type="heading", text="H1", level=1, order_index=0),
                OrderedDocumentBlock(block_id="b2", block_type="heading", text="H3", level=3, order_index=1),
            ]
        )
        result = self.generator.generate(bad_stream)
        # The normalizer should cap H3 to H2
        self.assertIn("## H3", result.markdown)

    def test_health_check(self):
        self.assertEqual(self.generator.health_check()["status"], "ok")