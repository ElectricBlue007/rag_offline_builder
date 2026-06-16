"""Integration tests for RAG-Optimized Markdown Generator end-to-end flow."""
import unittest
from core.contracts.stream import DocumentStream, OrderedDocumentBlock
from core.contracts.tables import TableCellData
from core.contracts.document import ImageArtifact
from markdown.generators.generator import generate_markdown, result_to_json

class TestRAGMarkdownIntegration(unittest.TestCase):
    def test_full_document_stream_to_rag_markdown(self):
        """Tests the complete flow from a complex stream to validated JSON output."""
        blocks = [
            OrderedDocumentBlock(block_id="1", block_type="heading", text="Executive Summary", level=1, order_index=0),
            OrderedDocumentBlock(block_id="2", block_type="paragraph", text="This is a summary with a footnote[^1].", order_index=1),
            OrderedDocumentBlock(block_id="3", block_type="table", table_cells=[
                TableCellData(row_index=0, column_index=0, text="Metric"),
                TableCellData(row_index=0, column_index=1, text="Value"),
                TableCellData(row_index=1, column_index=0, text="Revenue"),
                TableCellData(row_index=1, column_index=1, text="$1M"),
            ], order_index=2),
            OrderedDocumentBlock(block_id="4", block_type="image", image=ImageArtifact(
                id="fig1", alt_text="Growth Chart", description="Upward trend", caption="Fig 1"
            ), order_index=3),
        ]
        
        stream = DocumentStream(
            source_path="/docs/report.docx", 
            blocks=blocks,
            warnings=["Minor formatting issue in source"]
        )
        
        metadata = {"title": "Q1 Report", "author": "Finance Dept"}
        
        # Execute
        result = generate_markdown(stream, metadata)
        json_output = result_to_json(result)
        
        # Validate Contract
        result.validate()
        
        # Assert Integration Points
        self.assertTrue(result.markdown.startswith("---"))
        self.assertIn("title: Q1 Report", result.markdown)
        self.assertIn("# Executive Summary", result.markdown)
        self.assertIn("[^1]", result.markdown) # Footnote reference preserved
        self.assertIn("| Metric | Value |", result.markdown)
        self.assertIn("![Growth Chart](fig1)", result.markdown)
        self.assertIn("*Visual Context: Upward trend*", result.markdown)
        self.assertIn("Minor formatting issue in source", result.warnings)
        
        # Assert JSON serialization works without errors
        self.assertIsInstance(json_output, str)
        self.assertIn('"source_path": "/docs/report.docx"', json_output)