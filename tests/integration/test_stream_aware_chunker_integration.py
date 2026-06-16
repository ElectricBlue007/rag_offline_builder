import unittest
from core.contracts.stream import DocumentStream, OrderedDocumentBlock
from core.contracts.document import ImageArtifact
from core.contracts.tables import TableCellData
from chunking.splitters.stream_aware_splitter import StreamAwareChunker

class TestStreamAwareChunkerIntegration(unittest.TestCase):
    def test_full_document_hierarchy_and_table_preservation(self):
        blocks = [
            OrderedDocumentBlock(block_id="b1", block_type="heading", text="Main Topic", level=1, order_index=0),
            OrderedDocumentBlock(block_id="b2", block_type="heading", text="Sub Topic", level=2, order_index=1),
            OrderedDocumentBlock(block_id="b3", block_type="paragraph", text="Some context.", order_index=2),
            OrderedDocumentBlock(
                block_id="b4", 
                block_type="table", 
                table_cells=[TableCellData(row_index=0, column_index=0, text="Data")], 
                order_index=3
            ),
            OrderedDocumentBlock(
                block_id="b5", 
                block_type="image", 
                image=ImageArtifact(id="img1", description="A chart"), 
                order_index=4
            ),
        ]
        stream = DocumentStream(source_path="test.docx", blocks=blocks)
        result = StreamAwareChunker().chunk_stream(stream, "doc-456")

        self.assertEqual(len(result.chunks), 3)
        
        # Verify section path inheritance
        self.assertEqual(result.chunks[1].metadata["section_path"], ["Main Topic", "Sub Topic"])
        
        # Verify image description is attached to context
        self.assertTrue(result.chunks[2].metadata["contains_image"])
        self.assertIn("[Image Description: A chart]", result.chunks[2].text)

if __name__ == "__main__":
    unittest.main()