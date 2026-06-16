import unittest
from core.contracts.stream import DocumentStream, OrderedDocumentBlock
from core.contracts.tables import TableCellData
from chunking.splitters.stream_aware_splitter import StreamAwareChunker

class TestStreamAwareChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = StreamAwareChunker()

    def test_table_is_never_split(self):
        blocks = [
            OrderedDocumentBlock(block_id="b1", block_type="heading", text="Section 1", level=1, order_index=0),
            OrderedDocumentBlock(block_id="b2", block_type="paragraph", text="Intro text.", order_index=1),
            OrderedDocumentBlock(
                block_id="b3", 
                block_type="table", 
                table_cells=[
                    TableCellData(row_index=0, column_index=0, text="Header"),
                    TableCellData(row_index=1, column_index=0, text="Row 1"),
                ],
                order_index=2
            ),
        ]
        stream = DocumentStream(source_path="test.docx", blocks=blocks)
        result = self.chunker.chunk_stream(stream, "doc-123")

        self.assertEqual(len(result.chunks), 2)
        # Chunk 1: Intro text
        self.assertFalse(result.chunks[0].metadata["contains_table"])
        # Chunk 2: Table (intact)
        self.assertTrue(result.chunks[1].metadata["contains_table"])
        self.assertIn("Header", result.chunks[1].text)
        self.assertIn("Row 1", result.chunks[1].text)
        self.assertEqual(result.chunks[1].metadata["section_path"], ["Section 1"])

if __name__ == "__main__":
    unittest.main()