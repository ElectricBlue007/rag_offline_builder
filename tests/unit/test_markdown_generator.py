import unittest
from core.contracts.stream import DocumentStream, OrderedDocumentBlock
from core.contracts.document import ImageArtifact
from core.contracts.tables import TableCellData
from markdown.generators.generator import MarkdownGenerator

class TestMarkdownGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = MarkdownGenerator()

    def test_generate_markdown(self):
        blocks = [
            OrderedDocumentBlock(
                block_id="b1", block_type="heading", text="Test Heading", level=1, order_index=0
            ),
            OrderedDocumentBlock(
                block_id="b2", block_type="paragraph", text="This is a test paragraph.", order_index=1
            ),
            OrderedDocumentBlock(
                block_id="b3", 
                block_type="table", 
                table_cells=[
                    TableCellData(row_index=0, column_index=0, text="Header 1"),
                    TableCellData(row_index=0, column_index=1, text="Header 2"),
                    TableCellData(row_index=1, column_index=0, text="Row 1 Col 1"),
                    TableCellData(row_index=1, column_index=1, text="Row 1 Col 2"),
                ],
                order_index=2
            ),
            OrderedDocumentBlock(
                block_id="b4",
                block_type="image",
                image=ImageArtifact(id="img1.png", description="A test image description"),
                order_index=3
            )
        ]
        stream = DocumentStream(source_path="test.docx", blocks=blocks)
        
        result = self.generator.generate(stream)
        
        self.assertIn("# Test Heading", result.markdown)
        self.assertIn("This is a test paragraph.", result.markdown)
        self.assertIn("Header 1", result.markdown)
        self.assertIn("![img1.png](img1.png)", result.markdown)
        self.assertIn("*Visual Context: A test image description*",result.markdown)

if __name__ == "__main__":
    unittest.main()