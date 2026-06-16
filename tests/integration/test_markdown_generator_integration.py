import unittest
from core.contracts.stream import DocumentStream, OrderedDocumentBlock
from core.contracts.document import ImageArtifact
from markdown.generators.generator import MarkdownGenerator

class TestMarkdownGeneratorIntegration(unittest.TestCase):
    def test_pipeline_level_markdown_boundary(self):
        # Simulate a realistic stream coming from the StreamMerger
        blocks = [
            OrderedDocumentBlock(block_id="b1", block_type="heading", text="Introduction", level=1, order_index=0),
            OrderedDocumentBlock(block_id="b2", block_type="paragraph", text="Welcome to the document.", order_index=1),
            OrderedDocumentBlock(
                block_id="b3", 
                block_type="image", 
                image=ImageArtifact(id="chart.png", description="A bar chart showing growth."), 
                order_index=2
            ),
            OrderedDocumentBlock(block_id="b4", block_type="paragraph", text="As seen in the chart above.", order_index=3),
        ]
        stream = DocumentStream(source_path="integration_test.docx", blocks=blocks)
        
        result = MarkdownGenerator().generate(stream)
        
        self.assertIn("# Introduction",result.markdown)
        self.assertIn("As seen in the chart above.", result.markdown)
        
        # Verify image is inline, appearing BEFORE the paragraph that references it
        chart_pos = result.markdown.find("chart.png")
        seen_pos = result.markdown.find("As seen in the chart above.")
        self.assertLess(chart_pos, seen_pos, "Image should appear inline before the referencing text")

if __name__ == "__main__":
    unittest.main()