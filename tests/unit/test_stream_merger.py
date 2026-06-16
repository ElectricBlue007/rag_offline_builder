import unittest
from layout.stream_merger.merger import StreamMerger
from core.contracts.layout import ReadingOrderResult, LayoutNode
from core.contracts.content import ContentExtractionResult, ContentElement
from core.contracts.document import ImageArtifact

class TestStreamMerger(unittest.TestCase):
    def setUp(self):
        self.merger = StreamMerger()

    def test_merge_basic_stream(self):
        ro = ReadingOrderResult(
            source_path="test.docx",
            nodes=[
                LayoutNode(node_id="n1", node_type="paragraph", text="Hello", order_index=0),
                LayoutNode(node_id="n2", node_type="image", order_index=1, metadata={"image_index": 0}),
            ]
        )
        content = ContentExtractionResult(
            source_path="test.docx",
            document_type="docx",
            elements=[ContentElement(element_id="e1", element_type="paragraph", text="Hello")]
        )
        images = [ImageArtifact(id="img1.png", description="A test image")]

        stream = self.merger.merge("test.docx", ro, content, None, images)
        
        self.assertEqual(len(stream.blocks), 2)
        self.assertEqual(stream.blocks[0].block_type, "paragraph")
        self.assertEqual(stream.blocks[1].block_type, "image")
        self.assertEqual(stream.blocks[1].image.id, "img1.png")

if __name__ == "__main__":
    unittest.main()