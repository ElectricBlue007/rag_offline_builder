import unittest
from core.contracts.stream import OrderedDocumentBlock, DocumentStream
from core.contracts.document import ImageArtifact
from core.exceptions.base import ContractValidationError

class TestStreamContracts(unittest.TestCase):
    def test_valid_paragraph_block(self):
        block = OrderedDocumentBlock(block_id="b1", block_type="paragraph", text="Hello", order_index=0)
        block.validate()

    def test_valid_image_block(self):
        img = ImageArtifact(id="img1")
        block = OrderedDocumentBlock(block_id="b2", block_type="image", image=img, order_index=1)
        block.validate()

    def test_invalid_block_missing_payload(self):
        block = OrderedDocumentBlock(block_id="b3", block_type="paragraph", text="", order_index=0)
        with self.assertRaises(ContractValidationError):
            block.validate()

    def test_valid_stream(self):
        block = OrderedDocumentBlock(block_id="b1", block_type="paragraph", text="Hello", order_index=0)
        stream = DocumentStream(source_path="test.docx", blocks=[block])
        stream.validate()

if __name__ == "__main__":
    unittest.main()