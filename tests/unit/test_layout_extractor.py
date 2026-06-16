import unittest
import tempfile
import zipfile
from pathlib import Path

from layout.reading_order.extractor import ReadingOrderExtractor


class TestLayoutExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = ReadingOrderExtractor()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.docx_path = Path(self.temp_dir.name) / "test.docx"
        self._create_mock_docx()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mock_docx(self):
        # Create a minimal valid DOCX with:
        # 1. A heading (style="Heading1")
        # 2. A paragraph (style="Normal", ends with a period to pass heuristic check)
        # 3. A table
        document_xml = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:body>
                <w:p>
                    <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
                    <w:r><w:t>My Heading</w:t></w:r>
                </w:p>
                <w:p>
                    <w:pPr><w:pStyle w:val="Normal"/></w:pPr>
                    <w:r><w:t>This is a normal paragraph with a period.</w:t></w:r>
                </w:p>
                <w:tbl>
                    <w:tr><w:tc><w:p><w:r><w:t>Table Cell</w:t></w:r></w:p></w:tc></w:tr>
                </w:tbl>
            </w:body>
        </w:document>"""
        
        with zipfile.ZipFile(self.docx_path, 'w') as zf:
            zf.writestr("word/document.xml", document_xml)
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"></Types>')

    def test_extract_reading_order_from_docx(self):
        result = self.extractor.extract(str(self.docx_path))
        
        self.assertEqual(len(result.nodes), 3)
        
        # The new StyleClassifier correctly identifies:
        # 1. "My Heading" -> heading (due to explicit style)
        # 2. "This is a normal paragraph with a period." -> paragraph (due to terminal punctuation)
        # 3. Table -> table
        self.assertEqual([node.node_type for node in result.nodes], ["heading", "paragraph", "table"])

    def test_extractor_health_and_debug(self):
        self.assertEqual(self.extractor.health_check()["status"], "ok")
        self.assertIn("extractor", self.extractor.debug())


if __name__ == "__main__":
    unittest.main()