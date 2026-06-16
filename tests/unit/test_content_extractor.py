from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from extractors.content.extractor import ContentExtractor


def build_minimal_docx(path: Path) -> None:
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
      <w:r><w:t>Chapter One</w:t></w:r>
    </w:p>
    <w:p>
      <w:r><w:t>Body paragraph.</w:t></w:r>
    </w:p>
  </w:body>
</w:document>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)


class TestContentExtractor(unittest.TestCase):
    def test_extract_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            path.write_text("INTRODUCTION\nFirst paragraph.\n", encoding="utf-8")

            result = ContentExtractor().extract(str(path))

            self.assertEqual(result.document_type, "txt")
            self.assertGreaterEqual(len(result.elements), 2)
            self.assertEqual(result.elements[0].element_type, "heading")

    def test_extract_docx(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.docx"
            build_minimal_docx(path)

            result = ContentExtractor().extract(str(path))

            self.assertEqual(result.document_type, "docx")
            self.assertEqual(result.elements[0].element_type, "heading")
            self.assertEqual(result.elements[1].element_type, "paragraph")

    def test_validate_and_health_check(self) -> None:
        extractor = ContentExtractor()
        self.assertEqual(extractor.validate()["status"], "ok")
        self.assertEqual(extractor.health_check()["extractor"], "content")


if __name__ == "__main__":
    unittest.main()

