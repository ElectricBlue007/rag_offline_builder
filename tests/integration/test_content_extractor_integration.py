from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from extractors.content.extractor import ContentExtractor


class TestContentExtractorIntegration(unittest.TestCase):
    def test_cli_level_content_pipeline_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.docx"
            document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
      <w:r><w:t>Chapter One</w:t></w:r>
    </w:p>
  </w:body>
</w:document>
"""
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", document_xml)

            result = ContentExtractor().extract(str(path))

            self.assertEqual(result.source_path, str(path))
            self.assertEqual(result.elements[0].text, "Chapter One")


if __name__ == "__main__":
    unittest.main()

