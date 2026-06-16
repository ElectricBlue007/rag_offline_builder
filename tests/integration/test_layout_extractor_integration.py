from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from layout.reading_order.extractor import ReadingOrderExtractor


class TestLayoutExtractorIntegration(unittest.TestCase):
    def test_cli_level_layout_pipeline_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.docx"
            document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:r><w:t>Alpha</w:t></w:r>
    </w:p>
  </w:body>
</w:document>
"""
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", document_xml)

            result = ReadingOrderExtractor().extract(str(path))

            self.assertEqual(result.nodes[0].text, "Alpha")
            self.assertEqual(result.source_path, str(path))


if __name__ == "__main__":
    unittest.main()

