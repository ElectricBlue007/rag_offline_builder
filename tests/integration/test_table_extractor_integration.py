from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from extractors.tables.extractor import TableExtractor


class TestTableExtractorIntegration(unittest.TestCase):
    def test_cli_level_table_pipeline_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.docx"
            document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Only Cell</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>
"""
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", document_xml)

            result = TableExtractor().extract(str(path))

            self.assertEqual(result.source_path, str(path))
            self.assertEqual(result.tables[0][0].text, "Only Cell")


if __name__ == "__main__":
    unittest.main()

