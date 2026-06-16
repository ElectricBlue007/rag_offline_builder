from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from extractors.tables.extractor import TableExtractor


def build_docx_with_table(path: Path) -> None:
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>H1</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>H2</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>A1</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>A2</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)


class TestTableExtractor(unittest.TestCase):
    def test_extract_tables_from_docx(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.docx"
            build_docx_with_table(path)

            result = TableExtractor().extract(str(path))

            self.assertEqual(len(result.tables), 1)
            self.assertEqual(result.tables[0][0].text, "H1")
            self.assertEqual(result.tables[0][1].text, "H2")

    def test_validate_and_health_check(self) -> None:
        extractor = TableExtractor()
        self.assertEqual(extractor.validate()["status"], "ok")
        self.assertEqual(extractor.health_check()["extractor"], "tables")


if __name__ == "__main__":
    unittest.main()

