from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

from scripts.cli.convert_docx_to_markdown import main


class TestConvertDocxToMarkdown(unittest.TestCase):
    def test_writes_markdown_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "sample.docx"
            document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
      <w:r><w:t>Title</w:t></w:r>
    </w:p>
    <w:p>
      <w:r><w:t>Body text.</w:t></w:r>
    </w:p>
  </w:body>
</w:document>
"""
            with zipfile.ZipFile(input_path, "w") as archive:
                archive.writestr("word/document.xml", document_xml)

            output_path = Path(tmpdir) / "output.md"
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main([str(input_path), str(output_path)])

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())
            self.assertIn("# Title", output_path.read_text(encoding="utf-8"))
            self.assertEqual(stderr.getvalue(), "")
            self.assertIn("output.md",stdout.getvalue())


if __name__ == "__main__":
    unittest.main()

