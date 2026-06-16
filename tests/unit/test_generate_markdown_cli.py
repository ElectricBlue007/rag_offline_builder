from __future__ import annotations

import tempfile
import unittest
import zipfile
from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

from scripts.cli.generate_markdown import main


class TestGenerateMarkdownCli(unittest.TestCase):
    def test_cli_generates_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.docx"
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
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", document_xml)

            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main([str(path)])

            self.assertEqual(exit_code, 0)
            self.assertIn("# Title", stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()

