from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from pipeline.orchestrator import PipelineOrchestrator


def build_minimal_docx(path: Path) -> None:
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
      <w:r><w:t>Title</w:t></w:r>
    </w:p>
    <w:p>
      <w:r><w:t>Body sentence one. Body sentence two.</w:t></w:r>
    </w:p>
  </w:body>
</w:document>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)


class TestPipelineOrchestrator(unittest.TestCase):
    def test_run_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.docx"
            build_minimal_docx(path)

            result = PipelineOrchestrator().run(str(path))

            self.assertEqual(result.source_path, str(path))
            self.assertTrue(result.records)
            self.assertTrue(result.chunks)

    def test_validate_and_health_check(self) -> None:
        orchestrator = PipelineOrchestrator()
        self.assertEqual(orchestrator.validate()["status"], "ok")
        self.assertEqual(orchestrator.health_check()["orchestrator"], "pipeline")


if __name__ == "__main__":
    unittest.main()

