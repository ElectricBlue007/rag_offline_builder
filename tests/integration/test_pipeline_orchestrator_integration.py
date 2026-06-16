from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from pipeline.orchestrator import PipelineOrchestrator


class TestPipelineOrchestratorIntegration(unittest.TestCase):
    def test_cli_level_pipeline_boundary(self) -> None:
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
      <w:r><w:t>Body sentence one. Body sentence two.</w:t></w:r>
    </w:p>
  </w:body>
</w:document>
"""
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", document_xml)

            result = PipelineOrchestrator().run(str(path))

            self.assertEqual(result.document_id, result.records[0].document_id)
            self.assertGreaterEqual(len(result.records), 1)


if __name__ == "__main__":
    unittest.main()

