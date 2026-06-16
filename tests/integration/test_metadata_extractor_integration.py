from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from extractors.metadata.extractor import MetadataExtractor
from core.contracts.extraction import SourceDocumentRequest


class TestMetadataExtractorIntegration(unittest.TestCase):
    def test_cli_level_metadata_pipeline_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.pdf"
            path.write_bytes(b"%PDF-1.4 sample")

            result = MetadataExtractor().extract(SourceDocumentRequest(file_path=str(path), source_type="pdf"))

            self.assertEqual(result.request.source_type, "pdf")
            self.assertTrue(result.metadata.source_hash)
            self.assertEqual(result.metadata.file_name, "report.pdf")


if __name__ == "__main__":
    unittest.main()

