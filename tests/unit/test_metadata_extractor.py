from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from extractors.metadata.extractor import MetadataExtractor, extract_metadata


class TestMetadataExtractor(unittest.TestCase):
    def test_extract_metadata_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            path.write_text("hello metadata", encoding="utf-8")

            result = extract_metadata(str(path))

            self.assertEqual(result.metadata.file_name, "sample.txt")
            self.assertEqual(result.extractor_name, "metadata")
            self.assertEqual(result.debug_info["file_size_bytes"], path.stat().st_size)
            result.validate()

    def test_validate_and_health_check(self) -> None:
        extractor = MetadataExtractor()
        self.assertEqual(extractor.validate()["status"], "ok")
        self.assertEqual(extractor.health_check()["extractor"], "metadata")


if __name__ == "__main__":
    unittest.main()

