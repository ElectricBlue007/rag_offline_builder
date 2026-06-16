from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.contracts.content import ContentElement
from core.contracts.markdown import MarkdownGenerationResult, MarkdownSection
from chunking.splitters.splitter import ChunkSplitter


class TestChunkerIntegration(unittest.TestCase):
    def test_pipeline_level_chunking_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            path.write_text("placeholder", encoding="utf-8")
            markdown = MarkdownGenerationResult(
                source_path=str(path),
                markdown="# Title\n\nBody sentence one. Body sentence two.",
                sections=[MarkdownSection(section_id="s1", title="Title", level=0, markdown="# Title")],
                content_elements=[ContentElement(element_id="e1", element_type="heading", text="Title", level=0)],
            )

            result = ChunkSplitter().split(str(path), "doc-1", markdown)

            self.assertEqual(result.source_path, str(path))
            self.assertGreaterEqual(len(result.chunks), 1)


if __name__ == "__main__":
    unittest.main()

