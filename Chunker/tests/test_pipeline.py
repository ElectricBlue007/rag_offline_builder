"""
tests/test_pipeline.py

Integration tests for MarkdownChunkingPipeline.
Uses a mock SemanticChunker (no embedding model needed) to test
the full pipeline flow against real Chandra OCR output files.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Chunker.config import PipelineConfig, SemanticChunkerConfig
from Chunker.pipeline import MarkdownChunkingPipeline, _discover_md_files, _load_md
from Chunker.enrichers.metadata_enricher import EnrichedChunk
from Chunker.splitters.semantic_chunker import SemanticChunker
from Chunker.splitters.structure_splitter import StructuralSection
from Chunker.tests.fixtures_path import FIXTURES_DIR


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def make_passthrough_chunker() -> SemanticChunker:
    """
    A SemanticChunker that returns each section as a single chunk
    without loading any embedding model.
    """
    from Chunker.splitters.semantic_chunker import SemanticChunk

    class PassthroughChunker(SemanticChunker):
        def chunk_section(self, section: StructuralSection):
            if not section.content.strip():
                return []
            return [SemanticChunk(
                content=section.content.strip(),
                section_type=section.section_type,
                breadcrumb=section.breadcrumb,
                header_levels=list(section.header_levels),
                header_texts=list(section.header_texts),
                source_path=section.source_path,
            )]

    cfg = SemanticChunkerConfig()
    return PassthroughChunker(cfg)


def make_pipeline(input_dir: Path, output_dir: Path) -> MarkdownChunkingPipeline:
    cfg = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        verbose=False,
    )
    pipeline = MarkdownChunkingPipeline(cfg)
    pipeline.sem_chunker = make_passthrough_chunker()
    return pipeline


# ════════════════════════════════════════════════════════════════════
# Tests for module-level loader helpers
# ════════════════════════════════════════════════════════════════════

class TestDiscoverMdFiles(unittest.TestCase):

    def test_finds_md_files_at_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "file.md").write_text("# A")
            found = _discover_md_files(p, "**/*.md")
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].name, "file.md")

    def test_finds_nested_md_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "sub1").mkdir()
            (p / "sub1" / "deep").mkdir()
            (p / "a.md").write_text("# A")
            (p / "sub1" / "b.md").write_text("# B")
            (p / "sub1" / "deep" / "c.md").write_text("# C")
            found = _discover_md_files(p, "**/*.md")
            self.assertEqual(len(found), 3)

    def test_ignores_json_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "doc.md").write_text("# A")
            (p / "doc_metadata.json").write_text("{}")
            found = _discover_md_files(p, "**/*.md")
            self.assertEqual(len(found), 1)

    def test_empty_directory_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            found = _discover_md_files(Path(tmp), "**/*.md")
            self.assertEqual(found, [])

    def test_returns_sorted_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for name in ["c.md", "a.md", "b.md"]:
                (p / name).write_text("# x")
            found = _discover_md_files(p, "**/*.md")
            names = [f.name for f in found]
            self.assertEqual(names, sorted(names))


class TestLoadMd(unittest.TestCase):

    def test_reads_utf8_content(self):
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w",
                                         encoding="utf-8", delete=False) as f:
            f.write("# Title\n\nContent here.")
            path = Path(f.name)
        text = _load_md(path)
        self.assertEqual(text, "# Title\n\nContent here.")
        path.unlink()

    def test_reads_accented_characters(self):
        """Spanish accented chars from Binder2.md must survive round-trip."""
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w",
                                         encoding="utf-8", delete=False) as f:
            f.write("Definición del botón de conjunto de trabajo.")
            path = Path(f.name)
        text = _load_md(path)
        self.assertIn("Definición", text)
        self.assertIn("botón", text)
        path.unlink()


# ════════════════════════════════════════════════════════════════════
# Integration tests for process_file()
# ════════════════════════════════════════════════════════════════════

class TestProcessFile(unittest.TestCase):

    def _process(self, content: str, filename: str = "test.md",
                 meta: dict = None) -> list[EnrichedChunk]:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            md_path = p / filename
            md_path.write_text(content, encoding="utf-8")
            if meta:
                stem = Path(filename).stem
                (p / f"{stem}_metadata.json").write_text(
                    json.dumps(meta), encoding="utf-8"
                )
            pipeline = make_pipeline(p, p / "out")
            return pipeline.process_file(md_path)

    # ── Return type ──────────────────────────────────────────────────────────

    def test_returns_list_of_enriched_chunks(self):
        result = self._process("### Header\n\nSome content text here.")
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, EnrichedChunk)

    def test_empty_file_returns_no_chunks(self):
        result = self._process("")
        self.assertEqual(result, [])

    # ── Tables preserved ─────────────────────────────────────────────────────

    def test_table_chunk_type_in_output(self):
        content = (
            "### Actions\n\n"
            "<table border='1'><thead><tr><th>Action</th><th>Entry</th></tr></thead>"
            "<tbody><tr><td>Create</td><td>Enter NT</td></tr></tbody></table>"
        )
        result = self._process(content)
        table_chunks = [c for c in result if c.metadata.get("chunk_type") == "table"]
        self.assertGreater(len(table_chunks), 0)

    def test_table_html_intact_in_content(self):
        content = (
            "### Actions\n\n"
            "<table><tr><th>System report</th></tr>"
            "<tr><td>CMI Report</td></tr></table>"
        )
        result = self._process(content)
        table_chunks = [c for c in result if c.metadata.get("chunk_type") == "table"]
        self.assertGreater(len(table_chunks), 0)
        self.assertIn("<table", table_chunks[0].content)
        self.assertIn("CMI Report", table_chunks[0].content)

    def test_two_tables_produce_two_table_chunks(self):
        content = (
            "### System reports\n\n"
            "<table><tr><td>Table A</td></tr></table>\n\n"
            "<table><tr><td>Table B</td></tr></table>"
        )
        result = self._process(content)
        table_chunks = [c for c in result if c.metadata.get("chunk_type") == "table"]
        self.assertEqual(len(table_chunks), 2)

    # ── Image descriptions ────────────────────────────────────────────────────

    def test_image_description_extracted(self):
        content = (
            "### MMI Disposition\n\n"
            "![Screenshot of the MAC menu](img_hash_4.webp)\n\n"
            "Some text follows here."
        )
        result = self._process(content)
        img_chunks = [c for c in result
                      if c.metadata.get("chunk_type") == "image_description"]
        self.assertGreater(len(img_chunks), 0)
        self.assertIn("Screenshot of the MAC menu", img_chunks[0].content)

    # ── TOC tables removed ────────────────────────────────────────────────────

    def test_toc_table_not_in_output(self):
        content = (
            "# Table of contents\n\n"
            "<table><tr><td>Introduction .....</td><td>6</td></tr></table>\n\n"
            "## Introduction\n\nReal content starts here now."
        )
        result = self._process(content)
        all_content = " ".join(c.content for c in result)
        self.assertNotIn("Introduction .....", all_content)

    # ── Adjacent tables fixed ─────────────────────────────────────────────────

    def test_adjacent_tables_both_preserved(self):
        content = (
            "### System reports\n\n"
            "<table><tr><td>EN report</td></tr></table>"
            "<table><tr><td>ES informe</td></tr></table>"
        )
        result = self._process(content)
        table_chunks = [c for c in result if c.metadata.get("chunk_type") == "table"]
        # Both tables must appear as separate chunks
        self.assertEqual(len(table_chunks), 2)
        contents = " ".join(c.content for c in table_chunks)
        self.assertIn("EN report", contents)
        self.assertIn("ES informe", contents)

    # ── Superscripts removed ──────────────────────────────────────────────────

    def test_superscripts_removed_from_text_chunks(self):
        content = (
            "### Notes\n\n"
            "Parameter enabled<sup>d</sup> by system configuration. "
            "Few-shot prompting<sup>7</sup> uses multiple examples."
        )
        result = self._process(content)
        text_chunks = [c for c in result if c.metadata.get("chunk_type") == "text"]
        for chunk in text_chunks:
            self.assertNotIn("<sup>", chunk.content)

    # ── Metadata correctness ──────────────────────────────────────────────────

    def test_all_chunks_have_source_file(self):
        result = self._process("### H\n\nContent with enough text here.")
        for chunk in result:
            self.assertIn("source_file", chunk.metadata)

    def test_all_chunks_have_breadcrumb(self):
        result = self._process("### H\n\nContent text.")
        for chunk in result:
            self.assertIn("breadcrumb", chunk.metadata)

    def test_all_chunks_have_chunk_type(self):
        result = self._process("### H\n\nContent text.")
        for chunk in result:
            self.assertIn("chunk_type", chunk.metadata)

    def test_chunk_indices_sequential(self):
        content = (
            "### Section A\n\nText content here.\n\n"
            "<table><tr><td>T</td></tr></table>\n\n"
            "### Section B\n\nMore text content here."
        )
        result = self._process(content)
        indices = [c.metadata.get("chunk_index", -1) for c in result]
        self.assertEqual(indices, list(range(len(result))))

    def test_breadcrumb_h3_from_binder2_style(self):
        """Binder2.md starts at ###; breadcrumb must reflect that."""
        content = "### SERVICE: Great circle\n\nThis service allows operator to manipulate GCI."
        result = self._process(content)
        text_chunks = [c for c in result if c.metadata.get("chunk_type") == "text"]
        self.assertGreater(len(text_chunks), 0)
        bc = text_chunks[0].metadata["breadcrumb"]
        self.assertIn("###", bc)
        self.assertIn("Great circle", bc)

    def test_metadata_json_sibling_used(self):
        content = "### H\n\nContent text here long enough."
        meta = {
            "file_name": "test.pdf",
            "num_pages": 1,
            "pages": [{"page_num": 0, "num_chunks": 5, "num_images": 0}]
        }
        result = self._process(content, "test.md", meta)
        chunks_with_pdf = [c for c in result if "source_pdf" in c.metadata]
        self.assertGreater(len(chunks_with_pdf), 0)
        self.assertEqual(chunks_with_pdf[0].metadata["source_pdf"], "test.pdf")

    # ── Output serialisability ────────────────────────────────────────────────

    def test_all_chunks_serialisable_to_json(self):
        content = (
            "### Header\n\n"
            "Content text here.\n\n"
            "<table><tr><td>Data</td></tr></table>"
        )
        result = self._process(content)
        for chunk in result:
            # Must not raise
            serialized = json.dumps(chunk.to_dict())
            self.assertIsInstance(serialized, str)


# ════════════════════════════════════════════════════════════════════
# Tests for run() method — file I/O
# ════════════════════════════════════════════════════════════════════

class TestPipelineRun(unittest.TestCase):

    def test_run_produces_json_output_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            input_dir = p / "input"
            output_dir = p / "output"
            input_dir.mkdir()
            (input_dir / "doc.md").write_text(
                "### Section\n\nSome content text here.", encoding="utf-8"
            )
            pipeline = make_pipeline(input_dir, output_dir)
            summary = pipeline.run()
            self.assertEqual(summary["files_processed"], 1)
            json_files = list(output_dir.rglob("*.chunks.json"))
            self.assertEqual(len(json_files), 1)

    def test_run_json_valid_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            input_dir = p / "input"
            output_dir = p / "output"
            input_dir.mkdir()
            (input_dir / "doc.md").write_text(
                "### Header\n\nBody text here.", encoding="utf-8"
            )
            pipeline = make_pipeline(input_dir, output_dir)
            pipeline.run()
            json_file = next(output_dir.rglob("*.chunks.json"))
            data = json.loads(json_file.read_text(encoding="utf-8"))
            self.assertIn("source_file", data)
            self.assertIn("total_chunks", data)
            self.assertIn("chunks", data)
            self.assertIsInstance(data["chunks"], list)

    def test_run_mirrors_directory_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            input_dir = p / "input"
            sub = input_dir / "vol1" / "chap2"
            sub.mkdir(parents=True)
            (sub / "file.md").write_text("### H\n\nContent.", encoding="utf-8")
            output_dir = p / "output"
            pipeline = make_pipeline(input_dir, output_dir)
            pipeline.run()
            expected = output_dir / "vol1" / "chap2" / "file.chunks.json"
            self.assertTrue(expected.exists(),
                            f"Expected output at {expected}")

    def test_run_no_md_files_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            input_dir = p / "input"
            input_dir.mkdir()
            pipeline = make_pipeline(input_dir, p / "output")
            summary = pipeline.run()
            self.assertEqual(summary["files_processed"], 0)
            self.assertEqual(summary["total_chunks"], 0)

    def test_run_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            input_dir = p / "input"
            input_dir.mkdir()
            for i in range(3):
                (input_dir / f"doc{i}.md").write_text(
                    f"### Section {i}\n\nContent {i} text here.",
                    encoding="utf-8"
                )
            pipeline = make_pipeline(input_dir, p / "output")
            summary = pipeline.run()
            self.assertEqual(summary["files_processed"], 3)

    def test_run_summary_total_chunks_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            input_dir = p / "input"
            input_dir.mkdir()
            (input_dir / "doc.md").write_text(
                "### A\n\nText A.\n\n### B\n\nText B.", encoding="utf-8"
            )
            pipeline = make_pipeline(input_dir, p / "output")
            summary = pipeline.run()
            self.assertGreater(summary["total_chunks"], 0)

    def test_run_json_chunks_have_content_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            input_dir = p / "input"
            input_dir.mkdir()
            (input_dir / "doc.md").write_text(
                "### Header\n\nContent text here.", encoding="utf-8"
            )
            pipeline = make_pipeline(input_dir, p / "output")
            pipeline.run()
            json_file = next((p / "output").rglob("*.chunks.json"))
            data = json.loads(json_file.read_text(encoding="utf-8"))
            for chunk in data["chunks"]:
                self.assertIn("content", chunk)
                self.assertIn("metadata", chunk)


# ════════════════════════════════════════════════════════════════════
# Real file integration tests
# ════════════════════════════════════════════════════════════════════

class TestRealFilesIntegration(unittest.TestCase):

    def _run_on_real_file(self, md_filename: str) -> list[EnrichedChunk]:
        md_path = FIXTURES_DIR / md_filename
        if not md_path.exists():
            self.skipTest(f"{md_filename} not available")

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "output"
            # copy files into temp so metadata sibling is found
            import shutil
            tmp_input = Path(tmp) / "input"
            tmp_input.mkdir()
            shutil.copy(md_path, tmp_input / md_filename)
            stem = Path(md_filename).stem
            meta_src = FIXTURES_DIR / f"{stem}_metadata.json"
            if meta_src.exists():
                shutil.copy(meta_src, tmp_input / f"{stem}_metadata.json")

            pipeline = make_pipeline(tmp_input, out_dir)
            return pipeline.process_file(tmp_input / md_filename)

    def test_binder2_produces_chunks(self):
        chunks = self._run_on_real_file("Binder2.md")
        self.assertGreater(len(chunks), 0)

    def test_binder2_has_table_chunks(self):
        chunks = self._run_on_real_file("Binder2.md")
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        self.assertGreaterEqual(len(table_chunks), 4)

    def test_binder2_has_image_description_chunk(self):
        chunks = self._run_on_real_file("Binder2.md")
        img_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "image_description"]
        self.assertGreater(len(img_chunks), 0)

    def test_binder2_no_sup_tags_in_any_chunk(self):
        chunks = self._run_on_real_file("Binder2.md")
        for chunk in chunks:
            self.assertNotIn("<sup>", chunk.content,
                             f"<sup> found in chunk: {chunk.content[:80]}")

    def test_binder2_no_webp_refs_in_chunks(self):
        chunks = self._run_on_real_file("Binder2.md")
        for chunk in chunks:
            self.assertNotIn(".webp", chunk.content,
                             f".webp found in chunk: {chunk.content[:80]}")

    def test_binder2_no_toc_content(self):
        chunks = self._run_on_real_file("Binder2.md")
        all_content = " ".join(c.content for c in chunks)
        self.assertNotIn("Introduction .....", all_content)

    def test_binder2_chunks_have_all_metadata_fields(self):
        chunks = self._run_on_real_file("Binder2.md")
        required_fields = {"source_file", "breadcrumb", "chunk_type",
                           "chunk_index", "language"}
        for chunk in chunks:
            for field in required_fields:
                self.assertIn(field, chunk.metadata,
                              f"Field '{field}' missing from chunk metadata")

    def test_binder2_chunk_indices_sequential(self):
        chunks = self._run_on_real_file("Binder2.md")
        indices = [c.metadata["chunk_index"] for c in chunks]
        self.assertEqual(indices, list(range(len(chunks))))

    def test_binder2_all_chunks_json_serialisable(self):
        chunks = self._run_on_real_file("Binder2.md")
        for chunk in chunks:
            try:
                json.dumps(chunk.to_dict())
            except (TypeError, ValueError) as e:
                self.fail(f"Chunk not JSON-serialisable: {e}\n{chunk.to_dict()}")

    def test_testing_produces_chunks(self):
        chunks = self._run_on_real_file("Testing.md")
        self.assertGreater(len(chunks), 0)

    def test_testing_has_four_image_description_chunks(self):
        chunks = self._run_on_real_file("Testing.md")
        img_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "image_description"]
        self.assertEqual(len(img_chunks), 4)

    def test_testing_has_table_chunks(self):
        chunks = self._run_on_real_file("Testing.md")
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        self.assertGreaterEqual(len(table_chunks), 2)

    def test_testing_no_toc_content(self):
        chunks = self._run_on_real_file("Testing.md")
        all_content = " ".join(c.content for c in chunks)
        self.assertNotIn("Introduction .....", all_content)

    def test_testing_multi_level_breadcrumbs(self):
        chunks = self._run_on_real_file("Testing.md")
        deep_bc = [c for c in chunks if ">" in c.metadata.get("breadcrumb", "")]
        self.assertGreater(len(deep_bc), 0,
                           "Testing.md should produce multi-level breadcrumbs")

    def test_testing_english_chunks_detected(self):
        chunks = self._run_on_real_file("Testing.md")
        en_chunks = [c for c in chunks
                     if c.metadata.get("language") == "en"
                     and c.metadata.get("chunk_type") == "text"]
        self.assertGreater(len(en_chunks), 0)

    def test_binder2_spanish_chunks_detected(self):
        chunks = self._run_on_real_file("Binder2.md")
        es_chunks = [c for c in chunks if c.metadata.get("language") == "es"]
        self.assertGreater(len(es_chunks), 0,
                           "Binder2.md has Spanish content that should be detected")

    def test_testing_no_sup_tags(self):
        chunks = self._run_on_real_file("Testing.md")
        for chunk in chunks:
            self.assertNotIn("<sup>", chunk.content)

    def test_testing_no_webp_in_chunks(self):
        chunks = self._run_on_real_file("Testing.md")
        for chunk in chunks:
            self.assertNotIn(".webp", chunk.content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
