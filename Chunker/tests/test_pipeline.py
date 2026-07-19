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

    # ── Table normalization wiring (real TableChunkNormalizer, not mocked) ────

    def test_table_normalized_text_differs_from_raw_html(self):
        """
        This is the key end-to-end proof that TableChunkNormalizer is
        actually wired into process_file — normalized_text must NOT
        equal the raw HTML content for a table chunk.
        """
        content = (
            "### Actions\n\n"
            "<table border='1'><thead><tr><th>Action</th><th>Entry</th></tr></thead>"
            "<tbody><tr><td>Create SOI</td><td>CMI track</td></tr></tbody></table>"
        )
        result = self._process(content)
        table_chunks = [c for c in result if c.metadata.get("chunk_type") == "table"]
        self.assertGreater(len(table_chunks), 0)
        self.assertNotEqual(table_chunks[0].content, table_chunks[0].normalized_text)

    def test_table_normalized_text_has_no_html_tags(self):
        content = (
            "### Actions\n\n"
            "<table border='1'><thead><tr><th>Action</th><th>Entry</th></tr></thead>"
            "<tbody><tr><td>Create SOI</td><td>CMI track</td></tr></tbody></table>"
        )
        result = self._process(content)
        table_chunks = [c for c in result if c.metadata.get("chunk_type") == "table"]
        self.assertNotIn("<table", table_chunks[0].normalized_text)
        self.assertNotIn("<td", table_chunks[0].normalized_text)
        self.assertNotIn("<th", table_chunks[0].normalized_text)

    def test_table_normalized_text_preserves_field_relationship(self):
        """The header-row normalizer must produce 'field: value' style text."""
        content = (
            "### Actions\n\n"
            "<table border='1'><thead><tr><th>Action</th><th>Entry</th></tr></thead>"
            "<tbody><tr><td>Create SOI</td><td>CMI track</td></tr></tbody></table>"
        )
        result = self._process(content)
        table_chunks = [c for c in result if c.metadata.get("chunk_type") == "table"]
        self.assertIn("Create SOI", table_chunks[0].normalized_text)
        self.assertIn("entry:", table_chunks[0].normalized_text)
        self.assertIn("CMI track", table_chunks[0].normalized_text)

    def test_table_content_field_still_has_raw_html(self):
        """content must remain the original HTML — normalized_text is separate."""
        content = (
            "### Actions\n\n"
            "<table border='1'><thead><tr><th>Action</th></tr></thead>"
            "<tbody><tr><td>Create SOI</td></tr></tbody></table>"
        )
        result = self._process(content)
        table_chunks = [c for c in result if c.metadata.get("chunk_type") == "table"]
        self.assertIn("<table", table_chunks[0].content)
        self.assertIn("<td>", table_chunks[0].content)

    def test_text_chunk_normalized_text_equals_content(self):
        """Non-table chunks must have normalized_text == content, unchanged."""
        content = "### Header\n\nThis is plain text content with enough length."
        result = self._process(content)
        text_chunks = [c for c in result if c.metadata.get("chunk_type") == "text"]
        self.assertGreater(len(text_chunks), 0)
        self.assertEqual(text_chunks[0].content, text_chunks[0].normalized_text)

    def test_image_description_normalized_text_equals_content(self):
        content = (
            "### MMI Disposition\n\n"
            "![A screenshot of the interface](img.webp)\n\n"
            "Some following text here."
        )
        result = self._process(content)
        img_chunks = [c for c in result
                      if c.metadata.get("chunk_type") == "image_description"]
        self.assertGreater(len(img_chunks), 0)
        self.assertEqual(img_chunks[0].content, img_chunks[0].normalized_text)

    def test_key_value_table_normalized_as_single_record(self):
        """Testing.md-style key-value table (bold label, no <th>) via full pipeline."""
        content = (
            "### Example\n\n"
            "<table border='1'>"
            "<tr><td><b>Model</b></td><td colspan='3'>gemini-pro</td></tr>"
            "<tr><td><b>Temperature</b></td><td>0.1</td>"
            "<td><b>Top-P</b></td><td>1</td></tr>"
            "</table>"
        )
        result = self._process(content)
        table_chunks = [c for c in result if c.metadata.get("chunk_type") == "table"]
        self.assertGreater(len(table_chunks), 0)
        norm = table_chunks[0].normalized_text
        self.assertIn("Model: gemini-pro", norm)
        self.assertIn("Temperature: 0.1", norm)
        self.assertIn("Top-P: 1", norm)

    def test_malformed_table_falls_back_gracefully_not_crash(self):
        """A pipeline run must never crash on one bad table."""
        content = "### Broken\n\n<table><tr><td>Unclosed cell</table>"
        # Should not raise — either normalizes via lenient HTML parsing
        # or falls back to raw content, but never throws.
        try:
            result = self._process(content)
        except Exception as exc:
            self.fail(f"Pipeline crashed on malformed table: {exc}")
        self.assertIsInstance(result, list)

    def test_all_enriched_chunks_have_normalized_text_field(self):
        content = (
            "### Section\n\n"
            "Some text here.\n\n"
            "<table><tr><td>Data</td></tr></table>\n\n"
            "![An image](x.webp)"
        )
        result = self._process(content)
        for chunk in result:
            self.assertTrue(hasattr(chunk, "normalized_text"))
            self.assertIsInstance(chunk.normalized_text, str)

    def test_to_dict_includes_normalized_text(self):
        content = "### H\n\n<table><tr><td>Data</td></tr></table>"
        result = self._process(content)
        for chunk in result:
            d = chunk.to_dict()
            self.assertIn("normalized_text", d)

    # ── Contextual enrichment wiring (real ContextualEnricher, deterministic) ──

    def test_contextualized_text_includes_source_filename(self):
        content = "### Section\n\nSome content text here."
        result = self._process(content, filename="MyDoc.md")
        for chunk in result:
            self.assertIn("MyDoc.md", chunk.contextualized_text)

    def test_contextualized_text_includes_breadcrumb(self):
        content = "### System reports\n\nSome content text here long enough."
        result = self._process(content)
        text_chunks = [c for c in result if c.metadata.get("chunk_type") == "text"]
        self.assertGreater(len(text_chunks), 0)
        self.assertIn("System reports", text_chunks[0].contextualized_text)

    def test_contextualized_text_differs_from_normalized_text(self):
        """Proves the stage actually ran, not just defaulted through."""
        content = "### Section\n\nSome content text here."
        result = self._process(content)
        for chunk in result:
            self.assertNotEqual(chunk.contextualized_text, chunk.normalized_text)

    def test_table_contextualized_text_has_both_context_and_prose(self):
        content = (
            "### Actions\n\n"
            "<table border='1'><thead><tr><th>Action</th><th>Entry</th></tr></thead>"
            "<tbody><tr><td>Create SOI</td><td>CMI track</td></tr></tbody></table>"
        )
        result = self._process(content, filename="Ops.md")
        table_chunks = [c for c in result if c.metadata.get("chunk_type") == "table"]
        self.assertGreater(len(table_chunks), 0)
        ctext = table_chunks[0].contextualized_text
        self.assertIn("Ops.md", ctext)
        self.assertIn("Create SOI", ctext)
        self.assertNotIn("<table", ctext)  # must use normalized_text as base, not HTML

    def test_to_dict_includes_contextualized_text(self):
        content = "### H\n\n<table><tr><td>Data</td></tr></table>"
        result = self._process(content)
        for chunk in result:
            d = chunk.to_dict()
            self.assertIn("contextualized_text", d)

    # ── Embedding instruction-prefix contract (EmbedTextBuilder wiring) ──────
    # Current default (qwen3-embedding): no passage-side prefix at all, so
    # embed_text should exactly equal contextualized_text. These tests
    # deliberately don't hardcode a prefix string, since that's a
    # model-dependent choice (see EmbedTextConfig) that will change again
    # if the embedding model changes — the prefix MECHANISM itself is
    # covered directly in tests/test_embed_text_builder.py.

    def test_embed_text_equals_contextualized_text_by_default(self):
        content = "### Section\n\nSome content text here."
        result = self._process(content)
        for chunk in result:
            self.assertEqual(chunk.embed_text, chunk.contextualized_text)

    def test_embed_text_contains_contextualized_content(self):
        content = "### System reports\n\nSome content text here long enough."
        result = self._process(content, filename="Ops.md")
        text_chunks = [c for c in result if c.metadata.get("chunk_type") == "text"]
        self.assertGreater(len(text_chunks), 0)
        embed_text = text_chunks[0].embed_text
        self.assertIn("Ops.md", embed_text)          # from contextual enrichment
        self.assertIn("System reports", embed_text)   # from contextual enrichment

    def test_embed_text_table_has_no_html(self):
        content = (
            "### Actions\n\n"
            "<table border='1'><thead><tr><th>Action</th></tr></thead>"
            "<tbody><tr><td>Create SOI</td></tr></tbody></table>"
        )
        result = self._process(content)
        table_chunks = [c for c in result if c.metadata.get("chunk_type") == "table"]
        self.assertGreater(len(table_chunks), 0)
        self.assertNotIn("<table", table_chunks[0].embed_text)
        self.assertEqual(table_chunks[0].embed_text, table_chunks[0].contextualized_text)

    def test_to_dict_includes_embed_text(self):
        content = "### H\n\n<table><tr><td>Data</td></tr></table>"
        result = self._process(content)
        for chunk in result:
            d = chunk.to_dict()
            self.assertIn("embed_text", d)

    # ── Stable chunk_id ──────────────────────────────────────────────────────

    def test_chunk_id_present_and_nonempty(self):
        content = "### Section\n\nSome text content here."
        result = self._process(content)
        for chunk in result:
            self.assertTrue(chunk.chunk_id)
            self.assertIsInstance(chunk.chunk_id, str)

    def test_chunk_id_unique_within_document(self):
        content = "### A\n\nText A.\n\n### B\n\nText B.\n\n### C\n\nText C."
        result = self._process(content)
        ids = [c.chunk_id for c in result]
        self.assertEqual(len(ids), len(set(ids)))

    def test_chunk_id_deterministic_across_runs(self):
        """Same input processed twice must yield identical chunk_ids —
        this is what makes downstream ChromaDB upserts idempotent."""
        content = "### Section\n\nStable content for hashing."
        # Must reuse the SAME temp directory (i.e. the SAME resolved
        # md_path) across both calls — determinism is about "same path,
        # same content, same chunk_index" producing the same ID, not
        # about two different filesystem locations that happen to share
        # a filename. compute_chunk_id() correctly incorporates the
        # full source path, so two different temp dirs are expected to
        # differ; that's not what this test is meant to check.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            md_path = p / "Stable.md"
            md_path.write_text(content, encoding="utf-8")
            pipeline = make_pipeline(p, p / "out")

            result1 = pipeline.process_file(md_path)
            result2 = pipeline.process_file(md_path)

        ids1 = [c.chunk_id for c in result1]
        ids2 = [c.chunk_id for c in result2]
        self.assertEqual(ids1, ids2)

    def test_chunk_id_differs_across_source_files(self):
        """Same content, different filename → different chunk_id (since
        source_file is part of the hash input)."""
        content = "### Section\n\nSame content here."
        result_a = self._process(content, filename="A.md")
        result_b = self._process(content, filename="B.md")
        self.assertNotEqual(result_a[0].chunk_id, result_b[0].chunk_id)

    def test_to_dict_includes_chunk_id(self):
        content = "### H\n\n<table><tr><td>Data</td></tr></table>"
        result = self._process(content)
        for chunk in result:
            d = chunk.to_dict()
            self.assertIn("chunk_id", d)
            self.assertTrue(d["chunk_id"])

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

    def test_run_twice_produces_identical_chunk_ids(self):
        """
        The realistic idempotency scenario: same PipelineConfig.input_dir,
        pipeline run twice (e.g. re-run after adding an unrelated new
        file) — chunk_ids for unchanged files must come out identical
        both times, which is what makes ChromaDB .upsert() safe to call
        repeatedly without accumulating duplicates.
        """
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            input_dir = p / "input"
            input_dir.mkdir()
            (input_dir / "doc.md").write_text(
                "### Section\n\nContent that will not change.", encoding="utf-8"
            )
            pipeline = make_pipeline(input_dir, p / "output1")
            pipeline.run()
            json_file1 = next((p / "output1").rglob("*.chunks.json"))
            ids_run1 = [c["chunk_id"] for c in
                        json.loads(json_file1.read_text())["chunks"]]

            pipeline2 = make_pipeline(input_dir, p / "output2")
            pipeline2.run()
            json_file2 = next((p / "output2").rglob("*.chunks.json"))
            ids_run2 = [c["chunk_id"] for c in
                        json.loads(json_file2.read_text())["chunks"]]

            self.assertEqual(ids_run1, ids_run2)

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

    def test_binder2_table_chunks_normalized_no_html(self):
        """
        Every real table in Binder2.md must come out of the FULL pipeline
        with normalized_text containing no HTML tags — proves the
        normalizer stage runs on real production data, not just fixtures.
        """
        chunks = self._run_on_real_file("Binder2.md")
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        self.assertGreaterEqual(len(table_chunks), 4)
        for tc in table_chunks:
            self.assertNotIn("<table", tc.normalized_text)
            self.assertNotIn("<td", tc.normalized_text)
            self.assertNotIn("<tr", tc.normalized_text)

    def test_binder2_table_normalized_text_differs_from_content(self):
        chunks = self._run_on_real_file("Binder2.md")
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        for tc in table_chunks:
            self.assertNotEqual(tc.content, tc.normalized_text)

    def test_binder2_table_content_still_raw_html(self):
        """content must be untouched — still the original HTML for display."""
        chunks = self._run_on_real_file("Binder2.md")
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        for tc in table_chunks:
            self.assertIn("<table", tc.content)

    def test_binder2_key_action_preserved_in_normalized_text(self):
        """Spot-check: 'Create SOI' must survive normalization somewhere."""
        chunks = self._run_on_real_file("Binder2.md")
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        all_normalized = " ".join(tc.normalized_text for tc in table_chunks)
        self.assertIn("Create SOI", all_normalized)

    def test_testing_table_chunks_normalized_no_html(self):
        chunks = self._run_on_real_file("Testing.md")
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        self.assertGreaterEqual(len(table_chunks), 2)
        for tc in table_chunks:
            self.assertNotIn("<table", tc.normalized_text)
            self.assertNotIn("<td", tc.normalized_text)

    def test_testing_key_value_table_content_preserved_in_normalized(self):
        """Testing.md tables are key-value style — spot-check gemini-pro survives."""
        chunks = self._run_on_real_file("Testing.md")
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        all_normalized = " ".join(tc.normalized_text for tc in table_chunks)
        self.assertIn("gemini-pro", all_normalized)

    def test_testing_all_chunks_json_serialisable_with_normalized_text(self):
        chunks = self._run_on_real_file("Testing.md")
        for chunk in chunks:
            d = chunk.to_dict()
            self.assertIn("normalized_text", d)
            json.dumps(d)  # must not raise

    def test_binder2_contextualized_text_has_filename(self):
        chunks = self._run_on_real_file("Binder2.md")
        for chunk in chunks:
            self.assertIn("Binder2.md", chunk.contextualized_text)

    def test_binder2_contextualized_text_has_breadcrumb_context(self):
        chunks = self._run_on_real_file("Binder2.md")
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        self.assertGreater(len(table_chunks), 0)
        # At least one table chunk's breadcrumb section name should appear
        # in its own contextualized_text
        matched = 0
        for tc in table_chunks:
            section_header = tc.metadata.get("section_header", "")
            if section_header and section_header in tc.contextualized_text:
                matched += 1
        self.assertGreater(matched, 0)

    def test_binder2_table_contextualized_text_no_html(self):
        """Contextual text must be built from normalized_text, never raw HTML."""
        chunks = self._run_on_real_file("Binder2.md")
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        for tc in table_chunks:
            self.assertNotIn("<table", tc.contextualized_text)
            self.assertNotIn("<td", tc.contextualized_text)

    def test_testing_contextualized_text_has_filename(self):
        chunks = self._run_on_real_file("Testing.md")
        for chunk in chunks:
            self.assertIn("Testing.md", chunk.contextualized_text)

    def test_all_real_chunks_json_serialisable_with_contextualized_text(self):
        for fname in ["Binder2.md", "Testing.md"]:
            chunks = self._run_on_real_file(fname)
            for chunk in chunks:
                d = chunk.to_dict()
                self.assertIn("contextualized_text", d)
                json.dumps(d)  # must not raise

    def test_binder2_all_embed_text_equal_contextualized_text(self):
        chunks = self._run_on_real_file("Binder2.md")
        for chunk in chunks:
            self.assertEqual(chunk.embed_text, chunk.contextualized_text)

    def test_binder2_all_chunk_ids_unique(self):
        chunks = self._run_on_real_file("Binder2.md")
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_testing_all_embed_text_equal_contextualized_text(self):
        chunks = self._run_on_real_file("Testing.md")
        for chunk in chunks:
            self.assertEqual(chunk.embed_text, chunk.contextualized_text)

    def test_testing_all_chunk_ids_unique(self):
        chunks = self._run_on_real_file("Testing.md")
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_binder2_and_testing_chunk_ids_do_not_collide(self):
        """Chunk IDs must be unique ACROSS files too, not just within one."""
        binder_chunks = self._run_on_real_file("Binder2.md")
        testing_chunks = self._run_on_real_file("Testing.md")
        all_ids = [c.chunk_id for c in binder_chunks] + [c.chunk_id for c in testing_chunks]
        self.assertEqual(len(all_ids), len(set(all_ids)))

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
