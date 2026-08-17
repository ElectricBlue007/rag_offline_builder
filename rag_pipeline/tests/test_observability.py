"""
tests/test_observability.py -- tests for the observability layer.

These tests protect one promise: observability must never change what the
pipeline produces, and must never crash a run that would otherwise succeed.
So they check both the happy path (steps are recorded, reports are written)
and the safety path (a disabled report records nothing, a failing step is
re-raised and not swallowed).

Everything here is stdlib-only: no Ollama, no Chroma, no network.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_pipeline.observability import i18n
from rag_pipeline.observability.health import (
    RED,
    analyze_chunks,
    corpus_stats,
    findings_md,
    load_chunks,
)
from rag_pipeline.observability.run_report import (
    FAILED,
    OK,
    STEP_DOCS,
    RunReport,
    doc_for,
)


def _chunk(cid="c1", text="x" * 400, language="en", breadcrumb="# Doc > ## Part",
           pages=(1,), embed=True, source="a.md"):
    """A chunk shaped exactly like the ones write_chunks writes to disk."""
    return {
        "chunk_id": cid,
        "content": text,
        "normalized_text": text,
        "contextualized_text": f"From {source}, {breadcrumb}. {text}",
        "embed_text": (text if embed else ""),
        "metadata": {
            "source_file": source,
            "breadcrumb": breadcrumb,
            "language": language,
            "page_numbers": list(pages),
            "chunk_type": "text",
        },
    }


# -- i18n ---------------------------------------------------------------------

class TestI18n(unittest.TestCase):
    def test_pct_of_zero_is_not_a_crash(self):
        """An empty corpus must not divide by zero."""
        self.assertEqual(i18n.pct(0, 0), "n/a")

    def test_pct_normal(self):
        self.assertEqual(i18n.pct(1, 2), "50.0%")

    def test_human_seconds_scales(self):
        self.assertIn("ms", i18n.human_seconds(0.045))
        self.assertIn("s", i18n.human_seconds(1.2))
        self.assertIn("m", i18n.human_seconds(184.0))

    def test_rtl_wraps_persian_for_notion_and_github(self):
        self.assertIn('dir="rtl"', i18n.rtl("سلام"))

    def test_heading_both_shows_both_languages(self):
        out = i18n.heading(("Chunks", "قطعات"), 2, i18n.BOTH)
        self.assertIn("Chunks", out)
        self.assertIn("قطعات", out)

    def test_heading_en_hides_persian(self):
        out = i18n.heading(("Chunks", "قطعات"), 2, i18n.EN)
        self.assertNotIn("قطعات", out)

    def test_fence_truncates_long_samples_and_marks_it(self):
        out = i18n.fence("y" * 5000, max_chars=100)
        self.assertLess(len(out), 1000)
        self.assertIn("truncated", out.lower())


# -- RunReport ----------------------------------------------------------------

class TestRunReport(unittest.TestCase):
    def test_disabled_report_records_nothing_and_writes_nothing(self):
        """This is what every existing caller gets, so it must be inert."""
        r = RunReport(enabled=False)
        with r.step("load_markdown", items_in=1) as st:
            st.done(1)
            st.note(("en", "fa"))
        r.record("write_chunks", items_out=5)
        self.assertEqual(r.steps, [])
        self.assertIsNone(r.write())

    def test_step_records_counts_and_time(self):
        r = RunReport(enabled=True)
        with r.step("split_structure", items_in=1) as st:
            st.done(12)
        self.assertEqual(len(r.steps), 1)
        step = r.steps[0]
        self.assertEqual(step.name, "split_structure")
        self.assertEqual(step.items_in, 1)
        self.assertEqual(step.items_out, 12)
        self.assertEqual(step.status, OK)
        self.assertGreaterEqual(step.seconds, 0.0)

    def test_failing_step_is_recorded_and_re_raised(self):
        """Observability must never hide a real failure."""
        r = RunReport(enabled=True)
        with self.assertRaises(ValueError):
            with r.step("split_semantic", items_in=3):
                raise ValueError("embedding model unreachable")
        self.assertEqual(r.steps[0].status, FAILED)
        self.assertIn("embedding model unreachable", r.steps[0].error)
        self.assertIn("ValueError", r.steps[0].error)

    def test_delta_reports_how_many_items_the_step_added_or_removed(self):
        r = RunReport(enabled=True)
        with r.step("split_semantic", items_in=4) as st:
            st.done(9)
        self.assertEqual(r.steps[0].delta, 5)

    def test_record_appends_a_step_in_order(self):
        r = RunReport(enabled=True)
        r.record("discover_files", items_out=2)
        r.record("write_chunks", items_in=2, items_out=2)
        self.assertEqual([s.name for s in r.steps], ["discover_files", "write_chunks"])

    def test_every_pipeline_stage_has_documentation(self):
        """A step with no explanation would defeat the whole point."""
        for name in ("load_markdown", "preprocess_markdown", "split_structure",
                     "resolve_images", "split_semantic", "refine_chunks",
                     "normalize_tables", "enrich_context", "build_embed_text",
                     "enrich_metadata", "write_chunks"):
            self.assertIn(name, STEP_DOCS, f"{name} has no StepDoc")
            self.assertTrue(doc_for(name).what[0], f"{name} has an empty explanation")
            self.assertTrue(doc_for(name).source, f"{name} does not name its source file")

    def test_unknown_step_still_returns_a_doc(self):
        self.assertIsNotNone(doc_for("a_step_that_does_not_exist").title)

    def test_write_produces_the_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = RunReport(out_dir=tmp, title="Test run", enabled=True, command="pytest")
            with r.step("load_markdown", items_in=1) as st:
                st.done(1)
                st.sample(("Sample", "نمونه"), "hello world")
            r.add_findings(analyze_chunks([_chunk(text="short")]))
            folder = r.write()
            self.assertIsNotNone(folder)
            self.assertTrue((folder / "00_SUMMARY.md").exists())
            self.assertTrue((folder / "01_load_markdown.md").exists())
            self.assertTrue((folder / "DATA_HEALTH.md").exists())
            trace = json.loads((folder / "RAW_TRACE.json").read_text(encoding="utf-8"))
            self.assertEqual(trace["steps"][0]["name"], "load_markdown")
            self.assertEqual(trace["command"], "pytest")

    def test_step_file_names_the_source_file_that_implements_the_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = RunReport(out_dir=tmp, enabled=True)
            with r.step("split_structure", items_in=1) as st:
                st.done(3)
            folder = r.write()
            text = (folder / "01_split_structure.md").read_text(encoding="utf-8")
            self.assertIn("structure_splitter", text)

    def test_console_summary_lists_steps(self):
        r = RunReport(enabled=True)
        with r.step("write_chunks", items_in=2) as st:
            st.done(2)
        self.assertIn("write_chunks", r.console_summary())


# -- Data health --------------------------------------------------------------

class TestHealthRules(unittest.TestCase):
    def ids(self, findings):
        return {f.id for f in findings}

    def test_no_chunks_is_reported_as_red(self):
        findings = analyze_chunks([])
        self.assertIn("no_chunks", self.ids(findings))
        self.assertEqual(findings[0].icon, RED)

    def test_empty_text_is_detected(self):
        bad = _chunk(cid="c9", text="")
        self.assertIn("empty_text", self.ids(analyze_chunks([bad, _chunk()])))

    def test_too_short_uses_the_projects_own_threshold(self):
        chunks = [_chunk(cid="c1", text="tiny"), _chunk(cid="c2")]
        self.assertIn("too_short", self.ids(analyze_chunks(chunks, min_chunk_chars=80)))
        self.assertNotIn("too_short", self.ids(analyze_chunks(chunks, min_chunk_chars=2)))

    def test_too_long_is_detected(self):
        chunks = [_chunk(cid="c1", text="z" * 5000)]
        self.assertIn("too_long", self.ids(analyze_chunks(chunks, max_chunk_chars=2000)))

    def test_duplicate_chunk_ids_are_detected(self):
        chunks = [_chunk(cid="same"), _chunk(cid="same", text="different text " * 40)]
        self.assertIn("duplicate_chunk_id", self.ids(analyze_chunks(chunks)))

    def test_unresolved_image_placeholder_is_detected(self):
        chunks = [_chunk(cid="c1", text="see [IMAGE_DESC_PLACEHOLDER_2] for details " * 12)]
        self.assertIn("unresolved_image_placeholder", self.ids(analyze_chunks(chunks)))

    def test_missing_embed_text_only_flagged_when_expected(self):
        chunks = [_chunk(cid="c1", embed=False)]
        self.assertIn("missing_embed_text",
                      self.ids(analyze_chunks(chunks, expect_embed_text=True)))
        self.assertNotIn("missing_embed_text",
                         self.ids(analyze_chunks(chunks, expect_embed_text=False)))

    def test_unknown_language_is_detected(self):
        chunks = [_chunk(cid="c1", language="")]
        self.assertIn("unknown_language", self.ids(analyze_chunks(chunks)))

    def test_healthy_corpus_reports_no_content_problems(self):
        chunks = [
            _chunk(cid="a1", source="a.md", text="good content here. " * 30),
            _chunk(cid="b1", source="b.md", text="other good content. " * 30),
        ]
        found = self.ids(analyze_chunks(chunks))
        for rule in ("no_chunks", "empty_text", "too_short", "too_long",
                     "duplicate_chunk_id", "missing_embed_text", "unknown_language"):
            self.assertNotIn(rule, found)

    def test_every_finding_says_why_and_how_to_fix_it(self):
        for f in analyze_chunks([_chunk(cid="c1", text="tiny")]):
            self.assertTrue(f.why[0], f"{f.id} does not explain why it matters")
            self.assertTrue(f.fix[0], f"{f.id} does not say how to fix it")

    def test_corpus_stats_describes_the_corpus(self):
        rows = corpus_stats([_chunk(cid="c1"), _chunk(cid="c2", source="b.md")])
        self.assertTrue(rows)

    def test_findings_md_is_readable_markdown(self):
        md = findings_md(analyze_chunks([_chunk(cid="c1", text="tiny")]))
        self.assertIn("#", md)
        self.assertIn("too_short", md)

    def test_load_chunks_reads_real_chunk_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "nested"
            folder.mkdir()
            (folder / "doc.chunks.json").write_text(json.dumps({
                "source_file": "doc.md", "total_chunks": 2,
                "chunks": [_chunk(cid="c1"), _chunk(cid="c2")],
            }, ensure_ascii=False), encoding="utf-8")
            chunks = load_chunks(Path(tmp))
            self.assertEqual(len(chunks), 2)
            self.assertIn("_file", chunks[0])

    def test_load_chunks_ignores_broken_json_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "broken.chunks.json").write_text("{not json", encoding="utf-8")
            self.assertEqual(load_chunks(Path(tmp)), [])


# -- The pipeline keeps working without a report ------------------------------

class TestPipelineStaysUnchanged(unittest.TestCase):
    def test_pipeline_can_be_built_without_a_report(self):
        from rag_pipeline.config import PipelineConfig
        from rag_pipeline.pipeline import MarkdownChunkingPipeline
        p = MarkdownChunkingPipeline(PipelineConfig())
        self.assertFalse(p.report.enabled)
        self.assertIsNone(p.report.write())

    def test_pipeline_accepts_a_report(self):
        from rag_pipeline.config import PipelineConfig
        from rag_pipeline.pipeline import MarkdownChunkingPipeline
        r = RunReport(enabled=True)
        p = MarkdownChunkingPipeline(PipelineConfig(), report=r)
        self.assertIs(p.report, r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
