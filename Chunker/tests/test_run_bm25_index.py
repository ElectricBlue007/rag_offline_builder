"""
tests/test_run_bm25_index.py

Tests for run_bm25_index.py's argument parsing and main() orchestration.
Since rank_bm25 is pure Python (no heavy model/service), these tests
run against the REAL BM25Index — no mocking needed for the index
itself, only for I/O edge cases like "no chunks found".
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_SPEC = importlib.util.spec_from_file_location(
    "run_bm25_index", Path(__file__).parent.parent / "run_bm25_index.py"
)
run_bm25_index = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_bm25_index)


class TestArgParsing(unittest.TestCase):

    def _parse(self, argv):
        with patch.object(sys, "argv", ["run_bm25_index.py"] + argv):
            return run_bm25_index.parse_args()

    def test_defaults(self):
        args = self._parse([])
        self.assertEqual(args.chunks_dir, "./chunks")
        self.assertEqual(args.output, "./bm25_index.pkl")
        self.assertEqual(args.k1, 1.5)
        self.assertEqual(args.b, 0.75)
        self.assertFalse(args.quiet)

    def test_custom_chunks_dir(self):
        args = self._parse(["--chunks-dir", "/custom"])
        self.assertEqual(args.chunks_dir, "/custom")

    def test_custom_output_short_flag(self):
        args = self._parse(["-o", "/custom/index.pkl"])
        self.assertEqual(args.output, "/custom/index.pkl")

    def test_custom_k1_and_b(self):
        args = self._parse(["--k1", "2.0", "--b", "0.5"])
        self.assertEqual(args.k1, 2.0)
        self.assertEqual(args.b, 0.5)

    def test_quiet_flag(self):
        args = self._parse(["-q"])
        self.assertTrue(args.quiet)


class TestMainOrchestrationRealBM25(unittest.TestCase):
    """
    Uses the REAL BM25Index (pure Python, no mocking needed) with
    real temp-file I/O — genuinely end-to-end for this module.
    """

    def test_no_chunks_found_exits_with_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(sys, "argv", ["run_bm25_index.py",
                                            "--chunks-dir", tmp, "--quiet"]):
                with self.assertRaises(SystemExit) as cm:
                    run_bm25_index.main()
                self.assertEqual(cm.exception.code, 1)

    def test_full_run_builds_and_saves_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            chunks_dir = Path(tmp) / "chunks"
            chunks_dir.mkdir()
            output_path = Path(tmp) / "index.pkl"

            payload = {
                "source_file": "test.md",
                "chunks": [
                    {"chunk_id": "a", "contextualized_text": "CMI track content here"},
                    {"chunk_id": "b", "contextualized_text": "prompt engineering content"},
                ],
            }
            import json
            (chunks_dir / "test.chunks.json").write_text(json.dumps(payload))

            with patch.object(sys, "argv", [
                "run_bm25_index.py",
                "--chunks-dir", str(chunks_dir),
                "--output", str(output_path),
                "--quiet",
            ]):
                with self.assertRaises(SystemExit) as cm:
                    run_bm25_index.main()
                self.assertEqual(cm.exception.code, 0)

            self.assertTrue(output_path.exists())

            # Verify the saved index is actually usable
            from Chunker.storage.bm25_index import BM25Index
            index = BM25Index()
            index.load(str(output_path))
            self.assertEqual(index.size, 2)
            results = index.search("CMI track")
            self.assertEqual(results[0].chunk_id, "a")

    def test_custom_k1_b_passed_through_to_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            chunks_dir = Path(tmp) / "chunks"
            chunks_dir.mkdir()
            output_path = Path(tmp) / "index.pkl"

            import json
            payload = {"chunks": [{"chunk_id": "a", "contextualized_text": "content"}]}
            (chunks_dir / "test.chunks.json").write_text(json.dumps(payload))

            with patch.object(sys, "argv", [
                "run_bm25_index.py",
                "--chunks-dir", str(chunks_dir),
                "--output", str(output_path),
                "--k1", "2.5",
                "--b", "0.3",
                "--quiet",
            ]):
                with self.assertRaises(SystemExit):
                    run_bm25_index.main()

            from Chunker.storage.bm25_index import BM25Index
            index = BM25Index()
            index.load(str(output_path))
            # The underlying rank_bm25 object stores k1/b as attributes
            self.assertEqual(index._bm25.k1, 2.5)
            self.assertEqual(index._bm25.b, 0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
