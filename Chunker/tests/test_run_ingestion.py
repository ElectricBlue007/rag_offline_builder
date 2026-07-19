"""
tests/test_run_ingestion.py

Tests for run_ingestion.py's argument parsing and main() orchestration
logic, using mocked EmbedGenerator/ChromaIngestor — no live model or
ChromaDB service required.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# run_ingestion.py lives at the project root inside Chunker/, and
# is a script (not part of the Chunker package's normal import
# path) — import it directly via its file location.
import importlib.util

_SPEC = importlib.util.spec_from_file_location(
    "run_ingestion", Path(__file__).parent.parent / "run_ingestion.py"
)
run_ingestion = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_ingestion)


class TestArgParsing(unittest.TestCase):

    def _parse(self, argv):
        with patch.object(sys, "argv", ["run_ingestion.py"] + argv):
            return run_ingestion.parse_args()

    def test_defaults(self):
        args = self._parse([])
        self.assertEqual(args.chunks_dir, "./chunks")
        self.assertEqual(args.chroma_dir, "./chroma_db")
        self.assertEqual(args.collection, "rag_docs")
        self.assertEqual(args.model, "qwen3-embedding:8b")
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.host, "http://localhost:11434")
        self.assertFalse(args.quiet)

    def test_custom_chunks_dir(self):
        args = self._parse(["--chunks-dir", "/custom/chunks"])
        self.assertEqual(args.chunks_dir, "/custom/chunks")

    def test_custom_chroma_dir_short_flag(self):
        args = self._parse(["-d", "/custom/chroma"])
        self.assertEqual(args.chroma_dir, "/custom/chroma")

    def test_custom_collection(self):
        args = self._parse(["--collection", "my_docs"])
        self.assertEqual(args.collection, "my_docs")

    def test_custom_model_short_flag(self):
        args = self._parse(["-m", "custom-model"])
        self.assertEqual(args.model, "custom-model")

    def test_custom_batch_size(self):
        args = self._parse(["--batch-size", "16"])
        self.assertEqual(args.batch_size, 16)
        self.assertIsInstance(args.batch_size, int)

    def test_custom_host(self):
        args = self._parse(["--host", "http://example-host:11434"])
        self.assertEqual(args.host, "http://example-host:11434")

    def test_quiet_flag(self):
        args = self._parse(["--quiet"])
        self.assertTrue(args.quiet)

    def test_quiet_short_flag(self):
        args = self._parse(["-q"])
        self.assertTrue(args.quiet)


class TestMainOrchestration(unittest.TestCase):
    """
    Tests main()'s control flow using mocked EmbedGenerator and
    ChromaIngestor — verifies the script wires arguments through
    correctly without needing real models or a real ChromaDB instance.
    """

    def _run_main(self, argv, mock_chunks):
        with patch.object(sys, "argv", ["run_ingestion.py"] + argv), \
             patch.object(run_ingestion, "load_all_chunks", return_value=mock_chunks), \
             patch.object(run_ingestion, "EmbedGenerator") as MockGen, \
             patch.object(run_ingestion, "ChromaIngestor") as MockIngestor:

            mock_gen_instance = MockGen.return_value
            mock_gen_instance.embed_chunks.return_value = []

            mock_ingestor_instance = MockIngestor.return_value
            mock_summary = MagicMock()
            mock_summary.ingested_count = len(mock_chunks)
            mock_summary.skipped_no_vector = 0
            mock_summary.collection_count_after = len(mock_chunks)
            mock_ingestor_instance.ingest_chunks.return_value = mock_summary

            try:
                run_ingestion.main()
            except SystemExit as e:
                return e.code, MockGen, MockIngestor
        return None, MockGen, MockIngestor

    def test_no_chunks_found_exits_with_error(self):
        with patch.object(sys, "argv", ["run_ingestion.py", "--chunks-dir", "/empty"]), \
             patch.object(run_ingestion, "load_all_chunks", return_value=[]):
            with self.assertRaises(SystemExit) as cm:
                run_ingestion.main()
            self.assertEqual(cm.exception.code, 1)

    def test_success_exits_zero(self):
        code, _, _ = self._run_main(
            ["--quiet"],
            [{"chunk_id": "a", "embed_text": "passage: test"}],
        )
        self.assertEqual(code, 0)

    def test_embed_generator_receives_model_arg(self):
        _, MockGen, _ = self._run_main(
            ["--model", "custom-model", "--quiet"],
            [{"chunk_id": "a", "embed_text": "passage: test"}],
        )
        call_args = MockGen.call_args[0][0]  # EmbedGeneratorConfig instance
        self.assertEqual(call_args.model_name, "custom-model")

    def test_chroma_ingestor_receives_dir_and_collection(self):
        _, _, MockIngestor = self._run_main(
            ["--chroma-dir", "/my/chroma", "--collection", "custom_coll", "--quiet"],
            [{"chunk_id": "a", "embed_text": "passage: test"}],
        )
        call_args = MockIngestor.call_args[0][0]  # ChromaIngestionConfig instance
        self.assertEqual(call_args.persist_directory, "/my/chroma")
        self.assertEqual(call_args.collection_name, "custom_coll")

    def test_batch_size_passed_to_embed_config(self):
        _, MockGen, _ = self._run_main(
            ["--batch-size", "8", "--quiet"],
            [{"chunk_id": "a", "embed_text": "passage: test"}],
        )
        call_args = MockGen.call_args[0][0]
        self.assertEqual(call_args.batch_size, 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
