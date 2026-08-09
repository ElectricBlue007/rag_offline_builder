"""
tests/test_eval_dataset.py

Unit tests for eval_dataset.py — pure JSON load/save logic, no
external dependencies.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_pipeline.evaluation.eval_dataset import (
    EvalQuestion,
    load_eval_questions,
    save_eval_questions,
)


class TestEvalQuestion(unittest.TestCase):

    def test_default_language_is_en(self):
        q = EvalQuestion(question="What is X?", ground_truth="X is Y.")
        self.assertEqual(q.language, "en")

    def test_default_category_is_empty(self):
        q = EvalQuestion(question="What is X?", ground_truth="X is Y.")
        self.assertEqual(q.category, "")

    def test_all_fields_settable(self):
        q = EvalQuestion(
            question="What is X?", ground_truth="X is Y.",
            language="es", category="table",
        )
        self.assertEqual(q.language, "es")
        self.assertEqual(q.category, "table")


class TestSaveAndLoadRoundtrip(unittest.TestCase):

    def test_roundtrip_preserves_all_fields(self):
        questions = [
            EvalQuestion(question="Q1", ground_truth="A1", language="en", category="table"),
            EvalQuestion(question="Q2", ground_truth="A2", language="es", category="prose"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"
            save_eval_questions(questions, path)
            loaded = load_eval_questions(path)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].question, "Q1")
        self.assertEqual(loaded[0].language, "en")
        self.assertEqual(loaded[1].category, "prose")

    def test_save_creates_parent_directories(self):
        questions = [EvalQuestion(question="Q", ground_truth="A")]
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "a" / "b" / "eval.json"
            save_eval_questions(questions, nested)
            self.assertTrue(nested.exists())

    def test_save_produces_valid_json_list(self):
        questions = [EvalQuestion(question="Q", ground_truth="A")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"
            save_eval_questions(questions, path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsInstance(data, list)
            self.assertEqual(data[0]["question"], "Q")

    def test_unicode_preserved_roundtrip(self):
        questions = [EvalQuestion(
            question="\u00bfQu\u00e9 es un CMI?", ground_truth="Un CMI es...", language="es",
        )]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"
            save_eval_questions(questions, path)
            loaded = load_eval_questions(path)
        self.assertEqual(loaded[0].question, "\u00bfQu\u00e9 es un CMI?")

    def test_empty_list_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"
            save_eval_questions([], path)
            loaded = load_eval_questions(path)
        self.assertEqual(loaded, [])


class TestSampleEvalQuestions(unittest.TestCase):
    """Validates the shipped starter evaluation set."""

    def _get_sample_path(self) -> Path:
        return Path(__file__).parent.parent / "evaluation" / "sample_eval_questions.json"

    def test_sample_file_exists(self):
        self.assertTrue(self._get_sample_path().exists())

    def test_sample_file_loads_without_error(self):
        questions = load_eval_questions(self._get_sample_path())
        self.assertGreater(len(questions), 0)

    def test_sample_has_at_least_ten_questions(self):
        questions = load_eval_questions(self._get_sample_path())
        self.assertGreaterEqual(len(questions), 10)

    def test_sample_includes_both_languages(self):
        questions = load_eval_questions(self._get_sample_path())
        languages = {q.language for q in questions}
        self.assertIn("en", languages)
        self.assertIn("es", languages)

    def test_sample_includes_multiple_categories(self):
        questions = load_eval_questions(self._get_sample_path())
        categories = {q.category for q in questions}
        self.assertGreaterEqual(len(categories), 2)

    def test_sample_all_questions_have_ground_truth(self):
        questions = load_eval_questions(self._get_sample_path())
        for q in questions:
            self.assertTrue(q.ground_truth.strip())

    def test_sample_all_questions_nonempty(self):
        questions = load_eval_questions(self._get_sample_path())
        for q in questions:
            self.assertTrue(q.question.strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
