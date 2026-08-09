"""
tests/test_token_counter.py

Unit tests for TokenCounter — covering the heuristic fallback path
(since `transformers` is not installed in this environment, exactly
the situation a fully-offline deployment may be in) and the interface
contract that semantic_chunker.py depends on.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_pipeline.splitters.token_counter import TokenCounter


class TestTokenCounterHeuristicFallback(unittest.TestCase):
    """
    Tests the heuristic path deterministically via attempt_real_tokenizer=
    False — NOT by relying on transformers being absent from the
    environment. That assumption held in the original sandbox (no
    transformers installed) but does NOT hold universally: a real
    deployment may well have transformers installed as a transitive
    dependency of something else (sentence-transformers, langchain,
    etc.), which would silently switch these tests onto the real-
    tokenizer code path and break assertions written against the
    heuristic's specific math. Forcing the flag makes the test outcome
    independent of what happens to be installed.
    """

    def setUp(self):
        self.counter = TokenCounter(
            model_name="intfloat/multilingual-e5-large",
            attempt_real_tokenizer=False,
        )

    def test_empty_string_returns_zero(self):
        self.assertEqual(self.counter.count_tokens(""), 0)

    def test_single_word(self):
        result = self.counter.count_tokens("hello")
        self.assertGreaterEqual(result, 1)

    def test_longer_text_returns_more_tokens_than_shorter(self):
        short = self.counter.count_tokens("A short sentence.")
        long = self.counter.count_tokens(
            "A much longer sentence with many more words in it than the short one."
        )
        self.assertGreater(long, short)

    def test_heuristic_scales_with_word_count(self):
        """Doubling word count should roughly double the token estimate."""
        ten_words = self.counter.count_tokens("one two three four five six seven eight nine ten")
        twenty_words = self.counter.count_tokens(
            "one two three four five six seven eight nine ten "
            "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
        )
        self.assertAlmostEqual(twenty_words / ten_words, 2.0, delta=0.3)

    def test_using_real_tokenizer_false_when_disabled(self):
        """With attempt_real_tokenizer=False, using_real_tokenizer must be
        False regardless of whether transformers is actually installed —
        this is what the flag is for."""
        self.assertFalse(self.counter.using_real_tokenizer)

    def test_load_attempted_only_once(self):
        """Repeated calls must not retry the (failing) import every time."""
        _ = self.counter.count_tokens("first call")
        self.assertTrue(self.counter._load_attempted)
        with patch("builtins.__import__") as mock_import:
            _ = self.counter.count_tokens("second call")
            # transformers import should NOT be attempted again
            for call in mock_import.call_args_list:
                self.assertNotEqual(call[0][0], "transformers")

    def test_whitespace_only_text(self):
        result = self.counter.count_tokens("   ")
        # "".split() on whitespace-only text is [] → 0 words → 0 tokens
        self.assertEqual(result, 0)

    def test_count_batch_matches_individual_calls(self):
        texts = ["one", "two words here", "three word text here now"]
        batch_result = self.counter.count_batch(texts)
        individual_result = [self.counter.count_tokens(t) for t in texts]
        self.assertEqual(batch_result, individual_result)

    def test_count_batch_empty_list(self):
        self.assertEqual(self.counter.count_batch([]), [])

    def test_attempt_real_tokenizer_false_skips_import(self):
        counter = TokenCounter(
            model_name="intfloat/multilingual-e5-large",
            attempt_real_tokenizer=False,
        )
        with patch("builtins.__import__") as mock_import:
            counter.count_tokens("some text")
            for call in mock_import.call_args_list:
                self.assertNotEqual(call[0][0], "transformers")
        self.assertFalse(counter.using_real_tokenizer)

    def test_spanish_text_counted(self):
        """Non-English text must not crash the heuristic path."""
        result = self.counter.count_tokens(
            "Definición del botón de conjunto de trabajo nueva posición"
        )
        self.assertGreater(result, 0)


class TestTokenCounterWithMockedTransformers(unittest.TestCase):
    """
    Simulate an environment where `transformers` IS installed, using a
    mock tokenizer, to verify TokenCounter correctly delegates to it
    and reports using_real_tokenizer=True.
    """

    def _make_mock_tokenizer_module(self, encode_return_len: int = 7):
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.encode.return_value = list(range(encode_return_len))

        mock_auto_tokenizer_cls = MagicMock()
        mock_auto_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance

        mock_transformers_module = MagicMock()
        mock_transformers_module.AutoTokenizer = mock_auto_tokenizer_cls
        return mock_transformers_module, mock_tokenizer_instance

    def test_real_tokenizer_used_when_available(self):
        mock_module, mock_tok = self._make_mock_tokenizer_module(encode_return_len=5)
        counter = TokenCounter(model_name="intfloat/multilingual-e5-large")

        with patch.dict(sys.modules, {"transformers": mock_module}):
            result = counter.count_tokens("any text")

        self.assertEqual(result, 5)
        self.assertTrue(counter.using_real_tokenizer)

    def test_real_tokenizer_encode_called_without_special_tokens(self):
        mock_module, mock_tok = self._make_mock_tokenizer_module()
        counter = TokenCounter(model_name="intfloat/multilingual-e5-large")

        with patch.dict(sys.modules, {"transformers": mock_module}):
            counter.count_tokens("some text")

        mock_tok.encode.assert_called_once_with("some text", add_special_tokens=False)

    def test_tokenizer_load_failure_falls_back_to_heuristic(self):
        mock_module = MagicMock()
        mock_module.AutoTokenizer.from_pretrained.side_effect = OSError("no network")
        counter = TokenCounter(model_name="intfloat/multilingual-e5-large")

        with patch.dict(sys.modules, {"transformers": mock_module}):
            result = counter.count_tokens("hello world")

        self.assertFalse(counter.using_real_tokenizer)
        self.assertGreater(result, 0)  # heuristic still produces a count

    def test_tokenizer_encode_failure_falls_back_to_heuristic_per_call(self):
        """If the loaded tokenizer raises on a specific input, don't crash."""
        mock_module, mock_tok = self._make_mock_tokenizer_module()
        mock_tok.encode.side_effect = RuntimeError("bad input")
        counter = TokenCounter(model_name="intfloat/multilingual-e5-large")

        with patch.dict(sys.modules, {"transformers": mock_module}):
            result = counter.count_tokens("hello world")

        self.assertGreater(result, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
