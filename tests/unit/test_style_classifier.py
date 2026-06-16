import unittest
from core.utils.style_classifier import StyleClassifier

class TestStyleClassifier(unittest.TestCase):
    def test_explicit_outline_level(self):
        is_heading, level = StyleClassifier.classify("Normal", "Some text", outline_level=0)
        self.assertTrue(is_heading)
        self.assertEqual(level, 1)

    def test_keyword_match(self):
        is_heading, level = StyleClassifier.classify("Heading 2", "Some text")
        self.assertTrue(is_heading)
        self.assertEqual(level, 2)

    def test_heuristic_short_no_punctuation(self):
        is_heading, level = StyleClassifier.classify("Normal", "SERVICE Great circle")
        self.assertTrue(is_heading)
        self.assertEqual(level, 2)

    def test_heuristic_fails_on_long_text(self):
        is_heading, level = StyleClassifier.classify("Normal", "This is a very long paragraph that exceeds the heuristic length limit and should not be classified as a heading.")
        self.assertFalse(is_heading)
        self.assertEqual(level, 0)

    def test_heuristic_fails_on_punctuation(self):
        is_heading, level = StyleClassifier.classify("Normal", "This is a sentence.")
        self.assertFalse(is_heading)
        self.assertEqual(level, 0)

if __name__ == "__main__":
    unittest.main()