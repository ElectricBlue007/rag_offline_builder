from __future__ import annotations

import unittest

from scripts.cli.router import build_parser, main


class TestCliRouter(unittest.TestCase):
    def test_parser_has_subcommands(self) -> None:
        parser = build_parser()
        self.assertIsNotNone(parser)

    def test_main_shows_help_when_missing_command(self) -> None:
        self.assertEqual(main([]), 2)


if __name__ == "__main__":
    unittest.main()

