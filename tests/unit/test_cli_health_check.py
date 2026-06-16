from __future__ import annotations

import unittest

from scripts.cli.health_check import health_check


class TestCliHealthCheck(unittest.TestCase):
    def test_health_check_returns_ok(self) -> None:
        payload = health_check()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["module"], "cli")


if __name__ == "__main__":
    unittest.main()

