from __future__ import annotations

import json
import logging
import unittest

from core.logging import JsonFormatter, configure_logging


class TestLogging(unittest.TestCase):
    def test_json_formatter(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello", (), None)
        record.event = "start"
        record.extra_data = {"file": "sample.docx"}
        payload = json.loads(formatter.format(record))
        self.assertEqual(payload["event"], "start")
        self.assertEqual(payload["extra"]["file"], "sample.docx")

    def test_configure_logging(self) -> None:
        logger = configure_logging()
        self.assertEqual(logger.name, "rag_offline_builder")


if __name__ == "__main__":
    unittest.main()

