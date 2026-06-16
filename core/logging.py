"""Structured logging helpers for the pipeline."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "event"):
            payload["event"] = getattr(record, "event")
        if hasattr(record, "extra_data"):
            payload["extra"] = getattr(record, "extra_data")
        return json.dumps(payload, sort_keys=True, default=str)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("rag_offline_builder")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, level: int, event: str, message: str, **extra: Any) -> None:
    logger.log(level, message, extra={"event": event, "extra_data": extra})

