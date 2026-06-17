"""Compatibility adapter for Chandra OCR Hugging Face Markdown output."""

from __future__ import annotations

from collections.abc import Mapping

from rag_doc_compiler.errors import CompilerError


class OCRInputAdapter:
    """Extract Markdown from common OCR output shapes without semantic edits."""

    def to_markdown(self, payload: str | Mapping[str, object]) -> str:
        if isinstance(payload, str):
            return payload
        for key in ("markdown", "text", "content"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        raise CompilerError("OCR payload must be a string or contain markdown/text/content.")
