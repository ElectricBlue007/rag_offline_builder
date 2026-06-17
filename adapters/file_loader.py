"""File loading abstraction for raw Markdown inputs."""

from __future__ import annotations

from pathlib import Path

from rag_doc_compiler.errors import CompilerError


class FileLoader:
    def load_text(self, path: str | Path) -> str:
        source = Path(path)
        if not source.exists():
            raise CompilerError(f"Input file does not exist: {source}")
        if not source.is_file():
            raise CompilerError(f"Input path is not a file: {source}")
        return source.read_text(encoding="utf-8")
