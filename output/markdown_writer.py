"""Strict Markdown writer."""

from __future__ import annotations

from pathlib import Path


class MarkdownWriter:
    def write(self, markdown: str, output_dir: str | Path, filename: str = "clean.md") -> Path:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        path.write_text(markdown, encoding="utf-8")
        return path
