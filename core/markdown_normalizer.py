"""Markdown normalization with strict preservation semantics.

The normalizer never rewrites meaning. It only applies deterministic repairs
that make structural parsing possible and records every repair as metadata.
Unsafe inconsistencies raise immediately.
"""

from __future__ import annotations

import re

from config.settings import CompilerSettings, DEFAULT_SETTINGS
from rag_doc_compiler.errors import NormalizationError
from rag_doc_compiler.models import NormalizationResult


_BROKEN_ATX_HEADING = re.compile(r"^(#{1,6})([^\s#].*)$")
_EMBEDDED_ATX_HEADING = re.compile(r"^(?P<prefix>.+?)(?P<heading>#{1,6}\s*[A-Z][^#]*)$")
_EMBEDDED_HTML_TABLE = re.compile(r"^(?P<prefix>.+?)(?P<table><table\b.*)$", re.IGNORECASE)
_TRAILING_AFTER_HTML_TABLE = re.compile(r"^(?P<table>.*?</table>)(?P<trailing>.+)$", re.IGNORECASE)
_FENCE = re.compile(r"^\s*(```|~~~)")


class MarkdownNormalizer:
    """Normalize OCR Markdown only where structural parsing requires it."""

    def __init__(self, settings: CompilerSettings = DEFAULT_SETTINGS) -> None:
        self.settings = settings

    def normalize(self, markdown: str) -> NormalizationResult:
        if not isinstance(markdown, str):
            raise NormalizationError("Input markdown must be a string.")

        lines = markdown.splitlines(keepends=True)
        flags_by_line: dict[int, list[str]] = {}
        normalized: list[str] = []
        fence_marker: str | None = None

        for index, line in enumerate(lines, start=1):
            stripped_newline = line.rstrip("\r\n")
            newline = line[len(stripped_newline) :]

            fence_match = _FENCE.match(stripped_newline)
            if fence_match:
                marker = fence_match.group(1)
                if fence_marker is None:
                    fence_marker = marker
                elif marker == fence_marker:
                    fence_marker = None
                normalized.append(line)
                continue

            if fence_marker is not None:
                normalized.append(line)
                continue

            repaired_lines = [stripped_newline]
            embedded_heading = _EMBEDDED_ATX_HEADING.match(stripped_newline)
            embedded_table = _EMBEDDED_HTML_TABLE.match(stripped_newline)
            trailing_after_table = _TRAILING_AFTER_HTML_TABLE.match(stripped_newline)
            if trailing_after_table:
                table = trailing_after_table.group("table").rstrip()
                trailing = trailing_after_table.group("trailing").lstrip()
                repaired_lines = [table, trailing]
                flags_by_line.setdefault(len(normalized) + 2, []).append("trailing_after_html_table_split")
            elif embedded_table:
                prefix = embedded_table.group("prefix").rstrip()
                table = embedded_table.group("table").lstrip()
                repaired_lines = [prefix, table]
                flags_by_line.setdefault(len(normalized) + 2, []).append("embedded_html_table_split")
            elif embedded_heading and not stripped_newline.lstrip().startswith("#"):
                prefix = embedded_heading.group("prefix").rstrip()
                heading = embedded_heading.group("heading").lstrip()
                repaired_lines = [prefix, heading]
                flags_by_line.setdefault(len(normalized) + 2, []).append("embedded_atx_heading_split")

            for repaired in repaired_lines:
                heading_match = _BROKEN_ATX_HEADING.match(repaired)
                if heading_match and self.settings.allow_header_spacing_normalization:
                    repaired = f"{heading_match.group(1)} {heading_match.group(2)}"
                    flags_by_line.setdefault(len(normalized) + 1, []).append("atx_heading_space_inserted")

                if "\x00" in repaired:
                    raise NormalizationError(f"Null byte detected at line {index}.")

                normalized.append(repaired + newline)

        if fence_marker is not None:
            raise NormalizationError("Unclosed fenced code block detected.")

        return NormalizationResult(markdown="".join(normalized), flags_by_line=flags_by_line)
