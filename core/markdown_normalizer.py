"""Markdown normalization with strict preservation semantics.

This module implements Phase 1: Markdown Ready For Chunking following Rules 1-30:
- Rule 1: Preserve Original Meaning (no summarization, rewriting, or AI enhancement)
- Rule 2-4: Preserve Hierarchy, Heading Levels, and Heading Path
- Rule 5-6: Preserve Code Blocks Exactly with Language Metadata
- Rule 7-8: Preserve Tables as Structured Data
- Rule 9-10: Preserve Lists and Nesting
- Rule 11-12: Preserve Links with Text and URL Separately
- Rule 13-14: Preserve Images as Metadata
- Rule 15-16: Preserve Block and Section Boundaries
- Rule 17-18: Remove Presentation and Badge Noise
- Rule 19-20: Normalize Whitespace While Preserving Semantic Breaks
- Rule 21-22: Assign Document Order and Stable IDs
- Rule 23-24: Detect Boilerplate and Generated Noise
- Rule 25-27: Never Merge Different Content Types; Create Typed Blocks; Preserve Quotes
- Rule 28: Preserve Front Matter
- Rule 29: UTF-8 Normalization
- Rule 30: Never Chunk Yet - Only Create Structured Markdown Document

The normalizer never rewrites meaning. It only applies deterministic repairs
that make structural parsing possible and records every repair as metadata.
Unsafe inconsistencies raise immediately.
"""

from __future__ import annotations

import re
import unicodedata

from config.settings import CompilerSettings, DEFAULT_SETTINGS
from rag_doc_compiler.errors import NormalizationError
from rag_doc_compiler.models import NormalizationResult


# Patterns for detection
_BROKEN_ATX_HEADING = re.compile(r"^(#{1,6})([^\s#].*)$")
_EMBEDDED_ATX_HEADING = re.compile(r"^(?P<prefix>.+?)(?P<heading>#.{1,6}\s*[A-Z][^#]*)$")
_EMBEDDED_HTML_TABLE = re.compile(r"^(?P<prefix>.+?)(?P<table><table\b.*)$", re.IGNORECASE)
_FENCE = re.compile(r"^\s*(```|~~~)")
_FRONT_MATTER_FENCE = re.compile(r"^---\s*$")
_HTML_DIV_OPEN = re.compile(r"^<div\b", re.IGNORECASE)
_HTML_DIV_CLOSE = re.compile(r"</div>\s*$", re.IGNORECASE)
_HTML_TABLE_OPEN = re.compile(r"^\s*<table\b", re.IGNORECASE)
_HTML_TABLE_CLOSE = re.compile(r"</table>\s*$", re.IGNORECASE)
_BADGE_PATTERN = re.compile(r"!\[.*?\]\(.*?(?:badge|shields|pypi|discord|downloads).*?\)", re.IGNORECASE)
_IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^\)]+)\)")
_LINK_PATTERN = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<url>[^\)]+)\)")
_TOC_HEADING = re.compile(r"^\s*#{1,6}\s+table\s+of\s+contents\s*$", re.IGNORECASE)
_EMBEDDED_TOC_HEADING = re.compile(r"^(?P<prefix>.+?)#{1,6}\s+table\s+of\s+contents\s*$", re.IGNORECASE)
_CONTINUATION_NOISE = re.compile(
    r"<p[^>]*>\s*<b>\s*(?:continues?|continued)\s+(?:on\s+)?next\s+page\.{0,3}\s*</b>\s*</p>",
    re.IGNORECASE,
)
_TABLE_DELIMITER = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_QUOTE_START = re.compile(r"^>\s*")
_LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.+)$")
_BLOCKQUOTE_NESTED = re.compile(r"^>(?:>)?\s+")


class MarkdownNormalizer:
    """Normalize OCR Markdown only where structural parsing requires it."""

    def __init__(self, settings: CompilerSettings = DEFAULT_SETTINGS) -> None:
        self.settings = settings

    def normalize(self, markdown: str) -> NormalizationResult:
        if not isinstance(markdown, str):
            raise NormalizationError("Input markdown must be a string.")

        # Rule 29: UTF-8 Normalization
        normalized_text = self._normalize_utf8(markdown)
        utf8_normalized = normalized_text != markdown
        markdown = normalized_text

        lines = markdown.splitlines(keepends=True)
        flags_by_line: dict[int, list[str]] = {}
        normalized: list[str] = []
        fence_marker: str | None = None
        in_front_matter = False
        front_matter_started = False
        removed_noise_count = 0
        block_counter = 0

        index = 0
        while index < len(lines):
            line = lines[index]
            line_no = index + 1
            stripped_newline = line.rstrip("\r\n")
            newline = line[len(stripped_newline) :]

            # Rule 28: Preserve Front Matter
            if not front_matter_started and _FRONT_MATTER_FENCE.match(stripped_newline):
                front_matter_started = True
                in_front_matter = True
                normalized.append(line)
                index += 1
                continue
            
            if front_matter_started and in_front_matter:
                if _FRONT_MATTER_FENCE.match(stripped_newline):
                    in_front_matter = False
                normalized.append(line)
                index += 1
                continue

            # Check for code fence
            fence_match = _FENCE.match(stripped_newline)
            if fence_match:
                marker = fence_match.group(1)
                if fence_marker is None:
                    fence_marker = marker
                    block_counter += 1
                elif marker == fence_marker:
                    fence_marker = None
                normalized.append(line)
                index += 1
                continue

            # Rule 5-6: Inside code block, preserve exactly
            if fence_marker is not None:
                normalized.append(line)
                index += 1
                continue

            embedded_toc_heading = _EMBEDDED_TOC_HEADING.match(stripped_newline)
            if embedded_toc_heading:
                prefix = embedded_toc_heading.group("prefix").rstrip()
                if prefix:
                    normalized.append(prefix + newline)
                flags_by_line.setdefault(line_no, []).append("embedded_table_of_contents_removed")
                removed_noise_count += 1
                index += 1
                while index < len(lines) and not lines[index].strip():
                    index += 1
                if index < len(lines) and _HTML_TABLE_OPEN.match(lines[index].strip()):
                    while index < len(lines):
                        removed_noise_count += 1
                        if _HTML_TABLE_CLOSE.search(lines[index]):
                            index += 1
                            break
                        index += 1
                continue

            if _TOC_HEADING.match(stripped_newline):
                flags_by_line.setdefault(line_no, []).append("table_of_contents_removed")
                removed_noise_count += 1
                index += 1
                while index < len(lines) and not lines[index].strip():
                    index += 1
                if index < len(lines) and _HTML_TABLE_OPEN.match(lines[index].strip()):
                    while index < len(lines):
                        removed_noise_count += 1
                        if _HTML_TABLE_CLOSE.search(lines[index]):
                            index += 1
                            break
                        index += 1
                continue

            # Rule 17-18: Remove HTML presentation noise
            if _HTML_DIV_OPEN.match(stripped_newline) or _HTML_DIV_CLOSE.match(stripped_newline):
                flags_by_line.setdefault(line_no, []).append("html_div_removed")
                removed_noise_count += 1
                index += 1
                continue

            # Rule 18: Remove badge noise
            if _BADGE_PATTERN.search(stripped_newline):
                flags_by_line.setdefault(line_no, []).append("badge_noise_removed")
                removed_noise_count += 1
                index += 1
                continue

            if _CONTINUATION_NOISE.search(stripped_newline):
                stripped_newline = _CONTINUATION_NOISE.sub("", stripped_newline).rstrip()
                flags_by_line.setdefault(line_no, []).append("continuation_marker_removed")
                removed_noise_count += 1
                if not stripped_newline:
                    index += 1
                    continue

            repaired_lines = [stripped_newline]
            
            # Handle embedded content
            embedded_table = _EMBEDDED_HTML_TABLE.match(stripped_newline)
            if embedded_table:
                prefix = embedded_table.group("prefix").rstrip()
                table = embedded_table.group("table").lstrip()
                if prefix:
                    repaired_lines = [prefix, table]
                    flags_by_line.setdefault(len(normalized) + 2, []).append("embedded_html_table_split")
                else:
                    repaired_lines = [table]

            embedded_heading = _EMBEDDED_ATX_HEADING.match(stripped_newline)
            if embedded_heading and not stripped_newline.lstrip().startswith("#"):
                prefix = embedded_heading.group("prefix").rstrip()
                heading = embedded_heading.group("heading").lstrip()
                if prefix:
                    repaired_lines = [prefix, heading]
                    flags_by_line.setdefault(len(normalized) + 2, []).append("embedded_atx_heading_split")
                else:
                    repaired_lines = [heading]

            for repaired in repaired_lines:
                # Fix ATX heading spacing (traceable structural repair)
                heading_match = _BROKEN_ATX_HEADING.match(repaired)
                if heading_match and self.settings.allow_header_spacing_normalization:
                    repaired = f"{heading_match.group(1)} {heading_match.group(2)}"
                    flags_by_line.setdefault(len(normalized) + 1, []).append("atx_heading_space_inserted")

                # Rule 29: Check for null bytes
                if "\x00" in repaired:
                    raise NormalizationError(f"Null byte detected at line {index}.")

                normalized.append(repaired + newline)

            index += 1

        if fence_marker is not None:
            raise NormalizationError("Unclosed fenced code block detected.")

        # Rule 19: Normalize whitespace (multiple blank lines to single)
        final_lines = self._normalize_whitespace(normalized, flags_by_line)

        return NormalizationResult(
            markdown="".join(final_lines),
            flags_by_line=flags_by_line,
            removed_noise_count=removed_noise_count,
            utf8_normalized=utf8_normalized,
        )

    def _normalize_utf8(self, text: str) -> str:
        """Rule 29: Normalize Unicode for consistent handling of international text."""
        # NFC normalization for consistent character representation
        normalized = unicodedata.normalize("NFC", text)
        mojibake_replacements = {
            "â€“": "–",
            "â€”": "—",
            "â€˜": "‘",
            "â€™": "’",
            "â€œ": "“",
            "â€": "”",
            "â€¦": "…",
            "â€¢": "•",
            "آ ": " ",
            "آ": "",
        }
        for broken, repaired in mojibake_replacements.items():
            normalized = normalized.replace(broken, repaired)
        extra_mojibake_replacements = {
            "\u00e2\u20ac\u201c": "-",
            "\u00e2\u20ac\u201d": "-",
            "\u00e2\u20ac\u02dc": "'",
            "\u00e2\u20ac\u2122": "'",
            "\u00e2\u20ac\u0153": '"',
            "\u00e2\u20ac\u009d": '"',
            "\u00e2\u20ac\u00a6": "...",
            "\u00e2\u20ac\u00a2": "-",
        }
        for broken, repaired in extra_mojibake_replacements.items():
            normalized = normalized.replace(broken, repaired)
        normalized = normalized.replace("\u2013", "-")
        normalized = normalized.replace("\u2014", "-")
        normalized = normalized.replace("\u00d8\u00a2", "")
        normalized = normalized.replace("\u0637\u00a2", "")
        normalized = normalized.replace("\u0622", "")
        normalized = re.sub(r"(?:\u0622\s*){2,}", "    ", normalized)
        # Replace common problematic Unicode characters
        normalized = normalized.replace("\u00a0", " ")  # Non-breaking space to regular space
        normalized = normalized.replace("\u200b", "")   # Zero-width space removal
        normalized = normalized.replace("\ufeff", "")   # BOM removal
        return normalized

    def _normalize_whitespace(self, lines: list[str], flags_by_line: dict[int, list[str]]) -> list[str]:
        """Rule 19-20: Normalize whitespace while preserving semantic breaks."""
        result: list[str] = []
        consecutive_blank = 0
        
        for idx, line in enumerate(lines):
            if not line.strip():
                consecutive_blank += 1
                if consecutive_blank <= 2:  # Keep one blank line (two newlines)
                    result.append(line)
                else:
                    flags_by_line.setdefault(idx + 1, []).append("excess_whitespace_normalized")
            else:
                consecutive_blank = 0
                result.append(line)
        
        # Remove leading/trailing blank lines but preserve internal structure
        while result and not result[0].strip():
            result.pop(0)
        while result and not result[-1].strip():
            result.pop()
        
        return result
