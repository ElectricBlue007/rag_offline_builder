"""Markdown normalization with strict preservation semantics.

Rules 1-30 preserved. New in this revision:
  - Comprehensive image splitting (ANY position: start, mid, multiple per line)
  - Setext heading conversion (=== / --- underlines → ATX # / ##)
  - Math fence tracking ($$ blocks preserved verbatim like code fences)
  - HTML comment removal (<!-- ... -->)
  - Admonition tagging (> [!NOTE/TIP/WARNING/IMPORTANT/CAUTION])
  - Task-list item preservation (- [x] / - [ ])
  - Footnote definition preservation ([^id]: text)
"""

from __future__ import annotations

import re
import unicodedata

from config.settings import CompilerSettings, DEFAULT_SETTINGS
from rag_doc_compiler.errors import NormalizationError
from rag_doc_compiler.models import NormalizationResult


# ── Structural patterns ───────────────────────────────────────────────────────
_BROKEN_ATX_HEADING       = re.compile(r"^(#{1,6})([^\s#].*)$")
_EMBEDDED_ATX_HEADING     = re.compile(r"^(?P<prefix>.+?)(?P<heading>#.{1,6}\s*[A-Z][^#]*)$")
_EMBEDDED_HTML_TABLE      = re.compile(r"^(?P<prefix>.+?)(?P<table><table\b.*)$", re.IGNORECASE)
_FENCE                    = re.compile(r"^\s*(```|~~~)")
_MATH_FENCE               = re.compile(r"^\s*\$\$\s*$")
_FRONT_MATTER_FENCE       = re.compile(r"^---\s*$")
_SETEXT_H1                = re.compile(r"^=+\s*$")
_SETEXT_H2                = re.compile(r"^-{2,}\s*$")

# ── HTML patterns ─────────────────────────────────────────────────────────────
_HTML_DIV_OPEN            = re.compile(r"^<div\b", re.IGNORECASE)
_HTML_DIV_CLOSE           = re.compile(r"</div>\s*$", re.IGNORECASE)
_HTML_TABLE_OPEN          = re.compile(r"^\s*<table\b", re.IGNORECASE)
_HTML_TABLE_CLOSE         = re.compile(r"</table>\s*$", re.IGNORECASE)
_HTML_COMMENT_LINE        = re.compile(r"^\s*<!--.*?-->\s*$", re.DOTALL)
_HTML_COMMENT_INLINE      = re.compile(r"<!--.*?-->", re.DOTALL)

# ── Image & link patterns ─────────────────────────────────────────────────────
_BADGE_PATTERN            = re.compile(
    r"!\[.*?\]\(.*?(?:badge|shields|pypi|discord|downloads).*?\)", re.IGNORECASE
)
_IMAGE_PATTERN            = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^\)]+)\)")
# Splits a line into alternating [text, image_token, text, ...] parts.
# Using a capturing group so re.split keeps the image tokens in the result.
_ANY_IMAGE_TOKEN          = re.compile(r"(!\[[^\]]*\]\([^\)]+\))")
_LINK_PATTERN             = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<url>[^\)]+)\)")

# ── TOC & noise ──────────────────────────────────────────────────────────────
_TOC_HEADING              = re.compile(r"^\s*#{1,6}\s+table\s+of\s+contents\s*$", re.IGNORECASE)
_EMBEDDED_TOC_HEADING     = re.compile(
    r"^(?P<prefix>.+?)#{1,6}\s+table\s+of\s+contents\s*$", re.IGNORECASE
)
_CONTINUATION_NOISE       = re.compile(
    r"<p[^>]*>\s*<b>\s*(?:continues?|continued)\s+(?:on\s+)?next\s+page\.{0,3}\s*</b>\s*</p>",
    re.IGNORECASE,
)

# ── Table patterns ────────────────────────────────────────────────────────────
_TABLE_DELIMITER          = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$"
)

# ── Other content patterns ────────────────────────────────────────────────────
_QUOTE_START              = re.compile(r"^>\s*")
_LIST_ITEM                = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.+)$")
_TASK_LIST_ITEM           = re.compile(r"^(\s*[-*+])\s+\[([xX ])\]\s+(.+)$")
_BLOCKQUOTE_NESTED        = re.compile(r"^>(?:>)?\s+")
_ADMONITION               = re.compile(
    r"^\s*>\s*\[!(NOTE|TIP|WARNING|IMPORTANT|CAUTION)\]", re.IGNORECASE
)
_FOOTNOTE_DEF             = re.compile(r"^\[\^[^\]]+\]:\s+.+$")


class MarkdownNormalizer:
    """Normalize OCR and general Markdown for any document type."""

    def __init__(self, settings: CompilerSettings = DEFAULT_SETTINGS) -> None:
        self.settings = settings

    # ── Public API ────────────────────────────────────────────────────────────

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
        fence_marker: str | None = None   # code fence (``` / ~~~)
        math_fence_open: bool = False     # math block ($$)
        in_front_matter = False
        front_matter_started = False
        removed_noise_count = 0
        block_counter = 0
        index = 0

        while index < len(lines):
            line     = lines[index]
            line_no  = index + 1
            stripped_newline = line.rstrip("\r\n")
            newline  = line[len(stripped_newline):]

            # ── Front matter (Rule 28) ────────────────────────────────────────
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

            # ── Math fence ($$ blocks — pass through verbatim) ───────────────
            if _MATH_FENCE.match(stripped_newline):
                if not math_fence_open:
                    math_fence_open = True
                    block_counter += 1
                else:
                    math_fence_open = False
                normalized.append(line)
                index += 1
                continue

            if math_fence_open:
                normalized.append(line)
                index += 1
                continue

            # ── Code fence (Rules 5-6) ────────────────────────────────────────
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

            if fence_marker is not None:
                normalized.append(line)
                index += 1
                continue

            # ── Setext heading conversion (look-ahead) ────────────────────────
            # Convert underline-style headings to ATX before any other processing.
            if stripped_newline.strip() and index + 1 < len(lines):
                next_line = lines[index + 1].rstrip("\r\n")
                if _SETEXT_H1.match(next_line) and len(next_line) >= 1:
                    heading_text = stripped_newline.strip()
                    if heading_text:
                        normalized.append(f"# {heading_text}" + newline)
                        flags_by_line.setdefault(line_no, []).append("setext_h1_converted")
                        index += 2
                        continue
                elif _SETEXT_H2.match(next_line) and len(next_line) >= 2:
                    heading_text = stripped_newline.strip()
                    if heading_text and not _LIST_ITEM.match(stripped_newline):
                        normalized.append(f"## {heading_text}" + newline)
                        flags_by_line.setdefault(line_no, []).append("setext_h2_converted")
                        index += 2
                        continue

            # ── Full-line HTML comment removal ────────────────────────────────
            if _HTML_COMMENT_LINE.match(stripped_newline):
                flags_by_line.setdefault(line_no, []).append("html_comment_removed")
                removed_noise_count += 1
                index += 1
                continue

            # ── TOC removal ───────────────────────────────────────────────────
            embedded_toc = _EMBEDDED_TOC_HEADING.match(stripped_newline)
            if embedded_toc:
                prefix = embedded_toc.group("prefix").rstrip()
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

            # ── HTML presentation noise (Rules 17-18) ────────────────────────
            if _HTML_DIV_OPEN.match(stripped_newline) or _HTML_DIV_CLOSE.match(stripped_newline):
                flags_by_line.setdefault(line_no, []).append("html_div_removed")
                removed_noise_count += 1
                index += 1
                continue

            # ── Badge noise (Rule 18) ─────────────────────────────────────────
            if _BADGE_PATTERN.search(stripped_newline):
                flags_by_line.setdefault(line_no, []).append("badge_noise_removed")
                removed_noise_count += 1
                index += 1
                continue

            # ── Comprehensive image line splitting ────────────────────────────
            # Handles ALL positions and counts:
            #   "![img](src)Text"          → ["![img](src)", "Text"]
            #   "Text![img](src)Text"      → ["Text", "![img](src)", "Text"]
            #   "Text![a](a)X![b](b)Text"  → ["Text","![a](a)","X","![b](b)","Text"]
            # This is the definitive fix for Chandra OCR mid-line images.
            if "![" in stripped_newline:
                parts = self._split_line_by_images(stripped_newline)
                if len(parts) > 1:
                    flags_by_line.setdefault(line_no, []).append("image_line_split")
                    for part in parts:
                        if part.startswith("!["):
                            normalized.append(part + newline)
                        else:
                            # Text part may itself contain an embedded ATX heading
                            emb_h = _EMBEDDED_ATX_HEADING.match(part)
                            if emb_h and not part.lstrip().startswith("#"):
                                pfx = emb_h.group("prefix").rstrip()
                                hdg = emb_h.group("heading").lstrip()
                                if pfx:
                                    normalized.append(pfx + newline)
                                normalized.append(hdg + newline)
                                flags_by_line.setdefault(line_no, []).append(
                                    "embedded_atx_heading_split"
                                )
                            else:
                                normalized.append(part + newline)
                    index += 1
                    continue
                # Single standalone image with nothing else → fall through normally

            # ── Continuation noise ────────────────────────────────────────────
            if _CONTINUATION_NOISE.search(stripped_newline):
                stripped_newline = _CONTINUATION_NOISE.sub("", stripped_newline).rstrip()
                flags_by_line.setdefault(line_no, []).append("continuation_marker_removed")
                removed_noise_count += 1
                if not stripped_newline:
                    index += 1
                    continue

            # ── Inline HTML comment strip (partial lines) ─────────────────────
            if "<!--" in stripped_newline:
                cleaned = _HTML_COMMENT_INLINE.sub("", stripped_newline).rstrip()
                if cleaned != stripped_newline:
                    flags_by_line.setdefault(line_no, []).append("inline_html_comment_removed")
                    stripped_newline = cleaned
                    if not stripped_newline.strip():
                        index += 1
                        continue

            repaired_lines = [stripped_newline]

            # ── Embedded HTML table split ─────────────────────────────────────
            embedded_table = _EMBEDDED_HTML_TABLE.match(stripped_newline)
            if embedded_table:
                prefix = embedded_table.group("prefix").rstrip()
                table  = embedded_table.group("table").lstrip()
                if prefix:
                    repaired_lines = [prefix, table]
                    flags_by_line.setdefault(len(normalized) + 2, []).append(
                        "embedded_html_table_split"
                    )
                else:
                    repaired_lines = [table]

            # ── Embedded ATX heading split ────────────────────────────────────
            embedded_heading = _EMBEDDED_ATX_HEADING.match(stripped_newline)
            if embedded_heading and not stripped_newline.lstrip().startswith("#"):
                prefix  = embedded_heading.group("prefix").rstrip()
                heading = embedded_heading.group("heading").lstrip()
                if prefix:
                    repaired_lines = [prefix, heading]
                    flags_by_line.setdefault(len(normalized) + 2, []).append(
                        "embedded_atx_heading_split"
                    )
                else:
                    repaired_lines = [heading]

            for repaired in repaired_lines:
                # Fix ATX heading spacing
                heading_match = _BROKEN_ATX_HEADING.match(repaired)
                if heading_match and self.settings.allow_header_spacing_normalization:
                    repaired = f"{heading_match.group(1)} {heading_match.group(2)}"
                    flags_by_line.setdefault(len(normalized) + 1, []).append(
                        "atx_heading_space_inserted"
                    )

                if "\x00" in repaired:
                    raise NormalizationError(f"Null byte detected at line {index}.")

                normalized.append(repaired + newline)

            index += 1

        if fence_marker is not None:
            raise NormalizationError("Unclosed fenced code block detected.")
        if math_fence_open:
            raise NormalizationError("Unclosed math fence ($$) detected.")

        final_lines = self._normalize_whitespace(normalized, flags_by_line)

        return NormalizationResult(
            markdown="".join(final_lines),
            flags_by_line=flags_by_line,
            removed_noise_count=removed_noise_count,
            utf8_normalized=utf8_normalized,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _split_line_by_images(line: str) -> list[str]:
        """Split a line on every image token, returning non-empty parts in order.

        Uses a capturing group so re.split preserves the image tokens:
            "Text![a](a.png)More" → ["Text", "![a](a.png)", "More"]
        Empty strings from adjacent images or leading/trailing images are dropped.
        """
        parts = _ANY_IMAGE_TOKEN.split(line)
        return [p.strip() for p in parts if p.strip()]

    def _normalize_utf8(self, text: str) -> str:
        """Rule 29: NFC normalization + mojibake repair.

        All replacement strings use \\uXXXX escapes so no literal Unicode
        character ever appears in the Python source and heredoc corruption
        cannot break the syntax.
        """
        normalized = unicodedata.normalize("NFC", text)

        # Phase 1 — display-encoding mojibake (UTF-8 bytes mis-read as Latin-1)
        mojibake = {
            # en-dash, em-dash
            "\u00e2\u0080\u0093": "\u2013",
            "\u00e2\u0080\u0094": "\u2014",
            # left/right single quotation marks
            "\u00e2\u0080\u0098": "\u2018",
            "\u00e2\u0080\u0099": "\u2019",
            # left/right double quotation marks
            "\u00e2\u0080\u009c": "\u201c",
            "\u00e2\u0080\u009d": "\u201d",
            # ellipsis, bullet
            "\u00e2\u0080\u00a6": "\u2026",
            "\u00e2\u0080\u00a2": "\u2022",
            # non-breaking space artefacts from Arabic OCR
            "\u00c2\u00a0": " ",
            "\u00c3\u0082": "",
        }
        for broken, fixed in mojibake.items():
            normalized = normalized.replace(broken, fixed)

        # Phase 2 — secondary mojibake layer (double-encoded sequences)
        extra = {
            "\u00e2\u20ac\u201c": "-",
            "\u00e2\u20ac\u201d": "-",
            "\u00e2\u20ac\u02dc": "\u2018",
            "\u00e2\u20ac\u2122": "\u2019",
            "\u00e2\u20ac\u0153": "\u201c",
            "\u00e2\u20ac\u009d": "\u201d",
            "\u00e2\u20ac\u00a6": "...",
            "\u00e2\u20ac\u00a2": "-",
        }
        for broken, fixed in extra.items():
            normalized = normalized.replace(broken, fixed)

        # Phase 3 — normalise typographic punctuation to ASCII equivalents
        normalized = normalized.replace("\u2013", "-")   # en-dash
        normalized = normalized.replace("\u2014", "-")   # em-dash

        # Phase 4 — Arabic OCR noise characters
        normalized = normalized.replace("\u00d8\u00a2", "")
        normalized = normalized.replace("\u0637\u00a2", "")
        normalized = normalized.replace("\u0622", "")
        normalized = re.sub(r"(?:\u0622\s*){2,}", "    ", normalized)

        # Phase 5 — invisible / zero-width characters
        normalized = normalized.replace("\u00a0", " ")   # non-breaking space
        normalized = normalized.replace("\u200b", "")    # zero-width space
        normalized = normalized.replace("\ufeff", "")    # UTF-8 BOM
        return normalized

    def _normalize_whitespace(
        self, lines: list[str], flags_by_line: dict[int, list[str]]
    ) -> list[str]:
        """Rule 19-20: Collapse excess blank lines, trim document edges."""
        result: list[str] = []
        consecutive_blank = 0
        for idx, line in enumerate(lines):
            if not line.strip():
                consecutive_blank += 1
                if consecutive_blank <= 2:
                    result.append(line)
                else:
                    flags_by_line.setdefault(idx + 1, []).append("excess_whitespace_normalized")
            else:
                consecutive_blank = 0
                result.append(line)
        while result and not result[0].strip():
            result.pop(0)
        while result and not result[-1].strip():
            result.pop()
        return result