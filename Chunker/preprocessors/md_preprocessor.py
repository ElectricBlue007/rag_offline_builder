"""
preprocessors/md_preprocessor.py

Cleans raw Chandra OCR markdown output before any splitting happens.

Problems it solves (observed in real output files):
  1. Image alt-texts glued directly to surrounding prose
  2. Two <table> blocks with no separator between them
  3. Table-of-contents tables (detected by "....." pattern)
  4. HTML superscripts  <sup>d</sup>  <sup>7</sup>
  5. Decorative lines  "Continues next page…"
  6. .webp image references (filenames, not content)
"""

import re
from dataclasses import dataclass
from typing import Optional

from Chunker.config import PreprocessorConfig


# ── Regex patterns ────────────────────────────────────────────────────────────

# Matches  ![alt text](filename.webp)  — captures alt text separately
_RE_IMAGE = re.compile(
    r'!\[([^\]]*)\]\([^)]*\.webp\)',
    re.IGNORECASE,
)

# Matches a closing </table> immediately followed by <table  (no blank line)
_RE_ADJACENT_TABLES = re.compile(
    r'(</table>)\s*(<table)',
    re.IGNORECASE | re.DOTALL,
)

# TOC tables contain "....." (page-reference dots).
# Threshold is 4+ dots specifically to avoid false-positives on a plain
# ellipsis "..." (3 dots), which appears inside <pre>/code blocks in real
# OCR output (e.g. the pizza-order JSON example in Testing.md).
_RE_TOC_DOTS = re.compile(r'\.{4,}')

# HTML superscripts
_RE_SUPERSCRIPT = re.compile(r'<sup>[^<]*</sup>', re.IGNORECASE)

# Decorative / boilerplate lines
_DECORATIVE_PATTERNS = [
    re.compile(r'^\s*Continues next page\s*\.?\s*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\s*Press enter or click to view image in full size\s*$', re.IGNORECASE | re.MULTILINE),
]

# Whole <table>…</table> block
_RE_TABLE_BLOCK = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)


@dataclass
class ImageAltChunk:
    """Represents an image description extracted from the markdown."""
    alt_text: str
    position: int        # character offset in the ORIGINAL text


@dataclass
class PreprocessedDocument:
    """Output of the preprocessor for a single markdown file."""
    cleaned_text: str                        # main text after cleaning
    image_alts: list[ImageAltChunk]         # extracted image descriptions
    original_text: str                       # untouched original
    source_path: str


def _is_toc_table(table_html: str) -> bool:
    """Return True if this table looks like a table of contents."""
    return bool(_RE_TOC_DOTS.search(table_html))


def _extract_images(text: str, strategy: str) -> tuple[str, list[ImageAltChunk]]:
    """
    Handle image markdown according to strategy.

    Returns (modified_text, list_of_ImageAltChunk).
    """
    alts: list[ImageAltChunk] = []
    offset_delta = 0  # track how offsets shift as we replace text

    def _replacer(match: re.Match) -> str:
        nonlocal offset_delta
        alt = match.group(1).strip()
        pos = match.start() + offset_delta

        if strategy == "drop":
            replacement = ""
        elif strategy == "separate":
            # Leave a placeholder so we can later detect where the image was
            placeholder = f"\n[IMAGE_DESC_PLACEHOLDER_{len(alts)}]\n"
            alts.append(ImageAltChunk(alt_text=alt, position=pos))
            replacement = placeholder
        else:  # "keep" — replace the markdown syntax with plain alt text
            replacement = f"\n{alt}\n"

        offset_delta += len(replacement) - len(match.group(0))
        return replacement

    cleaned = _RE_IMAGE.sub(_replacer, text)
    return cleaned, alts


def _remove_toc_tables(text: str) -> str:
    """Remove table-of-contents tables from the markdown."""
    def _maybe_remove(match: re.Match) -> str:
        if _is_toc_table(match.group(0)):
            return "\n"
        return match.group(0)

    return _RE_TABLE_BLOCK.sub(_maybe_remove, text)


def _fix_adjacent_tables(text: str) -> str:
    """Insert a blank line between two <table> blocks that are directly adjacent."""
    return _RE_ADJACENT_TABLES.sub(r'\1\n\n\2', text)


def _strip_superscripts(text: str) -> str:
    return _RE_SUPERSCRIPT.sub("", text)


def _strip_decorative_lines(text: str) -> str:
    for pattern in _DECORATIVE_PATTERNS:
        text = pattern.sub("", text)
    return text


def _collapse_blank_lines(text: str) -> str:
    """Replace 3+ consecutive blank lines with 2."""
    return re.sub(r'\n{3,}', '\n\n', text)


class MarkdownPreprocessor:
    """
    Stateless transformer that cleans a single markdown string.

    Usage:
        prep = MarkdownPreprocessor(config)
        doc  = prep.process(raw_text, source_path="./outputs/foo/foo.md")
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.cfg = config or PreprocessorConfig()

    def process(self, raw_text: str, source_path: str = "") -> PreprocessedDocument:
        text = raw_text

        # Step 1 — fix adjacent tables before anything else touches the HTML
        if self.cfg.fix_adjacent_tables:
            text = _fix_adjacent_tables(text)

        # Step 2 — drop table-of-contents tables
        if self.cfg.drop_toc_tables:
            text = _remove_toc_tables(text)

        # Step 3 — handle images
        text, image_alts = _extract_images(text, self.cfg.image_alt_strategy)

        # Step 4 — strip superscripts
        if self.cfg.strip_superscripts:
            text = _strip_superscripts(text)

        # Step 5 — remove decorative lines
        if self.cfg.strip_decorative_lines:
            text = _strip_decorative_lines(text)

        # Step 6 — normalise whitespace
        text = _collapse_blank_lines(text).strip()

        return PreprocessedDocument(
            cleaned_text=text,
            image_alts=image_alts,
            original_text=raw_text,
            source_path=source_path,
        )
