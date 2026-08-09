"""
splitters/structure_splitter.py

Splits a pre-processed markdown document into STRUCTURAL SECTIONS by
respecting the header hierarchy (#, ##, ###).

Key guarantees:
  - HTML <table> blocks are NEVER split; they always become their own section.
  - Each section carries its full "breadcrumb" (ancestor header chain).
  - Image-description placeholders are detected and tagged.
  - Sections smaller than min_section_chars are merged upward into their
    parent section to avoid orphan micro-chunks.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from rag_pipeline.config import StructureSplitterConfig


# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class StructuralSection:
    """A single section extracted by the structure splitter."""
    content: str
    section_type: str          # "text" | "table" | "image_description"
    breadcrumb: str            # e.g. "# Prompt Engineering > ## Introduction"
    header_levels: list[int]   # depth of each breadcrumb part, e.g. [1, 2]
    header_texts: list[str]    # e.g. ["Prompt Engineering", "Introduction"]
    source_path: str = ""


# ── Regex helpers ─────────────────────────────────────────────────────────────

# Matches a markdown header line: group(1)=hashes, group(2)=title text
_RE_HEADER = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

# Matches a full HTML table block (non-greedy, handles nested tags poorly —
# but the OCR output uses flat single-level tables so this is fine)
_RE_TABLE = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)

# Placeholder left by the preprocessor for image descriptions
_RE_IMG_PLACEHOLDER = re.compile(r'\[IMAGE_DESC_PLACEHOLDER_(\d+)\]')


def _build_breadcrumb(header_texts: list[str], header_levels: list[int]) -> str:
    """Build a human-readable breadcrumb string from the header stack."""
    parts = [f"{'#' * lvl} {txt}" for lvl, txt in zip(header_levels, header_texts)]
    return " > ".join(parts) if parts else "(root)"


def _pop_to_level(stack: list[tuple[int, str]], new_level: int) -> list[tuple[int, str]]:
    """Remove headers from the breadcrumb stack that are at the same or deeper level."""
    return [(lvl, txt) for lvl, txt in stack if lvl < new_level]


class StructureSplitter:
    """
    Splits a pre-processed markdown string into StructuralSection objects.

    Algorithm:
      1. Find all table blocks — protect them as atomic tokens.
      2. Find all header positions.
      3. Walk through the text, emitting sections at each header boundary.
      4. Tables found within a section are split out as separate table-type sections.
      5. Tiny sections below min_section_chars are merged into previous sibling.
    """

    def __init__(self, config: Optional[StructureSplitterConfig] = None):
        self.cfg = config or StructureSplitterConfig()

    # ── Public API ──────────────────────────────────────────────────────────

    def split(self, text: str, source_path: str = "") -> list[StructuralSection]:
        """
        Split `text` into a list of StructuralSection objects.
        `source_path` is forwarded as metadata only.
        """
        # Step 1: protect tables by replacing them with unique tokens
        tables: list[str] = []
        tokenized = self._tokenize_tables(text, tables)

        # Step 2: find header positions in the tokenized text
        raw_sections = self._split_by_headers(tokenized)

        # Step 3: restore tables and explode each raw section into
        #         sub-sections (text segments + table segments)
        sections: list[StructuralSection] = []
        for content, breadcrumb, h_levels, h_texts in raw_sections:
            restored = self._restore_tables(content, tables)
            sub = self._explode_tables(
                restored, breadcrumb, h_levels, h_texts, source_path
            )
            sections.extend(sub)

        # Step 4: merge tiny text sections upward
        sections = self._merge_small_sections(sections)

        return sections

    # ── Internal helpers ────────────────────────────────────────────────────

    def _tokenize_tables(self, text: str, tables: list[str]) -> str:
        """Replace each <table>…</table> with a unique token."""
        def _replace(m: re.Match) -> str:
            idx = len(tables)
            tables.append(m.group(0))
            return f"\n\n[[TABLE_{idx}]]\n\n"

        return _RE_TABLE.sub(_replace, text)

    def _restore_tables(self, text: str, tables: list[str]) -> str:
        """Substitute [[TABLE_N]] tokens back to their HTML."""
        def _restore(m: re.Match) -> str:
            idx = int(m.group(1))
            return tables[idx] if idx < len(tables) else m.group(0)

        return re.sub(r'\[\[TABLE_(\d+)\]\]', _restore, text)

    def _split_by_headers(
        self, tokenized: str
    ) -> list[tuple[str, str, list[int], list[str]]]:
        """
        Walk the tokenized text and emit (content, breadcrumb, levels, texts)
        tuples — one per header-delimited section.
        """
        results: list[tuple[str, str, list[int], list[str]]] = []
        header_stack: list[tuple[int, str]] = []  # [(level, text), …]

        # Split at every header line
        # We keep the headers as part of the delimiter information
        parts = _RE_HEADER.split(tokenized)
        # parts structure after split with 2 capture groups:
        # [pre_text, hashes1, title1, between_text1, hashes2, title2, ...]

        # parts[0] is text before the first header
        pre_text = parts[0].strip()
        if pre_text:
            results.append((
                pre_text,
                _build_breadcrumb([], []),
                [],
                [],
            ))

        i = 1
        while i + 2 <= len(parts):
            hashes = parts[i]
            title = parts[i + 1].strip()
            body = parts[i + 2] if i + 2 < len(parts) else ""
            level = len(hashes)
            i += 3

            # Only process levels listed in config
            if level not in self.cfg.header_levels:
                # Treat as body text under the current breadcrumb
                if results:
                    results[-1] = (
                        results[-1][0] + f"\n{'#'*level} {title}\n" + body,
                        *results[-1][1:],
                    )
                continue

            # Update breadcrumb stack
            header_stack = _pop_to_level(header_stack, level)
            header_stack.append((level, title))

            levels = [lvl for lvl, _ in header_stack]
            texts  = [txt for _, txt in header_stack]
            bc     = _build_breadcrumb(texts, levels)

            content = body.strip()
            if content or True:   # always emit — merge step handles empties
                results.append((content, bc, levels, texts))

        return results

    def _explode_tables(
        self,
        text: str,
        breadcrumb: str,
        h_levels: list[int],
        h_texts: list[str],
        source_path: str,
    ) -> list[StructuralSection]:
        """
        Within a section's text, split out any <table> blocks as separate
        StructuralSection objects of type "table".
        Each image-description placeholder also becomes its OWN atomic
        section (together with any elaboration text that immediately
        follows it, up to the next placeholder/table/end) — this matters
        because Chandra OCR frequently emits several images back-to-back
        with no blank line between them, and each must stay distinct.
        """
        sections: list[StructuralSection] = []

        # Split on table blocks first — tables are always atomic
        table_pattern = re.compile(r'(<table[\s\S]*?</table>)', re.IGNORECASE)
        parts = table_pattern.split(text)

        for part in parts:
            if not part.strip():
                continue

            if re.match(r'<table', part.strip(), re.IGNORECASE):
                sections.append(StructuralSection(
                    content=part.strip(),
                    section_type="table",
                    breadcrumb=breadcrumb,
                    header_levels=h_levels,
                    header_texts=h_texts,
                    source_path=source_path,
                ))
                continue

            # Non-table part: split further at every image placeholder
            for seg_type, seg_text in self._split_by_image_placeholders(part):
                if seg_text.strip():
                    sections.append(StructuralSection(
                        content=seg_text.strip(),
                        section_type=seg_type,
                        breadcrumb=breadcrumb,
                        header_levels=h_levels,
                        header_texts=h_texts,
                        source_path=source_path,
                    ))

        return sections

    def _split_by_image_placeholders(
        self, text: str
    ) -> list[tuple[str, str]]:
        """
        Split `text` into (type, content) segments, where each image
        placeholder starts a new "image_description" segment that absorbs
        all text up to the next placeholder (or the end of `text`).
        Text appearing before the first placeholder is returned as a
        single "text" segment.

        Returns an empty list if `text` has no placeholders — caller
        falls back to treating it as one plain "text" segment.
        """
        matches = list(_RE_IMG_PLACEHOLDER.finditer(text))
        if not matches:
            return [("text", text)]

        segments: list[tuple[str, str]] = []

        # Text before the first placeholder
        pre = text[:matches[0].start()]
        if pre.strip():
            segments.append(("text", pre))

        # One segment per placeholder, absorbing text up to the next one
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            seg = text[start:end]
            if seg.strip():
                segments.append(("image_description", seg))

        return segments

    def _merge_small_sections(
        self, sections: list[StructuralSection]
    ) -> list[StructuralSection]:
        """
        Merge text sections below min_section_chars into the previous section
        (if it exists and is also text). Tables/image descriptions are never merged.
        """
        if not sections:
            return sections

        merged: list[StructuralSection] = [sections[0]]

        for sec in sections[1:]:
            prev = merged[-1]
            too_small = (
                sec.section_type == "text"
                and len(sec.content) < self.cfg.min_section_chars
            )
            can_merge = prev.section_type == "text" and too_small

            if can_merge:
                merged[-1] = StructuralSection(
                    content=prev.content + "\n\n" + sec.content,
                    section_type="text",
                    breadcrumb=prev.breadcrumb,
                    header_levels=prev.header_levels,
                    header_texts=prev.header_texts,
                    source_path=prev.source_path,
                )
            else:
                merged.append(sec)

        return merged
