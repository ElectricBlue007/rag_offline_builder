"""Markdown structure analyzer for hierarchy and observability."""

from __future__ import annotations

import re
from collections.abc import Iterable

from rag_doc_compiler.errors import StructureError
from rag_doc_compiler.models import DocumentNode, NormalizationResult, StructureAnalysis


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_LIST = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.+)$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE_DELIMITER = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_HTML_TABLE_OPEN = re.compile(r"^\s*<table\b", re.IGNORECASE)
_HTML_TABLE_CLOSE = re.compile(r"</table>\s*$", re.IGNORECASE)


class StructureAnalyzer:
    """Build a hierarchical document tree and flat reference map."""

    def analyze(self, normalized: NormalizationResult | str) -> StructureAnalysis:
        markdown = normalized.markdown if isinstance(normalized, NormalizationResult) else normalized
        lines = markdown.splitlines()
        root = DocumentNode("doc_root", "document", [1, max(len(lines), 1)], "", level=0)
        flat_map: dict[str, DocumentNode] = {root.node_id: root}
        ordered_blocks: list[DocumentNode] = []
        section_stack: list[DocumentNode] = [root]
        counters: dict[str, int] = {"sec": 0, "block": 0}
        index = 0

        while index < len(lines):
            line = lines[index]
            line_no = index + 1
            if not line.strip():
                index += 1
                continue

            heading = _HEADING.match(line)
            if heading:
                level = len(heading.group(1))
                title = heading.group(2).strip()
                if not title:
                    raise StructureError(f"Empty heading at line {line_no}.")
                if level > section_stack[-1].level + 1:
                    raise StructureError(
                        f"Heading level jumps from {section_stack[-1].level} to {level} at line {line_no}."
                    )
                while section_stack and section_stack[-1].level >= level:
                    section_stack.pop()
                counters["sec"] += 1
                parent = section_stack[-1]
                node = DocumentNode(
                    node_id=f"sec_{counters['sec']}",
                    node_type="section",
                    line_range=[line_no, line_no],
                    content=line,
                    level=level,
                    heading_path=parent.heading_path + [title],
                    parent_id=parent.node_id,
                    metadata={"title": title},
                )
                parent.children.append(node)
                flat_map[node.node_id] = node
                section_stack.append(node)
                index += 1
                continue

            node, index = self._consume_block(lines, index, section_stack[-1], counters)
            section_stack[-1].children.append(node)
            flat_map[node.node_id] = node
            ordered_blocks.append(node)

        for node in flat_map.values():
            if node.node_type == "section":
                child_ranges = [child.line_range for child in node.children]
                if child_ranges:
                    node.line_range[1] = max(end for _, end in child_ranges)

        return StructureAnalysis(tree=root, flat_map=flat_map, ordered_blocks=ordered_blocks)

    def _consume_block(
        self,
        lines: list[str],
        start_index: int,
        section: DocumentNode,
        counters: dict[str, int],
    ) -> tuple[DocumentNode, int]:
        line = lines[start_index]
        line_no = start_index + 1
        if _FENCE.match(line):
            return self._consume_until_fence_close(lines, start_index, section, counters)
        if _HTML_TABLE_OPEN.match(line):
            return self._consume_until_html_table_close(lines, start_index, section, counters)
        if _LIST.match(line):
            return self._consume_while(lines, start_index, section, counters, "list", self._is_list_continuation)
        if self._starts_table(lines, start_index):
            return self._consume_while(lines, start_index, section, counters, "table", lambda value: "|" in value)
        return self._consume_while(lines, start_index, section, counters, "paragraph", self._is_paragraph_continuation)

    def _consume_until_fence_close(
        self, lines: list[str], start_index: int, section: DocumentNode, counters: dict[str, int]
    ) -> tuple[DocumentNode, int]:
        marker = _FENCE.match(lines[start_index]).group(1)  # type: ignore[union-attr]
        end_index = start_index + 1
        while end_index < len(lines):
            if _FENCE.match(lines[end_index]) and _FENCE.match(lines[end_index]).group(1) == marker:  # type: ignore[union-attr]
                return self._make_block(lines, start_index, end_index + 1, section, counters, "code_block"), end_index + 1
            end_index += 1
        raise StructureError(f"Unclosed code block starting at line {start_index + 1}.")

    def _consume_until_html_table_close(
        self, lines: list[str], start_index: int, section: DocumentNode, counters: dict[str, int]
    ) -> tuple[DocumentNode, int]:
        end_index = start_index
        while end_index < len(lines):
            if _HTML_TABLE_CLOSE.search(lines[end_index]):
                return self._make_block(lines, start_index, end_index + 1, section, counters, "table"), end_index + 1
            end_index += 1
        raise StructureError(f"Unclosed HTML table starting at line {start_index + 1}.")

    def _consume_while(
        self,
        lines: list[str],
        start_index: int,
        section: DocumentNode,
        counters: dict[str, int],
        node_type: str,
        predicate,
    ) -> tuple[DocumentNode, int]:
        end_index = start_index
        while end_index < len(lines):
            candidate = lines[end_index]
            if not candidate.strip() or _HEADING.match(candidate) or _FENCE.match(candidate) or _HTML_TABLE_OPEN.match(candidate):
                break
            if not predicate(candidate):
                break
            end_index += 1
        return self._make_block(lines, start_index, end_index, section, counters, node_type), end_index

    def _make_block(
        self,
        lines: list[str],
        start_index: int,
        end_index: int,
        section: DocumentNode,
        counters: dict[str, int],
        node_type: str,
    ) -> DocumentNode:
        counters["block"] += 1
        content = "\n".join(lines[start_index:end_index])
        return DocumentNode(
            node_id=f"block_{counters['block']}",
            node_type=node_type,
            line_range=[start_index + 1, end_index],
            content=content,
            level=section.level + 1,
            heading_path=list(section.heading_path),
            parent_id=section.node_id,
        )

    @staticmethod
    def _starts_table(lines: list[str], start_index: int) -> bool:
        return (
            "|" in lines[start_index]
            and start_index + 1 < len(lines)
            and bool(_TABLE_DELIMITER.match(lines[start_index + 1]))
        )

    @staticmethod
    def _is_list_continuation(line: str) -> bool:
        return bool(_LIST.match(line)) or line.startswith((" ", "\t"))

    @staticmethod
    def _is_paragraph_continuation(line: str) -> bool:
        return not _LIST.match(line) and "|" not in line
