"""Hierarchy-aware chunk candidate preparation."""

from __future__ import annotations

import math
import re

from config.settings import CompilerSettings, DEFAULT_SETTINGS
from rag_doc_compiler.errors import ChunkPreparationError
from rag_doc_compiler.models import ChunkCandidate, DocumentNode, StructureAnalysis


class ChunkPreparer:
    """Prepare deterministic chunk candidates without binding to a vector DB."""

    _ALLOWED_INTENTS = {"definition", "explanation", "list", "warning", "example"}
    _ALLOWED_TYPES = {"technical", "narrative", "procedural", "reference"}

    def __init__(self, settings: CompilerSettings = DEFAULT_SETTINGS) -> None:
        self.settings = settings

    def prepare(self, analysis: StructureAnalysis, normalization_flags: dict[int, list[str]] | None = None) -> list[ChunkCandidate]:
        if not analysis.ordered_blocks:
            return []
        normalization_flags = normalization_flags or {}
        candidates = [
            self._candidate_for_block(index, block, analysis, normalization_flags)
            for index, block in enumerate(analysis.ordered_blocks, start=1)
        ]
        for index, candidate in enumerate(candidates):
            candidate.adjacency["previous_chunk_id"] = candidates[index - 1].chunk_id if index > 0 else None
            candidate.adjacency["next_chunk_id"] = candidates[index + 1].chunk_id if index + 1 < len(candidates) else None
        return candidates

    def _candidate_for_block(
        self,
        ordinal: int,
        block: DocumentNode,
        analysis: StructureAnalysis,
        normalization_flags: dict[int, list[str]],
    ) -> ChunkCandidate:
        if not block.parent_id:
            raise ChunkPreparationError(f"Block {block.node_id} has no parent section.")
        token_estimate = self._estimate_tokens(block.content)
        density = self._density(block.content)
        intent = self._intent(block)
        content_type = self._content_type(block)
        anchor = self._anchor(block.heading_path)
        parent_section_id = block.parent_id if block.parent_id != "doc_root" else "sec_0"
        chunk_id = f"{parent_section_id}::chunk_{ordinal}::{anchor}"
        context_lines = set(range(block.line_range[0], block.line_range[1] + 1))
        parent = analysis.flat_map.get(block.parent_id)
        while parent and parent.node_type == "section":
            context_lines.add(parent.line_range[0])
            parent = analysis.flat_map.get(parent.parent_id) if parent.parent_id else None
        flags = {
            line: values
            for line, values in normalization_flags.items()
            if line in context_lines
        }
        return ChunkCandidate(
            chunk_id=chunk_id,
            parent_section_id=parent_section_id,
            heading_path=block.heading_path,
            line_range=block.line_range,
            content_type=content_type,
            intent=intent,
            semantic_tags=self._semantic_tags(block),
            importance_score=self._importance(block, density, token_estimate),
            token_estimate=token_estimate,
            adjacency={"previous_chunk_id": None, "next_chunk_id": None},
            hierarchy_level=max(len(block.heading_path), 0),
            text=block.content,
            debug_metadata={
                "source_offset": {"start_line": block.line_range[0], "end_line": block.line_range[1]},
                "extraction_method": f"structure_analyzer::{block.node_type}",
                "normalization_flags": flags,
            },
            evaluation_metadata={
                "chunk_coherence_score": self._coherence(block),
                "density_score": density,
                "redundancy_flag": False,
            },
        )

    def _intent(self, block: DocumentNode) -> str:
        text = block.content.lower()
        if block.node_type == "list":
            return "list"
        if "warning" in text or "caution" in text or "important:" in text:
            return "warning"
        if "example" in text or block.node_type == "code_block":
            return "example"
        if re.search(r"\b(is|are|means|refers to)\b", text):
            return "definition"
        return "explanation"

    def _content_type(self, block: DocumentNode) -> str:
        text = block.content.lower()
        if block.node_type in {"table", "code_block"}:
            return "reference"
        if re.search(r"\b(step|install|configure|run|execute|first|then)\b", text):
            return "procedural"
        if re.search(r"\b(api|model|token|json|python|rag|embedding|pipeline)\b", text):
            return "technical"
        return "narrative"

    @staticmethod
    def _semantic_tags(block: DocumentNode) -> list[str]:
        tags = [block.node_type]
        tags.extend(f"h{index + 1}:{heading.lower().replace(' ', '-')}" for index, heading in enumerate(block.heading_path))
        return tags

    def _estimate_tokens(self, text: str) -> int:
        words = len(re.findall(r"\S+", text))
        chars = max(1, math.ceil(len(text) / self.settings.token_chars_per_token))
        return max(words, chars)

    @staticmethod
    def _density(text: str) -> float:
        tokens = re.findall(r"[A-Za-z0-9_]+", text)
        if not tokens:
            return 0.0
        unique_ratio = len(set(token.lower() for token in tokens)) / len(tokens)
        symbol_ratio = min(0.25, len(re.findall(r"[{}()[\]|`]", text)) / max(len(text), 1))
        return round(min(1.0, unique_ratio * 0.85 + symbol_ratio), 3)

    def _importance(self, block: DocumentNode, density: float, token_estimate: int) -> float:
        score = 0.35 + density * 0.45
        if block.node_type in {"table", "code_block", "list"}:
            score += 0.1
        if token_estimate > self.settings.max_candidate_tokens:
            score -= 0.1
        return round(max(0.0, min(1.0, score)), 3)

    @staticmethod
    def _coherence(block: DocumentNode) -> float:
        if block.node_type in {"table", "code_block", "list"}:
            return 0.9
        sentence_count = len(re.findall(r"[.!?](\s|$)", block.content))
        return round(min(1.0, 0.55 + sentence_count * 0.12), 3)

    @staticmethod
    def _anchor(heading_path: list[str]) -> str:
        if not heading_path:
            return "doc_root"
        anchor = re.sub(r"[^a-z0-9]+", "-", heading_path[-1].lower()).strip("-")
        return anchor or "section"
