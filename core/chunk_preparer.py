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
    _MIN_EMBEDDABLE_TOKENS = 80
    _NOISE_SECTION_TITLES = {
        "acknowledgements",
        "reviewers and contributors",
        "curators and editors",
        "technical writer",
        "designer",
        "table of contents",
    }
    _DECORATIVE_IMAGE = re.compile(
        r"!\[[^\]]*(?:google logo|abstract|geometric|crystal|gemstone|logo)[^\]]*\]"
        r"|\b(?:google logo|abstract(?:,\s*)?(?:three-dimensional\s+)?geometric shape|faceted geometric shape|crystal|gemstone)\b",
        re.IGNORECASE,
    )
    _AUTHOR_COVER_BLOCK = re.compile(r"^\s*author:\s+.+?!\[", re.IGNORECASE | re.DOTALL)
    _AUTHOR_ONLY_BLOCK = re.compile(r"^\s*author:\s+\S.+$", re.IGNORECASE)

    def __init__(self, settings: CompilerSettings = DEFAULT_SETTINGS) -> None:
        self.settings = settings

    def prepare(self, analysis: StructureAnalysis, normalization_flags: dict[int, list[str]] | None = None) -> list[ChunkCandidate]:
        if not analysis.ordered_blocks:
            return []
        normalization_flags = normalization_flags or {}
        candidates = [
            self._candidate_for_block(index, block, analysis, normalization_flags)
            for index, block in enumerate(analysis.ordered_blocks, start=1)
            if not self._is_embedding_noise_block(block)
        ]
        candidates = self._merge_sparse_candidates(candidates)
        for index, candidate in enumerate(candidates):
            candidate.adjacency["previous_chunk_id"] = candidates[index - 1].chunk_id if index > 0 else None
            candidate.adjacency["next_chunk_id"] = candidates[index + 1].chunk_id if index + 1 < len(candidates) else None
        return candidates

    def _merge_sparse_candidates(self, candidates: list[ChunkCandidate]) -> list[ChunkCandidate]:
        """Merge tiny adjacent text chunks so embedding candidates have semantic mass."""
        merged: list[ChunkCandidate] = []
        pending: ChunkCandidate | None = None

        for candidate in candidates:
            if self._can_merge_sparse(pending, candidate):
                pending = self._merge_pair(pending, candidate)  # type: ignore[arg-type]
                continue

            if pending is not None:
                merged.append(pending)
            pending = candidate

        if pending is not None:
            merged.append(pending)

        return merged

    def _can_merge_sparse(self, left: ChunkCandidate | None, right: ChunkCandidate) -> bool:
        if left is None:
            return False
        same_section = left.parent_section_id == right.parent_section_id and left.heading_path == right.heading_path
        sibling_section = (
            len(left.heading_path) > 1
            and len(right.heading_path) > 1
            and left.heading_path[:-1] == right.heading_path[:-1]
        )
        if not same_section and not sibling_section:
            return False
        if left.content_type == "reference" or right.content_type == "reference":
            return False
        combined_tokens = left.token_estimate + right.token_estimate
        return (
            left.token_estimate < self._MIN_EMBEDDABLE_TOKENS
            or right.token_estimate < self._MIN_EMBEDDABLE_TOKENS
        ) and combined_tokens <= self.settings.max_candidate_tokens

    def _merge_pair(self, left: ChunkCandidate, right: ChunkCandidate) -> ChunkCandidate:
        text = self._merge_text(left, right)
        token_estimate = self._estimate_tokens(text)
        density = self._density(text)
        line_range = [left.line_range[0], right.line_range[1]]
        merged_sources = left.debug_metadata.get("merged_source_chunks", [left.chunk_id])
        merged_sources = [*merged_sources, right.chunk_id]

        left.text = text
        left.line_range = line_range
        left.token_estimate = token_estimate
        left.importance_score = self._importance_text(left.content_type, density, token_estimate)
        left.debug_metadata["source_offset"] = {"start_line": line_range[0], "end_line": line_range[1]}
        left.debug_metadata["merged_source_chunks"] = merged_sources
        left.debug_metadata["extraction_method"] = "chunk_preparer::sparse_merge"
        left.evaluation_metadata["density_score"] = density
        left.evaluation_metadata["chunk_coherence_score"] = max(
            left.evaluation_metadata.get("chunk_coherence_score", 0.0),
            right.evaluation_metadata.get("chunk_coherence_score", 0.0),
        )
        return left

    def _merge_text(self, left: ChunkCandidate, right: ChunkCandidate) -> str:
        if left.heading_path == right.heading_path:
            return f"{left.text.rstrip()}\n\n{right.text.lstrip()}"

        left_label = left.heading_path[-1] if left.heading_path else "Section"
        right_label = right.heading_path[-1] if right.heading_path else "Section"
        left_text = left.text.rstrip()
        if not left_text.startswith(f"{left_label}:"):
            left_text = f"{left_label}:\n{left_text}"
        if f"{right_label}:" in left_text:
            return f"{left_text}\n\n{right.text.lstrip()}"
        return f"{left_text}\n\n{right_label}:\n{right.text.lstrip()}"

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
        parent = analysis.flat_map.get(block.parent_id)
        context_start = parent.line_range[0] if parent else block.line_range[0]
        flags = {
            line: values
            for line, values in normalization_flags.items()
            if context_start <= line <= block.line_range[1]
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

    def _is_embedding_noise_block(self, block: DocumentNode) -> bool:
        """Skip OCR/layout blocks that should not be embedded as knowledge."""
        heading_titles = {heading.strip().lower() for heading in block.heading_path}
        last_heading = block.heading_path[-1].strip().lower() if block.heading_path else ""
        if "acknowledgements" in heading_titles:
            return True
        if last_heading in self._NOISE_SECTION_TITLES:
            return True

        text = block.content.strip()
        if not text:
            return True
        if self._AUTHOR_ONLY_BLOCK.fullmatch(text):
            return True
        if self._AUTHOR_COVER_BLOCK.search(text) and self._DECORATIVE_IMAGE.search(text):
            return True
        if self._DECORATIVE_IMAGE.search(text) and not self._has_substantive_text(text):
            return True
        if "![" in text and self._DECORATIVE_IMAGE.search(text) and not self._has_substantive_text(text):
            return True
        return False

    @staticmethod
    def _has_substantive_text(text: str) -> bool:
        without_images = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
        words = re.findall(r"[A-Za-z]{4,}", without_images)
        domain_words = {
            "prompt",
            "prompts",
            "model",
            "models",
            "token",
            "tokens",
            "temperature",
            "sampling",
            "output",
            "json",
            "classification",
            "example",
        }
        return len(words) >= 35 and bool({word.lower() for word in words} & domain_words)

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
        return self._importance_text(block.node_type, density, token_estimate)

    def _importance_text(self, node_type: str, density: float, token_estimate: int) -> float:
        score = 0.35 + density * 0.45
        if node_type in {"table", "code_block", "list", "reference"}:
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
