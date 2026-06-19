"""Final chunking strategy applied after clean Markdown generation.

This stage produces retrieval-ready chunks. It is intentionally separate from
ChunkPreparer, which exposes lower-level candidates for debugging and analysis.
"""

from __future__ import annotations

import re
from collections import Counter

from config.settings import CompilerSettings, DEFAULT_SETTINGS
from core.chunk_preparer import ChunkPreparer
from rag_doc_compiler.errors import ChunkPreparationError
from rag_doc_compiler.models import DocumentNode, FinalChunk, StructureAnalysis, ChunkingResult


class HierarchyAwareChunkingStrategy:
    """Create deterministic no-overlap semantic chunks plus section anchors."""

    def __init__(self, settings: CompilerSettings = DEFAULT_SETTINGS) -> None:
        self.settings = settings
        self.classifier = ChunkPreparer(settings)

    def chunk(self, analysis: StructureAnalysis) -> ChunkingResult:
        chunks: list[FinalChunk] = []
        section_nodes = [
            node
            for node in analysis.flat_map.values()
            if node.node_type == "section"
        ]
        section_nodes.sort(key=lambda node: node.line_range[0])

        for section in section_nodes:
            chunks.append(self._section_chunk(section, len(chunks) + 1))

        for block in analysis.ordered_blocks:
            chunks.append(self._semantic_chunk(block, analysis, len(chunks) + 1))

        for index, chunk in enumerate(chunks):
            chunk.adjacency["previous_chunk_id"] = chunks[index - 1].chunk_id if index else None
            chunk.adjacency["next_chunk_id"] = chunks[index + 1].chunk_id if index + 1 < len(chunks) else None

        return ChunkingResult(
            strategy={
                "name": "hierarchy_aware_no_overlap_v1",
                "section_level_chunk_always_exists": True,
                "semantic_subchunks_always_exist": True,
                "overlap_policy": "none",
                "embedding_agnostic": True,
            },
            chunks=chunks,
        )

    def _section_chunk(self, section: DocumentNode, ordinal: int) -> FinalChunk:
        title = section.metadata.get("title", section.content.lstrip("# ").strip())
        original_text = section.content
        keywords = self._keywords(" ".join(section.heading_path + [original_text]))
        return FinalChunk(
            chunk_id=f"{section.node_id}::{self._slug(title)}::section::{ordinal:04d}",
            chunk_kind="section",
            parent_section_id=section.parent_id if section.parent_id != "doc_root" else None,
            source_node_id=section.node_id,
            heading_path=list(section.heading_path),
            line_range=list(section.line_range[:1]) + [section.line_range[0]],
            content_type="reference",
            intent="explanation",
            semantic_tags=["section", *self._heading_tags(section.heading_path)],
            keywords=keywords,
            language=self._language(original_text),
            importance_score=0.8,
            token_estimate=self.classifier._estimate_tokens(original_text),
            original_text=original_text,
            embedding_text=self._embedding_text(section.heading_path, original_text, keywords),
            adjacency={"previous_chunk_id": None, "next_chunk_id": None},
            provenance={
                "source_node_id": section.node_id,
                "source_node_type": section.node_type,
                "line_range": [section.line_range[0], section.line_range[0]],
            },
        )

    def _semantic_chunk(self, block: DocumentNode, analysis: StructureAnalysis, ordinal: int) -> FinalChunk:
        if not block.parent_id:
            raise ChunkPreparationError(f"Block {block.node_id} has no parent section.")
        content_type = self._content_type(block)
        intent = self.classifier._intent(block)
        keywords = self._keywords(" ".join(block.heading_path + [block.content]))
        parent_section_id = block.parent_id if block.parent_id != "doc_root" else None
        anchor = self._slug(block.heading_path[-1]) if block.heading_path else "root"
        return FinalChunk(
            chunk_id=f"{parent_section_id or 'doc_root'}::{anchor}::{intent}::{ordinal:04d}",
            chunk_kind="semantic",
            parent_section_id=parent_section_id,
            source_node_id=block.node_id,
            heading_path=list(block.heading_path),
            line_range=list(block.line_range),
            content_type=content_type,
            intent=intent,
            semantic_tags=[block.node_type, *self._heading_tags(block.heading_path)],
            keywords=keywords,
            language=self._language(block.content),
            importance_score=self.classifier._importance(
                block,
                self.classifier._density(block.content),
                self.classifier._estimate_tokens(block.content),
            ),
            token_estimate=self.classifier._estimate_tokens(block.content),
            original_text=block.content,
            embedding_text=self._embedding_text(block.heading_path, block.content, keywords),
            adjacency={"previous_chunk_id": None, "next_chunk_id": None},
            provenance={
                "source_node_id": block.node_id,
                "source_node_type": block.node_type,
                "parent_section_id": parent_section_id,
                "line_range": block.line_range,
            },
        )

    def _content_type(self, block: DocumentNode) -> str:
        candidate_type = self.classifier._content_type(block)
        if candidate_type == "narrative":
            return "explanation"
        if candidate_type == "procedural":
            return "technical"
        return candidate_type

    @staticmethod
    def _keywords(text: str) -> dict[str, list[str]]:
        tokens = [
            token.lower()
            for token in re.findall(r"[^\W\d_]{3,}", text, flags=re.UNICODE)
        ]
        stopwords = {
            "the", "and", "for", "with", "this", "that", "una", "del", "los",
            "las", "con", "por", "para", "que", "and", "are", "is", "you",
        }
        filtered = [token for token in tokens if token not in stopwords]
        lexical = [token for token, _ in Counter(filtered).most_common(8)]
        entity_candidates = re.findall(r"\b[A-Z][^\W\d_]{2,}\b", text, flags=re.UNICODE)
        return {
            "lexical": lexical,
            "semantic": lexical[:5],
            "conceptual_topics": lexical[:3],
            "entities": list(dict.fromkeys(entity_candidates))[:8],
        }

    @staticmethod
    def _language(text: str) -> dict[str, object]:
        lowered = text.lower()
        spanish_hits = sum(word in lowered for word in ("acción", "sistema", "datos", "informe", "seleccione"))
        english_hits = sum(word in lowered for word in ("the", "system", "data", "report", "select"))
        if spanish_hits > english_hits:
            return {"detected_language": "es", "confidence": 0.72}
        if english_hits > 0:
            return {"detected_language": "en", "confidence": 0.68}
        return {"detected_language": "unknown", "confidence": 0.0}

    @staticmethod
    def _embedding_text(heading_path: list[str], original_text: str, keywords: dict[str, list[str]]) -> str:
        heading = " > ".join(heading_path) if heading_path else "Document root"
        keyword_text = ", ".join(keywords["lexical"])
        return f"Heading path: {heading}\nKeywords: {keyword_text}\nOriginal text:\n{original_text}"

    @staticmethod
    def _heading_tags(heading_path: list[str]) -> list[str]:
        return [f"h{index + 1}:{HierarchyAwareChunkingStrategy._slug(heading)}" for index, heading in enumerate(heading_path)]

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return slug or "section"
