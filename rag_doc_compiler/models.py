"""Shared data models for the compiler pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class NormalizationResult:
    markdown: str
    flags_by_line: dict[int, list[str]] = field(default_factory=dict)


@dataclass
class DocumentNode:
    node_id: str
    node_type: str
    line_range: list[int]
    content: str
    level: int = 0
    heading_path: list[str] = field(default_factory=list)
    parent_id: str | None = None
    children: list["DocumentNode"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["children"] = [child.to_dict() for child in self.children]
        return data


@dataclass(frozen=True)
class StructureAnalysis:
    tree: DocumentNode
    flat_map: dict[str, DocumentNode]
    ordered_blocks: list[DocumentNode]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree": self.tree.to_dict(),
            "flat_reference_map": {
                node_id: node.to_dict() for node_id, node in self.flat_map.items()
            },
            "ordered_block_ids": [block.node_id for block in self.ordered_blocks],
        }


@dataclass
class ChunkCandidate:
    chunk_id: str
    parent_section_id: str
    heading_path: list[str]
    line_range: list[int]
    content_type: str
    intent: str
    semantic_tags: list[str]
    importance_score: float
    token_estimate: int
    adjacency: dict[str, str | None]
    hierarchy_level: int
    text: str
    debug_metadata: dict[str, Any]
    evaluation_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FinalChunk:
    chunk_id: str
    chunk_kind: str
    parent_section_id: str | None
    source_node_id: str
    heading_path: list[str]
    line_range: list[int]
    content_type: str
    intent: str
    semantic_tags: list[str]
    keywords: dict[str, list[str]]
    language: dict[str, Any]
    importance_score: float
    token_estimate: int
    original_text: str
    embedding_text: str
    adjacency: dict[str, str | None]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChunkingResult:
    strategy: dict[str, Any]
    chunks: list[FinalChunk]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "observability": {
                "chunk_count": len(self.chunks),
                "section_chunk_count": sum(1 for chunk in self.chunks if chunk.chunk_kind == "section"),
                "semantic_chunk_count": sum(1 for chunk in self.chunks if chunk.chunk_kind == "semantic"),
            },
        }
