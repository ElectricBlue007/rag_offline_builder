"""Shared data models for the compiler pipeline.

These models implement Phase 1: Markdown Ready For Chunking with strict adherence to:
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
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class BlockType(str, Enum):
    """Rule 26: Typed Blocks for different content types."""
    
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    CODE = "code"
    TABLE = "table"
    LIST = "list"
    IMAGE = "image"
    QUOTE = "quote"
    LINK = "link"
    FRONT_MATTER = "front_matter"
    HTML_NOISE = "html_noise"
    BADGE_NOISE = "badge_noise"


class ListType(str, Enum):
    """Rule 9: Distinguish between ordered and unordered lists."""
    
    ORDERED = "ordered"
    UNORDERED = "unordered"


@dataclass(frozen=True)
class LinkMetadata:
    """Rule 11-12: Preserve link text and URL separately."""
    
    text: str
    url: str
    
    def to_dict(self) -> dict[str, str]:
        return {"text": self.text, "url": self.url}


@dataclass(frozen=True)
class ImageMetadata:
    """Rule 13-14: Convert images to metadata without losing information."""
    
    alt: str
    src: str
    
    def to_dict(self) -> dict[str, str]:
        return {"alt": self.alt, "src": self.src}


@dataclass(frozen=True)
class TableStructure:
    """Rule 7-8: Preserve tables as structured data."""
    
    headers: list[str]
    rows: list[list[str]]
    alignment: list[str] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "headers": self.headers,
            "rows": self.rows,
        }
        if self.alignment:
            result["alignment"] = self.alignment
        return result


@dataclass(frozen=True)
class CodeBlockMetadata:
    """Rule 5-6: Preserve code blocks exactly with language metadata."""
    
    language: str | None = None
    fenced: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        return {"language": self.language, "fenced": self.fenced}


@dataclass(frozen=True)
class ListItem:
    """Rule 9-10: Preserve list items with nesting information."""
    
    content: str
    level: int = 0
    item_type: ListType = ListType.UNORDERED
    children: list[ListItem] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "level": self.level,
            "item_type": self.item_type.value,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass
class DocumentBlock:
    """A single structured block implementing Rules 1-30.
    
    Each block has:
    - Stable ID (Rule 22)
    - Document order (Rule 21)
    - Type (Rule 26)
    - Heading path (Rule 4)
    - Level preservation (Rule 3)
    - Boundary preservation (Rules 15-16)
    """
    
    block_id: str
    block_type: BlockType
    order: int
    line_range: tuple[int, int]
    content: str
    level: int = 0
    heading_path: list[str] = field(default_factory=list)
    parent_id: str | None = None
    
    # Type-specific metadata
    link_metadata: LinkMetadata | None = None
    image_metadata: ImageMetadata | None = None
    table_structure: TableStructure | None = None
    code_metadata: CodeBlockMetadata | None = None
    list_items: list[ListItem] | None = None
    quote_level: int = 0
    
    # Noise detection (Rules 17-18, 23-24)
    is_boilerplate: bool = False
    is_generated_noise: bool = False
    noise_flags: list[str] = field(default_factory=list)
    
    # Front matter (Rule 28)
    front_matter_data: dict[str, Any] | None = None
    
    # Normalization flags for audit trail
    normalization_flags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert block to dictionary representation."""
        metadata: dict[str, Any] = {
            "level": self.level,
            "heading_path": self.heading_path,
        }
        if self.link_metadata:
            metadata.update(self.link_metadata.to_dict())
        if self.image_metadata:
            metadata.update(self.image_metadata.to_dict())
        if self.table_structure:
            metadata.update(self.table_structure.to_dict())
        if self.code_metadata:
            metadata.update(self.code_metadata.to_dict())
        if self.list_items:
            metadata["items"] = [item.to_dict() for item in self.list_items]
            metadata["ordered"] = all(item.item_type == ListType.ORDERED for item in self.list_items)
        if self.quote_level > 0:
            metadata["quote_level"] = self.quote_level
        if self.front_matter_data:
            metadata["front_matter_data"] = self.front_matter_data

        data: dict[str, Any] = {
            "block_id": self.block_id,
            "type": self.block_type.value,
            "block_type": self.block_type.value,
            "order": self.order,
            "line_range": list(self.line_range),
            "content": self.content,
            "level": self.level,
            "heading_path": self.heading_path,
            "parent_id": self.parent_id,
            "metadata": metadata,
        }
        
        # Add type-specific metadata
        if self.link_metadata:
            data["link_metadata"] = self.link_metadata.to_dict()
        if self.image_metadata:
            data["image_metadata"] = self.image_metadata.to_dict()
        if self.table_structure:
            data["table_structure"] = self.table_structure.to_dict()
        if self.code_metadata:
            data["code_metadata"] = self.code_metadata.to_dict()
        if self.list_items:
            data["list_items"] = [item.to_dict() for item in self.list_items]
        if self.quote_level > 0:
            data["quote_level"] = self.quote_level
        if self.front_matter_data:
            data["front_matter_data"] = self.front_matter_data
        
        # Add noise detection flags
        if self.is_boilerplate:
            data["is_boilerplate"] = True
        if self.is_generated_noise:
            data["is_generated_noise"] = True
        if self.noise_flags:
            data["noise_flags"] = self.noise_flags
        if self.normalization_flags:
            data["normalization_flags"] = self.normalization_flags
        
        return data


@dataclass
class DocumentNode:
    """Hierarchical document node for tree structure (Rule 2-4)."""
    
    node_id: str
    node_type: str
    line_range: list[int]
    content: str
    level: int = 0
    heading_path: list[str] = field(default_factory=list)
    parent_id: str | None = None
    children: list[DocumentNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["children"] = [child.to_dict() for child in self.children]
        return data


@dataclass(frozen=True)
class NormalizationResult:
    """Result of markdown normalization with audit flags."""
    
    markdown: str
    flags_by_line: dict[int, list[str]] = field(default_factory=dict)
    removed_noise_count: int = 0
    utf8_normalized: bool = False


@dataclass(frozen=True)
class StructureAnalysis:
    """Complete structure analysis with flat and hierarchical views."""
    
    tree: DocumentNode
    flat_map: dict[str, DocumentNode]
    ordered_blocks: list[DocumentNode]
    structured_blocks: list[DocumentBlock] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "tree": self.tree.to_dict(),
            "flat_reference_map": {
                node_id: node.to_dict() for node_id, node in self.flat_map.items()
            },
            "ordered_block_ids": [block.node_id for block in self.ordered_blocks],
            "structured_blocks": [block.to_dict() for block in self.structured_blocks],
        }


@dataclass
class ChunkCandidate:
    """Prepared chunk candidate (not yet final chunking - Rule 30)."""
    
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
