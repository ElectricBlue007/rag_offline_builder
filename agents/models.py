"""
Phase 2: Document Qualification Engine
Agentic Validation System for RAG-ready Markdown Documents

This module implements a multi-agent validation system that qualifies documents
for chunking, embedding, and retrieval based on 50 rules across 6 layers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import re
import hashlib


class IssueSeverity(Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueType(Enum):
    """Types of validation issues."""
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    RETRIEVAL = "retrieval"
    CHUNK_PLANNING = "chunk_planning"
    EMBEDDING_READINESS = "embedding_readiness"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in the document."""
    rule_id: int
    rule_name: str
    severity: IssueSeverity
    issue_type: IssueType
    description: str
    location: Optional[str] = None
    block_id: Optional[str] = None
    impact: str = ""
    recommended_action: str = ""
    context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "issue_type": self.issue_type.value,
            "description": self.description,
            "location": self.location,
            "block_id": self.block_id,
            "impact": self.impact,
            "recommended_action": self.recommended_action,
            "context": self.context
        }


@dataclass
class SectionAnalysis:
    """Analysis results for a document section."""
    section_id: str
    heading_path: List[str]
    block_count: int
    token_count: int
    semantic_density: float
    has_code: bool
    has_table: bool
    has_list: bool
    has_definition: bool
    has_procedure: bool
    has_api_doc: bool
    is_boilerplate: bool
    is_navigation: bool
    is_footer: bool
    is_header_noise: bool
    has_badges: bool
    chunkability_score: float
    embedding_readiness_score: float
    retrieval_utility_score: float


@dataclass
class ChunkingStrategy:
    """Recommended chunking strategy for the document."""
    strategy_type: str
    min_chunk_size: int
    max_chunk_size: int
    preferred_boundary_types: List[str]
    anchoring_rules: List[str]
    context_injection: bool
    parent_context_depth: int
    merge_sparse_blocks: bool
    split_large_sections: bool
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_type": self.strategy_type,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "preferred_boundary_types": self.preferred_boundary_types,
            "anchoring_rules": self.anchoring_rules,
            "context_injection": self.context_injection,
            "parent_context_depth": self.parent_context_depth,
            "merge_sparse_blocks": self.merge_sparse_blocks,
            "split_large_sections": self.split_large_sections,
            "custom_parameters": self.custom_parameters
        }


@dataclass
class DocumentQualificationResult:
    """Complete qualification result for a document."""
    document_id: str
    is_chunkable: bool
    is_embeddable: bool
    is_retrievable: bool
    overall_score: float
    structural_score: float
    semantic_score: float
    retrieval_score: float
    embedding_readiness_score: float
    
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    chunking_strategy: Optional[ChunkingStrategy] = None
    section_analyses: List[SectionAnalysis] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "is_chunkable": self.is_chunkable,
            "is_embeddable": self.is_embeddable,
            "is_retrievable": self.is_retrievable,
            "overall_score": self.overall_score,
            "structural_score": self.structural_score,
            "semantic_score": self.semantic_score,
            "retrieval_score": self.retrieval_score,
            "embedding_readiness_score": self.embedding_readiness_score,
            "issues": [i.to_dict() for i in self.issues],
            "warnings": [w.to_dict() for w in self.warnings],
            "recommendations": self.recommendations,
            "chunking_strategy": self.chunking_strategy.to_dict() if self.chunking_strategy else None,
            "section_analyses": [s.__dict__ for s in self.section_analyses],
            "metadata": self.metadata
        }
