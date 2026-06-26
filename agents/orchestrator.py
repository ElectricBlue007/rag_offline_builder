"""
Phase 2: Document Qualification Engine - Orchestrator
Coordinates all validation agents and produces final qualification result.
"""

from typing import Dict, Any, Optional
from agents.models import (
    DocumentQualificationResult, ValidationIssue, 
    IssueSeverity, ChunkingStrategy
)
from agents.structure_agent import StructureAgent
from agents.semantic_agent import SemanticAgent
from agents.retrieval_agent import RetrievalAgent
from agents.chunk_planning_agent import ChunkPlanningAgent, EmbeddingReadinessAgent


class DocumentQualificationEngine:
    """
    Main orchestrator for Phase 2 document qualification.
    
    Coordinates four specialized agents:
    - StructureAgent: Validates structural integrity (Rules 101-110)
    - SemanticAgent: Validates semantic quality (Rules 111-120)
    - RetrievalAgent: Validates retrieval utility (Rules 121-130)
    - ChunkPlanningAgent: Plans chunk boundaries (Rules 131-140)
    - EmbeddingReadinessAgent: Assesses embedding readiness (Rules 141-150)
    
    Produces comprehensive qualification result with scores, issues,
    warnings, recommendations, and chunking strategy.
    """
    
    def __init__(self):
        self.structure_agent = StructureAgent()
        self.semantic_agent = SemanticAgent()
        self.retrieval_agent = RetrievalAgent()
        self.chunk_planning_agent = ChunkPlanningAgent()
        self.embedding_agent = EmbeddingReadinessAgent()
    
    def qualify(self, structured_doc) -> DocumentQualificationResult:
        """
        Perform complete document qualification.
        
        Args:
            structured_doc: Structured document from Phase 1 (StructureAnalysis or dict)
            
        Returns:
            DocumentQualificationResult with scores, issues, and recommendations
        """
        # Convert StructureAnalysis to dict if needed
        if hasattr(structured_doc, 'to_dict'):
            doc_dict = structured_doc.to_dict()
            # Add blocks from structured_blocks for agents
            if 'structured_blocks' in doc_dict and not doc_dict.get('blocks'):
                doc_dict['blocks'] = doc_dict['structured_blocks']
        elif isinstance(structured_doc, dict):
            doc_dict = structured_doc
            # Ensure blocks key exists
            if 'blocks' not in doc_dict:
                doc_dict['blocks'] = doc_dict.get('structured_blocks', [])
        else:
            raise ValueError(f"Unsupported document type: {type(structured_doc)}")
        
        # Initialize result
        doc_id = doc_dict.get('document_id', 'unknown')
        
        # Run all agents
        structure_issues, structure_warnings = self.structure_agent.validate(structured_doc)
        semantic_issues, semantic_warnings = self.semantic_agent.validate(structured_doc)
        retrieval_issues, retrieval_warnings, retrieval_metadata = self.retrieval_agent.validate(structured_doc)
        chunk_issues, chunk_warnings, chunk_metadata = self.chunk_planning_agent.validate(structured_doc)
        embed_issues, embed_warnings, embedding_scores = self.embedding_agent.validate(
            structured_doc, retrieval_metadata
        )
        
        # Aggregate all issues and warnings
        all_issues = structure_issues + semantic_issues + retrieval_issues + chunk_issues + embed_issues
        all_warnings = structure_warnings + semantic_warnings + retrieval_warnings + chunk_warnings + embed_warnings
        
        # Calculate scores
        scores = self._calculate_overall_scores(
            all_issues, all_warnings, embedding_scores, structured_doc
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            all_issues, all_warnings, scores, chunk_metadata
        )
        
        # Determine qualification status
        is_chunkable = self._determine_chunkable(all_issues, scores)
        is_embeddable = self._determine_embeddable(all_issues, scores)
        is_retrievable = self._determine_retrievable(all_issues, scores, retrieval_metadata)
        
        # Generate chunking strategy
        chunking_strategy = self._generate_chunking_strategy(chunk_metadata, scores)
        
        # Build result
        result = DocumentQualificationResult(
            document_id=doc_id,
            is_chunkable=is_chunkable,
            is_embeddable=is_embeddable,
            is_retrievable=is_retrievable,
            overall_score=scores['overall_score'],
            structural_score=scores['structural_score'],
            semantic_score=scores['semantic_score'],
            retrieval_score=scores['retrieval_score'],
            embedding_readiness_score=scores['embedding_readiness_score'],
            issues=all_issues,
            warnings=all_warnings,
            recommendations=recommendations,
            chunking_strategy=chunking_strategy,
            metadata={
                "retrieval_metadata": retrieval_metadata,
                "chunk_metadata": chunk_metadata,
                "embedding_scores": embedding_scores,
                "issue_count": len(all_issues),
                "warning_count": len(all_warnings),
            }
        )
        
        return result
    
    def _calculate_overall_scores(
        self, 
        issues: list, 
        warnings: list, 
        embedding_scores: dict,
        structured_doc: dict
    ) -> Dict[str, float]:
        """Calculate overall qualification scores."""
        # Convert StructureAnalysis to dict if needed
        if hasattr(structured_doc, 'to_dict'):
            doc_dict = structured_doc.to_dict()
            blocks = doc_dict.get('structured_blocks', [])
        elif isinstance(structured_doc, dict):
            blocks = structured_doc.get('blocks', structured_doc.get('structured_blocks', []))
        else:
            blocks = []
        
        total_blocks = len(blocks)
        
        if total_blocks == 0:
            return {
                "overall_score": 0.0,
                "structural_score": 0.0,
                "semantic_score": 0.0,
                "retrieval_score": 0.0,
                "embedding_readiness_score": 0.0,
            }
        
        # Penalize for issues
        critical_issues = sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL)
        high_issues = sum(1 for i in issues if i.severity == IssueSeverity.HIGH)
        medium_issues = sum(1 for i in issues if i.severity == IssueSeverity.MEDIUM)
        
        issue_penalty = (
            critical_issues * 0.3 +
            high_issues * 0.15 +
            medium_issues * 0.05
        )
        warning_penalty = min(
            0.25,
            sum(self._severity_penalty(w.severity) for w in warnings),
        )
        
        # Base scores from embedding agent
        base_embedding_score = embedding_scores.get('embedding_readiness_score', 0.5)
        
        # Calculate component scores
        structural_warning_penalty = min(
            0.2,
            sum(self._severity_penalty(w.severity) for w in warnings if w.issue_type.value == 'structural'),
        )
        semantic_warning_penalty = min(
            0.2,
            sum(self._severity_penalty(w.severity) for w in warnings if w.issue_type.value == 'semantic'),
        )
        embedding_warning_penalty = min(
            0.2,
            sum(self._severity_penalty(w.severity) for w in warnings if w.issue_type.value == 'embedding_readiness'),
        )

        structural_score = max(0.0, 1.0 - issue_penalty - structural_warning_penalty - (len([i for i in issues if i.issue_type.value == 'structural']) * 0.05))
        semantic_score = max(0.0, 1.0 - issue_penalty - semantic_warning_penalty - (len([i for i in issues if i.issue_type.value == 'semantic']) * 0.05))
        retrieval_score = max(0.0, min(1.0, base_embedding_score * embedding_scores.get('retrieval_utility_score', 0.5)))
        embedding_readiness_score = max(0.0, min(1.0, base_embedding_score - issue_penalty * 0.5 - embedding_warning_penalty))
        
        # Overall score (weighted average)
        overall_score = (
            structural_score * 0.25 +
            semantic_score * 0.25 +
            retrieval_score * 0.25 +
            embedding_readiness_score * 0.25
        )
        overall_score = max(0.0, min(1.0, overall_score - warning_penalty * 0.25))
        
        return {
            "overall_score": round(overall_score, 3),
            "structural_score": round(structural_score, 3),
            "semantic_score": round(semantic_score, 3),
            "retrieval_score": round(retrieval_score, 3),
            "embedding_readiness_score": round(embedding_readiness_score, 3),
        }

    @staticmethod
    def _severity_penalty(severity) -> float:
        value = getattr(severity, "value", str(severity)).lower()
        return {
            "critical": 0.08,
            "high": 0.05,
            "medium": 0.025,
            "low": 0.01,
            "info": 0.003,
        }.get(value, 0.01)
    
    def _generate_recommendations(
        self,
        issues: list,
        warnings: list,
        scores: dict,
        chunk_metadata: dict
    ) -> list:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Critical issues first
        critical_issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        if critical_issues:
            recommendations.append(
                f"Fix {len(critical_issues)} critical issues before proceeding to chunking"
            )
        
        # Check scores
        if scores['structural_score'] < 0.7:
            recommendations.append(
                "Improve document structure: fix heading hierarchy and broken markdown"
            )
        
        if scores['semantic_score'] < 0.7:
            recommendations.append(
                "Remove noise content: navigation elements, badges, and boilerplate"
            )
        
        if scores['retrieval_score'] < 0.6:
            recommendations.append(
                "Enrich document with definitions, procedures, or API documentation"
            )
        
        # Chunking-specific recommendations
        anchoring_points = chunk_metadata.get('anchoring_points', [])
        code_anchors = [a for a in anchoring_points if a.get('type') == 'code' and not a.get('has_explanation')]
        if code_anchors:
            recommendations.append(
                f"Add explanations near {len(code_anchors)} code blocks for better context"
            )
        
        boundary_candidates = chunk_metadata.get('boundary_candidates', [])
        if len(boundary_candidates) > 10:
            recommendations.append(
                "Document has many topic boundaries; consider splitting into multiple files"
            )
        
        # General best practices
        warning_count = len(warnings)
        if scores['overall_score'] >= 0.8 and warning_count <= 5:
            recommendations.append(
                "Document is well-structured and ready for advanced chunking strategies"
            )
        elif scores['overall_score'] >= 0.8:
            recommendations.append(
                f"Document is mostly usable but has {warning_count} warnings to review before production embedding"
            )
        elif scores['overall_score'] >= 0.6:
            recommendations.append(
                "Document is acceptable but would benefit from cleanup before production use"
            )
        else:
            recommendations.append(
                "Significant document improvement needed before RAG ingestion"
            )
        
        return recommendations
    
    def _determine_chunkable(self, issues: list, scores: dict) -> bool:
        """Determine if document is ready for chunking."""
        critical_issues = sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL)
        return critical_issues == 0 and scores['structural_score'] >= 0.5
    
    def _determine_embeddable(self, issues: list, scores: dict) -> bool:
        """Determine if document is ready for embedding."""
        critical_issues = sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL)
        high_issues = sum(1 for i in issues if i.severity == IssueSeverity.HIGH)
        return critical_issues == 0 and high_issues <= 2 and scores['embedding_readiness_score'] >= 0.4
    
    def _determine_retrievable(self, issues: list, scores: dict, retrieval_metadata: dict) -> bool:
        """Determine if document will support good retrieval."""
        has_high_value_content = any([
            retrieval_metadata.get('has_definitions', False),
            retrieval_metadata.get('has_procedures', False),
            retrieval_metadata.get('has_api_docs', False),
            retrieval_metadata.get('has_troubleshooting', False),
        ])
        return has_high_value_content and scores['retrieval_score'] >= 0.4
    
    def _generate_chunking_strategy(
        self, 
        chunk_metadata: dict, 
        scores: dict
    ) -> ChunkingStrategy:
        """Generate recommended chunking strategy based on analysis."""
        anchoring_points = chunk_metadata.get('anchoring_points', [])
        boundary_candidates = chunk_metadata.get('boundary_candidates', [])
        
        # Determine strategy type based on content
        has_code = any(a.get('type') == 'code' for a in anchoring_points)
        has_tables = any(a.get('type') == 'table' for a in anchoring_points)
        has_lists = any(a.get('type') == 'list' for a in anchoring_points)
        
        if has_code and has_tables:
            strategy_type = "semantic_with_anchoring"
        elif has_code:
            strategy_type = "code_aware_semantic"
        elif has_tables:
            strategy_type = "structure_preserving"
        else:
            strategy_type = "adaptive_semantic"
        
        # Determine chunk sizes based on content density
        if scores['embedding_readiness_score'] > 0.7:
            min_chunk = 100
            max_chunk = 512
        else:
            min_chunk = 150
            max_chunk = 400
        
        # Build anchoring rules
        anchoring_rules = []
        if has_code:
            anchoring_rules.append("keep_code_with_explanation")
        if has_tables:
            anchoring_rules.append("keep_table_with_context")
        if has_lists:
            anchoring_rules.append("preserve_list_integrity")
        
        return ChunkingStrategy(
            strategy_type=strategy_type,
            min_chunk_size=min_chunk,
            max_chunk_size=max_chunk,
            preferred_boundary_types=["heading_level_1", "heading_level_2"],
            anchoring_rules=anchoring_rules,
            context_injection=True,
            parent_context_depth=2,
            merge_sparse_blocks=True,
            split_large_sections=scores['structural_score'] > 0.6,
            custom_parameters={
                "heading_path_injection": True,
                "semantic_density_threshold": 0.3,
                "entity_preservation": True,
            }
        )
