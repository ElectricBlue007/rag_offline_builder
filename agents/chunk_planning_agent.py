"""
Layer D & E: Chunk Planning & Embedding Readiness Agents
Validates chunk boundaries and embedding readiness.
Implements Rules 131-150.
"""

from typing import List, Dict, Any, Tuple
import math
from agents.models import ValidationIssue, IssueSeverity, IssueType, ChunkingStrategy


class ChunkPlanningAgent:
    """
    Plans optimal chunk boundaries based on content structure.
    
    Rules implemented:
    - Rule 131: Code anchoring
    - Rule 132: Table anchoring
    - Rule 133: List anchoring
    - Rule 134: Quote anchoring
    - Rule 135: Example anchoring
    - Rule 136: Heading context anchoring
    - Rule 137: Parent context anchoring
    - Rule 138: Sibling context detection
    - Rule 139: Continuation detection
    - Rule 140: Topic boundary candidates
    """
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []
        self.anchoring_points: List[Dict[str, Any]] = []
        self.boundary_candidates: List[Dict[str, Any]] = []
        
    def validate(self, structured_doc: Any) -> tuple[List[ValidationIssue], List[ValidationIssue], Dict[str, Any]]:
        """Perform chunk planning validation."""
        self.issues = []
        self.warnings = []
        self.anchoring_points = []
        self.boundary_candidates = []
        
        # Convert StructureAnalysis to dict if needed
        if hasattr(structured_doc, 'to_dict'):
            doc_dict = structured_doc.to_dict()
            blocks = doc_dict.get('structured_blocks', [])
        elif isinstance(structured_doc, dict):
            blocks = structured_doc.get('blocks', structured_doc.get('structured_blocks', []))
        else:
            return self.issues, self.warnings, {}
        
        if not blocks:
            return self.issues, self.warnings, {}
        
        # Detect anchoring points
        self._detect_code_anchors(blocks)  # Rule 131
        self._detect_table_anchors(blocks)  # Rule 132
        self._detect_list_anchors(blocks)  # Rule 133
        self._detect_quote_anchors(blocks)  # Rule 134
        self._detect_example_anchors(blocks)  # Rule 135
        
        # Detect context requirements
        self._detect_heading_context(blocks)  # Rule 136
        self._detect_parent_context(blocks)  # Rule 137
        self._detect_sibling_context(blocks)  # Rule 138
        
        # Detect boundaries
        self._detect_continuations(blocks)  # Rule 139
        self._detect_topic_boundaries(blocks)  # Rule 140
        
        metadata = {
            "anchoring_points": self.anchoring_points,
            "boundary_candidates": self.boundary_candidates,
        }
        
        return self.issues, self.warnings, metadata
    
    def _detect_code_anchors(self, blocks: List[Dict]) -> None:
        """Rule 131: Detect code blocks that need anchoring."""
        for i, block in enumerate(blocks):
            if block.get('type') == 'code':
                block_id = block.get('block_id')
                
                # Check if code has explanation nearby
                has_explanation = False
                for j in range(max(0, i-3), min(len(blocks), i+4)):
                    if j == i:
                        continue
                    neighbor = blocks[j]
                    if neighbor.get('type') == 'paragraph':
                        content = str(neighbor.get('content', '')).lower()
                        if any(kw in content for kw in ['example', 'usage', 'use', 'call', 'invoke']):
                            has_explanation = True
                            break
                
                self.anchoring_points.append({
                    "type": "code",
                    "block_id": block_id,
                    "index": i,
                    "has_explanation": has_explanation,
                    "anchor_rule": "must_keep_with_explanation"
                })
                
                if not has_explanation:
                    self.warnings.append(ValidationIssue(
                        rule_id=131,
                        rule_name="Code Without Explanation",
                        severity=IssueSeverity.LOW,
                        issue_type=IssueType.CHUNK_PLANNING,
                        description="Code block lacks nearby explanation",
                        block_id=block_id,
                        impact="Low - Code may be hard to understand in isolation",
                        recommended_action="Ensure code is chunked with explanatory text",
                        context={"block_index": i}
                    ))
    
    def _detect_table_anchors(self, blocks: List[Dict]) -> None:
        """Rule 132: Detect tables that need anchoring."""
        for i, block in enumerate(blocks):
            if block.get('type') == 'table':
                block_id = block.get('block_id')
                
                # Check if table has explanation nearby
                has_explanation = False
                for j in range(max(0, i-2), min(len(blocks), i+3)):
                    if j == i:
                        continue
                    neighbor = blocks[j]
                    if neighbor.get('type') == 'paragraph':
                        has_explanation = True
                        break
                
                self.anchoring_points.append({
                    "type": "table",
                    "block_id": block_id,
                    "index": i,
                    "has_explanation": has_explanation,
                    "anchor_rule": "must_keep_with_explanation"
                })
    
    def _detect_list_anchors(self, blocks: List[Dict]) -> None:
        """Rule 133: Detect lists that need anchoring."""
        for i, block in enumerate(blocks):
            if block.get('type') == 'list':
                block_id = block.get('block_id')
                metadata = block.get('metadata', {})
                items = metadata.get('items', [])
                
                self.anchoring_points.append({
                    "type": "list",
                    "block_id": block_id,
                    "index": i,
                    "item_count": len(items),
                    "anchor_rule": "preserve_enumeration"
                })
    
    def _detect_quote_anchors(self, blocks: List[Dict]) -> None:
        """Rule 134: Detect quotes that need anchoring."""
        for i, block in enumerate(blocks):
            if block.get('type') == 'quote':
                block_id = block.get('block_id')
                
                self.anchoring_points.append({
                    "type": "quote",
                    "block_id": block_id,
                    "index": i,
                    "anchor_rule": "preserve_context"
                })
    
    def _detect_example_anchors(self, blocks: List[Dict]) -> None:
        """Rule 135: Detect example sections."""
        for i, block in enumerate(blocks):
            content = str(block.get('content', '')).lower()
            
            if 'example' in content or block.get('type') == 'heading' and 'example' in content:
                block_id = block.get('block_id')
                
                self.anchoring_points.append({
                    "type": "example",
                    "block_id": block_id,
                    "index": i,
                    "anchor_rule": "keep_with_related_content"
                })
    
    def _detect_heading_context(self, blocks: List[Dict]) -> None:
        """Rule 136: Detect heading context requirements."""
        headings = [i for i, b in enumerate(blocks) if b.get('type') == 'heading']
        
        for idx in headings:
            block = blocks[idx]
            block_id = block.get('block_id')
            level = block.get('metadata', {}).get('level', 0)
            
            self.anchoring_points.append({
                "type": "heading",
                "block_id": block_id,
                "index": idx,
                "level": level,
                "anchor_rule": "inject_into_child_chunks"
            })
    
    def _detect_parent_context(self, blocks: List[Dict]) -> None:
        """Rule 137: Detect parent context requirements."""
        # Build heading hierarchy
        heading_stack = []
        
        for i, block in enumerate(blocks):
            if block.get('type') == 'heading':
                level = block.get('metadata', {}).get('level', 0)
                
                # Adjust stack
                while heading_stack and heading_stack[-1][1] >= level:
                    heading_stack.pop()
                
                heading_stack.append((i, level))
                
                # Record parent context requirement
                if len(heading_stack) > 1:
                    parent_idx, parent_level = heading_stack[-2]
                    self.anchoring_points.append({
                        "type": "parent_context",
                        "child_index": i,
                        "parent_index": parent_idx,
                        "parent_level": parent_level,
                        "anchor_rule": "inject_parent_heading_path"
                    })
    
    def _detect_sibling_context(self, blocks: List[Dict]) -> None:
        """Rule 138: Detect sibling context opportunities."""
        # Group consecutive similar blocks
        i = 0
        while i < len(blocks):
            block = blocks[i]
            block_type = block.get('type')
            
            if block_type in ['paragraph', 'list']:
                # Look for consecutive similar blocks
                siblings = [i]
                j = i + 1
                while j < len(blocks) and blocks[j].get('type') == block_type:
                    siblings.append(j)
                    j += 1
                
                if len(siblings) > 1:
                    self.anchoring_points.append({
                        "type": "sibling_group",
                        "block_type": block_type,
                        "indices": siblings,
                        "anchor_rule": "consider_merging"
                    })
                
                i = j
            else:
                i += 1
    
    def _detect_continuations(self, blocks: List[Dict]) -> None:
        """Rule 139: Detect continuation sections."""
        continuation_indicators = ['continued', 'part', 'section', 'chapter']
        
        for i, block in enumerate(blocks):
            if block.get('type') == 'heading':
                content = str(block.get('content', '')).lower()
                
                if any(ind in content for ind in continuation_indicators):
                    block_id = block.get('block_id')
                    
                    self.warnings.append(ValidationIssue(
                        rule_id=139,
                        rule_name="Continuation Section",
                        severity=IssueSeverity.INFO,
                        issue_type=IssueType.CHUNK_PLANNING,
                        description=f"Heading suggests continuation: '{block.get('content')}'",
                        block_id=block_id,
                        impact="Info - May need to merge with previous section",
                        recommended_action="Consider merging with parent topic",
                        context={"heading": block.get('content')}
                    ))
    
    def _detect_topic_boundaries(self, blocks: List[Dict]) -> None:
        """Rule 140: Detect potential topic boundaries."""
        # H1 and H2 are natural topic boundaries
        for i, block in enumerate(blocks):
            if block.get('type') == 'heading':
                level = block.get('metadata', {}).get('level', 0)
                
                if level <= 2:
                    block_id = block.get('block_id')
                    self.boundary_candidates.append({
                        "type": "heading",
                        "level": level,
                        "index": i,
                        "block_id": block_id,
                        "confidence": 0.9 if level == 1 else 0.7
                    })


class EmbeddingReadinessAgent:
    """
    Assesses document readiness for embedding.
    
    Rules implemented:
    - Rule 141: Minimum semantic mass
    - Rule 142: Maximum semantic mass
    - Rule 143: Entity density
    - Rule 144: Keyword density
    - Rule 145: Definition presence
    - Rule 146: Reference integrity
    - Rule 147: Chunk self-sufficiency score
    - Rule 148: Context dependency score
    - Rule 149: Retrieval utility score
    - Rule 150: Embedding readiness score
    """
    
    MIN_TOKENS_PER_CHUNK = 50
    MAX_TOKENS_PER_CHUNK = 700
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []
        self.scores: Dict[str, float] = {}
        
    def validate(self, structured_doc: Any, retrieval_metadata: Dict[str, Any] = None) -> tuple[List[ValidationIssue], List[ValidationIssue], Dict[str, Any]]:
        """Perform embedding readiness validation."""
        self.issues = []
        self.warnings = []
        self.scores = {}
        
        # Convert StructureAnalysis to dict if needed
        if hasattr(structured_doc, 'to_dict'):
            doc_dict = structured_doc.to_dict()
            blocks = doc_dict.get('structured_blocks', [])
        elif isinstance(structured_doc, dict):
            blocks = structured_doc.get('blocks', structured_doc.get('structured_blocks', []))
        else:
            return self.issues, self.warnings, {}
        
        if not blocks:
            return self.issues, self.warnings, {}
        
        # Rule 141 & 142: Semantic mass validation
        self._validate_semantic_mass(blocks)
        
        # Rule 143: Entity density
        self._assess_entity_density(blocks)
        
        # Rule 144: Keyword density
        self._assess_keyword_density(blocks)
        
        # Rule 145: Definition presence
        self._check_definition_presence(blocks, retrieval_metadata)
        
        # Rule 146: Reference integrity
        self._check_reference_integrity(blocks)
        
        # Rule 147-150: Calculate scores
        self._calculate_scores(blocks, retrieval_metadata)
        
        return self.issues, self.warnings, self.scores
    
    def _validate_semantic_mass(self, blocks: List[Dict]) -> None:
        """Rules 141 & 142: Validate token counts."""
        for block in blocks:
            content = str(block.get('content', ''))
            block_id = block.get('block_id')
            block_type = block.get('type')
            
            if block_type in ['heading', 'image', 'link']:
                continue
            
            token_estimate = len(content.split()) * 1.3  # Rough estimate
            
            # Rule 141: Too small
            if token_estimate < self.MIN_TOKENS_PER_CHUNK and token_estimate > 0:
                self.warnings.append(ValidationIssue(
                    rule_id=141,
                    rule_name="Below Minimum Semantic Mass",
                    severity=IssueSeverity.LOW,
                    issue_type=IssueType.EMBEDDING_READINESS,
                    description=f"Block has ~{int(token_estimate)} tokens (min: {self.MIN_TOKENS_PER_CHUNK})",
                    block_id=block_id,
                    impact="Low - May produce weak embeddings",
                    recommended_action="Merge with adjacent content before embedding",
                    context={"token_count": token_estimate}
                ))
            
            # Rule 142: Too large
            if token_estimate > self.MAX_TOKENS_PER_CHUNK:
                self.warnings.append(ValidationIssue(
                    rule_id=142,
                    rule_name="Above Maximum Semantic Mass",
                    severity=IssueSeverity.MEDIUM,
                    issue_type=IssueType.EMBEDDING_READINESS,
                    description=f"Block has ~{int(token_estimate)} tokens (max: {self.MAX_TOKENS_PER_CHUNK})",
                    block_id=block_id,
                    impact="Medium - May cause semantic dilution",
                    recommended_action="Split into smaller coherent chunks",
                    context={"token_count": token_estimate}
                ))
    
    def _assess_entity_density(self, blocks: List[Dict]) -> None:
        """Rule 143: Assess entity density."""
        import re
        
        # Pattern for potential entities (capitalized terms, technical terms)
        entity_pattern = r'\b[A-Z][a-zA-Z]+\b|\b\w+(?:Chunker|Embedding|Model|API)\b'
        
        for block in blocks:
            content = str(block.get('content', ''))
            block_id = block.get('block_id')
            
            if len(content) < 50:
                continue
            
            entities = re.findall(entity_pattern, content)
            word_count = len(content.split())
            
            if word_count > 0:
                entity_density = len(entities) / word_count
                
                # High entity density is generally good for RAG
                if entity_density > 0.1:
                    pass  # Good
                elif entity_density < 0.02 and word_count > 100:
                    self.warnings.append(ValidationIssue(
                        rule_id=143,
                        rule_name="Low Entity Density",
                        severity=IssueSeverity.LOW,
                        issue_type=IssueType.EMBEDDING_READINESS,
                        description=f"Block has low entity density: {entity_density:.2%}",
                        block_id=block_id,
                        impact="Low - May lack specific concepts for retrieval",
                        recommended_action="Consider adding specific terms or examples",
                        context={"entity_density": entity_density, "entity_count": len(entities)}
                    ))
    
    def _assess_keyword_density(self, blocks: List[Dict]) -> None:
        """Rule 144: Assess keyword density."""
        # Domain-specific keywords would go here
        pass
    
    def _check_definition_presence(self, blocks: List[Dict], retrieval_metadata: Dict = None) -> None:
        """Rule 145: Check for definition presence."""
        if retrieval_metadata:
            has_definitions = retrieval_metadata.get('has_definitions', False)
            
            if not has_definitions:
                # Check manually
                for block in blocks:
                    content = str(block.get('content', '')).lower()
                    if 'is a' in content or 'are' in content or 'defined as' in content:
                        has_definitions = True
                        break
            
            if not has_definitions:
                self.warnings.append(ValidationIssue(
                    rule_id=145,
                    rule_name="No Definitions Found",
                    severity=IssueSeverity.LOW,
                    issue_type=IssueType.EMBEDDING_READINESS,
                    description="Document lacks explicit definitions",
                    impact="Low - May reduce answer quality for 'what is' queries",
                    recommended_action="Add definitions for key concepts"
                ))
    
    def _check_reference_integrity(self, blocks: List[Dict]) -> None:
        """Rule 146: Check reference integrity."""
        # Already handled in retrieval agent
        pass
    
    def _calculate_scores(self, blocks: List[Dict], retrieval_metadata: Dict = None) -> None:
        """Rules 147-150: Calculate various scores."""
        semantic_blocks = [b for b in blocks if b.get('type') not in ['heading', 'image']]
        total_blocks = len(semantic_blocks)
        
        if total_blocks == 0:
            self.scores = {
                "self_sufficiency_score": 0.0,
                "context_dependency_score": 1.0,
                "retrieval_utility_score": 0.0,
                "embedding_readiness_score": 0.0,
            }
            return
        
        # Rule 147: Self-sufficiency score
        # Higher if blocks have context (headings, explanations)
        blocks_with_context = sum(
            1 for b in semantic_blocks
            if b.get('metadata', {}).get('heading_path') or 
               b.get('type') in ['code', 'table']
        )
        self.scores['self_sufficiency_score'] = min(1.0, blocks_with_context / max(total_blocks, 1))
        
        # Rule 148: Context dependency score (lower is better)
        standalone_blocks = sum(
            1 for b in semantic_blocks
            if len(str(b.get('content', '')).split()) > 20
        )
        self.scores['context_dependency_score'] = max(0.0, 1.0 - (standalone_blocks / max(total_blocks, 1)))
        
        # Rule 149: Retrieval utility score
        if retrieval_metadata:
            high_value_types = sum(
                len(retrieval_metadata.get('content_types', {}).get(t, []))
                for t in ['definition', 'procedure', 'api', 'troubleshooting']
            )
            self.scores['retrieval_utility_score'] = min(1.0, high_value_types / max(total_blocks, 1) * 2)
        else:
            self.scores['retrieval_utility_score'] = 0.5
        
        # Rule 150: Overall embedding readiness score
        self.scores['embedding_readiness_score'] = (
            self.scores['self_sufficiency_score'] * 0.3 +
            (1.0 - self.scores['context_dependency_score']) * 0.3 +
            self.scores['retrieval_utility_score'] * 0.4
        )
        self.scores['embedding_readiness_score'] = max(0.0, min(1.0, self.scores['embedding_readiness_score']))
