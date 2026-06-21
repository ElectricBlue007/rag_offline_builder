"""
Layer C: Retrieval Validation Agent
Validates document blocks for retrieval utility.
Implements Rules 121-130.
"""

from typing import List, Dict, Any
import re
from agents.models import ValidationIssue, IssueSeverity, IssueType


class RetrievalAgent:
    """
    Validates retrieval utility of document blocks.
    
    Rules implemented:
    - Rule 121: Answerability detection
    - Rule 122: Definition detection
    - Rule 123: Procedure detection
    - Rule 124: API documentation detection
    - Rule 125: Requirements detection
    - Rule 126: Configuration detection
    - Rule 127: Troubleshooting detection
    - Rule 128: FAQ detection
    - Rule 129: Reference detection
    - Rule 130: Cross-reference detection
    """
    
    # Patterns for different content types
    DEFINITION_PATTERNS = [
        r'\b(is|are|refers? to|means?|defined as|consists of)\b',
        r'^[A-Z][a-z]+ (is|are) ',
        r':\s*(a|an|the)\s+\w+\s+(that|which|who)',
    ]
    
    PROCEDURE_PATTERNS = [
        r'\b(step|steps|first|then|next|finally)\b',
        r'\b(install|configure|setup|initialize|create|build|deploy)\b',
        r'^\d+\.\s+',
        r'\b(run|execute|start|stop|restart)\b',
    ]
    
    API_PATTERNS = [
        r'\b(GET|POST|PUT|DELETE|PATCH)\s+/',
        r'\b(endpoint|request|response|parameter|returns?)\b',
        r'`\w+\(\)`',
        r'\b(method|function|class|interface)\s+\w+',
        r'@param\s+',
        r'@returns?\s+',
        r'@throws?\s+',
    ]
    
    REQUIREMENT_PATTERNS = [
        r'\b(must|should|may|required|necessary|prerequisite)\b',
        r'\b(dependencies?|requirements?|needs?)\b',
        r'^[-*]\s+(install|require|need)',
    ]
    
    CONFIGURATION_PATTERNS = [
        r'\b(config|configuration|settings?|options?|parameters?)\b',
        r'\.yaml|\.yml|\.json|\.toml|\.ini|\.env',
        r'\b(environment variable|env var)\b',
    ]
    
    TROUBLESHOOTING_PATTERNS = [
        r'\b(problem|issue|error|bug|fix|solution|troubleshoot)\b',
        r'\b(fail|failed|failure|exception|traceback)\b',
        r'\b(why|how to fix|what to do)\b',
        r'Error\s*\d*:?',
    ]
    
    FAQ_PATTERNS = [
        r'\b(Question|Q:|FAQ|frequently asked)\b',
        r'\b(Answer|A:|Response)\b',
        r'^\?+\s*',
        r'\b(how|what|when|where|why|which)\s+\w+\?',
    ]
    
    CROSS_REF_PATTERNS = [
        r'\b(see|refer to|see also|cf\.)\s+',
        r'\b(section|chapter|page|above|below)\b',
        r'\[\[.*?\]\]',
        r'{@link\s+.*?}',
        r'<(?:x)?ref\s+.*?>',
    ]
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []
        self.content_types: Dict[str, List[str]] = {}
        
    def validate(self, structured_doc: Any) -> tuple[List[ValidationIssue], List[ValidationIssue], Dict[str, Any]]:
        """
        Perform retrieval validation on a structured document.
        
        Args:
            structured_doc: Document with typed blocks from Phase 1
            
        Returns:
            Tuple of (issues, warnings, content_type_map)
        """
        self.issues = []
        self.warnings = []
        self.content_types = {}
        
        blocks = structured_doc.get('blocks', [])
        if not blocks:
            return self.issues, self.warnings, self.content_types
        
        # Analyze each block for content type
        for block in blocks:
            self._analyze_block_content_type(block)
        
        # Rule 121: Check answerability
        self._assess_answerability(blocks)
        
        # Rule 130: Check cross-references
        self._detect_cross_references(blocks)
        
        # Generate recommendations based on content type distribution
        self._generate_retrieval_recommendations()
        
        metadata = {
            "content_types": self.content_types,
            "has_definitions": len(self.content_types.get('definition', [])) > 0,
            "has_procedures": len(self.content_types.get('procedure', [])) > 0,
            "has_api_docs": len(self.content_types.get('api', [])) > 0,
            "has_troubleshooting": len(self.content_types.get('troubleshooting', [])) > 0,
            "has_faq": len(self.content_types.get('faq', [])) > 0,
        }
        
        return self.issues, self.warnings, metadata
    
    def _analyze_block_content_type(self, block: Dict) -> None:
        """Analyze a block to determine its content type."""
        content = str(block.get('content', '')).lower()
        block_id = block.get('block_id')
        block_type = block.get('type')
        
        if not content or block_type in ['heading', 'image']:
            return
        
        detected_types = []
        
        # Rule 122: Definition detection
        if any(re.search(p, content) for p in self.DEFINITION_PATTERNS):
            detected_types.append('definition')
            self.content_types.setdefault('definition', []).append(block_id)
        
        # Rule 123: Procedure detection
        if any(re.search(p, content) for p in self.PROCEDURE_PATTERNS):
            detected_types.append('procedure')
            self.content_types.setdefault('procedure', []).append(block_id)
        
        # Rule 124: API documentation detection
        if any(re.search(p, content) for p in self.API_PATTERNS):
            detected_types.append('api')
            self.content_types.setdefault('api', []).append(block_id)
        
        # Rule 125: Requirements detection
        if any(re.search(p, content) for p in self.REQUIREMENT_PATTERNS):
            detected_types.append('requirement')
            self.content_types.setdefault('requirement', []).append(block_id)
        
        # Rule 126: Configuration detection
        if any(re.search(p, content) for p in self.CONFIGURATION_PATTERNS):
            detected_types.append('configuration')
            self.content_types.setdefault('configuration', []).append(block_id)
        
        # Rule 127: Troubleshooting detection
        if any(re.search(p, content) for p in self.TROUBLESHOOTING_PATTERNS):
            detected_types.append('troubleshooting')
            self.content_types.setdefault('troubleshooting', []).append(block_id)
        
        # Rule 128: FAQ detection
        if any(re.search(p, content) for p in self.FAQ_PATTERNS):
            detected_types.append('faq')
            self.content_types.setdefault('faq', []).append(block_id)
        
        # Rule 129: Reference detection
        if block_type == 'link' or 'reference' in content:
            detected_types.append('reference')
            self.content_types.setdefault('reference', []).append(block_id)
        
        # Store detected types in block metadata
        if detected_types:
            block.setdefault('metadata', {})['content_types'] = detected_types
    
    def _assess_answerability(self, blocks: List[Dict]) -> None:
        """Rule 121: Assess if blocks can answer questions."""
        high_value_types = {'definition', 'procedure', 'api', 'troubleshooting', 'faq'}
        
        for block in blocks:
            block_id = block.get('block_id')
            content_types = set(block.get('metadata', {}).get('content_types', []))
            content = str(block.get('content', '')).strip()
            
            # Skip very short blocks
            if len(content.split()) < 10:
                continue
            
            # Check if block has high retrieval value
            if not content_types.intersection(high_value_types):
                # Block may not be directly answerable
                # This is informational, not necessarily an issue
                pass
    
    def _detect_cross_references(self, blocks: List[Dict]) -> None:
        """Rule 130: Detect cross-references that may need resolution."""
        for block in blocks:
            content = str(block.get('content', ''))
            block_id = block.get('block_id')
            
            # Check for cross-reference patterns
            has_cross_ref = any(
                re.search(p, content, re.IGNORECASE) 
                for p in self.CROSS_REF_PATTERNS
            )
            
            if has_cross_ref:
                # Extract potential references
                refs = []
                for pattern in [r'see\s+(?:the\s+)?([A-Z][\w\s]+)', r'refer\s+to\s+([A-Z][\w\s]+)']:
                    matches = re.findall(pattern, content)
                    refs.extend(matches)
                
                if refs:
                    self.warnings.append(ValidationIssue(
                        rule_id=130,
                        rule_name="Cross-Reference Detected",
                        severity=IssueSeverity.INFO,
                        issue_type=IssueType.RETRIEVAL,
                        description=f"Block contains cross-references: {refs}",
                        block_id=block_id,
                        impact="Info - May need context injection for standalone understanding",
                        recommended_action="Consider injecting referenced content or adding context",
                        context={"references": refs}
                    ))
    
    def _generate_retrieval_recommendations(self) -> None:
        """Generate recommendations based on content type analysis."""
        # Check for balanced content types
        if not self.content_types:
            return
        
        total_blocks = sum(len(v) for v in self.content_types.values())
        
        # Check if document has good retrieval content
        high_value_count = sum(
            len(self.content_types.get(t, [])) 
            for t in ['definition', 'procedure', 'api', 'troubleshooting']
        )
        
        if high_value_count == 0 and total_blocks > 5:
            self.warnings.append(ValidationIssue(
                rule_id=121,
                rule_name="Low Retrieval Value",
                severity=IssueSeverity.MEDIUM,
                issue_type=IssueType.RETRIEVAL,
                description="Document lacks high-value retrieval content (definitions, procedures, API docs)",
                impact="Medium - Document may not answer user queries effectively",
                recommended_action="Enrich document with definitions, examples, or procedural content",
                context={"content_types_found": list(self.content_types.keys())}
            ))
