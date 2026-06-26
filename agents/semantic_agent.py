"""
Layer B: Semantic Validation Agent
Validates semantic quality, information density, and content types.
Implements Rules 111-120.
"""

from __future__ import annotations

from typing import List, Dict, Any, Set
import re
from agents.models import (
    ValidationIssue, IssueSeverity, IssueType, SectionAnalysis
)


class SemanticAgent:
    """
    Validates semantic quality of document blocks.
    
    Rules implemented:
    - Rule 111: Empty semantic blocks
    - Rule 112: Near-empty blocks
    - Rule 113: Low information density
    - Rule 114: Boilerplate detection
    - Rule 115: Navigation detection
    - Rule 116: Footer detection
    - Rule 117: Header noise detection
    - Rule 118: Badge detection
    - Rule 119: Repeated content detection
    - Rule 120: Duplicate section detection
    """
    
    # Common boilerplate phrases
    BOILERPLATE_PHRASES = {
        "contributing", "license", "acknowledgements", "changelog",
        "see also", "related projects", "back to top", "table of contents",
        "copyright", "all rights reserved", "made with", "powered by"
    }
    
    # Navigation patterns
    NAVIGATION_PATTERNS = {
        r'^\s*(home|docs|api|guide|tutorial|examples)\s*$',
        r'^\s*(previous|next|up|top|bottom)\s*$',
        r'^\s*(←|→|↑|↓)\s*',
        r'^\s*\[\s*(home|docs|api)\s*\]',
        r'^\s*edit this page\s*$',
        r'^\s*view on github\s*$',
        r'^\s*open issue\s*$',
    }
    
    # Badge patterns
    BADGE_PATTERNS = {
        r'!\[.*?\]\(.*?(badge|shields|travis|appveyor|codecov).*?\)',
        r'\[!\[.*?\]\(.*?\)\]\(.*?\)',
        r'!\[PyPI\]', r'!\[npm\]', r'!\[Downloads\]',
        r'!\[Build Status\]', r'!\[Coverage Status\]',
        r'!\[GitHub stars\]', r'!\[forks\]',
        r'!\[Discord\]', r'!\[Slack\]',
    }
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []
        self.content_hashes: Dict[str, List[str]] = {}
        
    def validate(self, structured_doc: Any) -> tuple[List[ValidationIssue], List[ValidationIssue]]:
        """
        Perform semantic validation on a structured document.
        
        Args:
            structured_doc: Document with typed blocks from Phase 1
            
        Returns:
            Tuple of (issues, warnings)
        """
        self.issues = []
        self.warnings = []
        self.content_hashes = {}
        
        # Convert StructureAnalysis to dict if needed
        if hasattr(structured_doc, 'to_dict'):
            doc_dict = structured_doc.to_dict()
            blocks = doc_dict.get('structured_blocks', [])
        elif isinstance(structured_doc, dict):
            blocks = structured_doc.get('blocks', structured_doc.get('structured_blocks', []))
        else:
            return self.issues, self.warnings
        
        if not blocks:
            return self.issues, self.warnings
        
        # Rule 111 & 112: Empty and near-empty blocks
        self._detect_empty_blocks(blocks)
        
        # Rule 113: Low information density
        self._assess_information_density(blocks)
        
        # Rule 114: Boilerplate detection
        self._detect_boilerplate(blocks)
        
        # Rule 115: Navigation detection
        self._detect_navigation(blocks)
        
        # Rule 116: Footer detection
        self._detect_footer(blocks)
        
        # Rule 117: Header noise detection
        self._detect_header_noise(blocks)
        
        # Rule 118: Badge detection
        self._detect_badges(blocks)
        
        # Rule 119 & 120: Repeated and duplicate content
        self._detect_repeated_content(blocks)
        
        return self.issues, self.warnings
    
    def _detect_empty_blocks(self, blocks: List[Dict]) -> None:
        """Rules 111 & 112: Detect empty and near-empty blocks."""
        for block in blocks:
            block_type = block.get('type')
            content = str(block.get('content', '')).strip()
            block_id = block.get('block_id')
            
            # Skip structural blocks
            if block_type in ['heading', 'image', 'link']:
                continue
            
            # Rule 111: Completely empty
            if not content:
                if block_type == 'paragraph':
                    self.warnings.append(ValidationIssue(
                        rule_id=111,
                        rule_name="Empty Semantic Block",
                        severity=IssueSeverity.LOW,
                        issue_type=IssueType.SEMANTIC,
                        description=f"Empty {block_type} block detected",
                        block_id=block_id,
                        impact="Low - Wastes embedding budget",
                        recommended_action="Remove empty block or add content",
                        context={"block_type": block_type}
                    ))
                continue
            
            # Rule 112: Near-empty (very short content)
            word_count = len(content.split())
            if word_count <= 3 and block_type == 'paragraph':
                self.warnings.append(ValidationIssue(
                    rule_id=112,
                    rule_name="Near-Empty Block",
                    severity=IssueSeverity.LOW,
                    issue_type=IssueType.SEMANTIC,
                    description=f"Block has only {word_count} words: '{content[:50]}'",
                    block_id=block_id,
                    impact="Low - Insufficient semantic content",
                    recommended_action="Merge with adjacent block or expand content",
                    context={"word_count": word_count, "content": content}
                ))
    
    def _assess_information_density(self, blocks: List[Dict]) -> None:
        """Rule 113: Assess information density of blocks."""
        # Keywords that indicate high information density
        high_density_keywords = {
            'define', 'definition', 'example', 'usage', 'parameter',
            'return', 'raises', 'throws', 'config', 'configuration',
            'install', 'setup', 'initialize', 'create', 'build',
            'endpoint', 'request', 'response', 'method', 'class',
            'function', 'argument', 'attribute', 'property'
        }
        
        for block in blocks:
            if block.get('type') not in ['paragraph', 'code', 'list']:
                continue
                
            content = str(block.get('content', '')).lower()
            block_id = block.get('block_id')
            
            # Count density indicators
            density_score = sum(1 for keyword in high_density_keywords if keyword in content)
            word_count = len(content.split())
            
            # Very low density for substantial blocks
            if word_count > 20 and density_score == 0:
                # Check if it's just filler text
                filler_patterns = [
                    r'^this (section|page|document) (describes|explains|covers)',
                    r'^the following (sections|examples|steps)',
                    r'^please (note|see|refer)',
                    r'^for more information',
                ]
                
                is_filler = any(re.match(p, content) for p in filler_patterns)
                
                if is_filler:
                    self.warnings.append(ValidationIssue(
                        rule_id=113,
                        rule_name="Low Information Density",
                        severity=IssueSeverity.LOW,
                        issue_type=IssueType.SEMANTIC,
                        description="Block contains low-information filler text",
                        block_id=block_id,
                        impact="Low - Reduces retrieval quality",
                        recommended_action="Replace with substantive content or remove",
                        context={"word_count": word_count, "density_score": density_score}
                    ))
    
    def _detect_boilerplate(self, blocks: List[Dict]) -> None:
        """Rule 114: Detect boilerplate content."""
        for block in blocks:
            content = str(block.get('content', '')).lower()
            block_id = block.get('block_id')
            block_type = block.get('type')
            
            if block_type != 'paragraph':
                continue
            
            # Check for boilerplate phrases
            found_phrases = [
                phrase for phrase in self.BOILERPLATE_PHRASES 
                if phrase in content
            ]
            
            if len(found_phrases) >= 2:
                self.warnings.append(ValidationIssue(
                    rule_id=114,
                    rule_name="Boilerplate Content",
                    severity=IssueSeverity.LOW,
                    issue_type=IssueType.SEMANTIC,
                    description=f"Block contains boilerplate phrases: {found_phrases}",
                    block_id=block_id,
                    impact="Low - May dilute retrieval signal",
                    recommended_action="Consider marking as low-priority for embedding",
                    context={"phrases": found_phrases}
                ))
    
    def _detect_navigation(self, blocks: List[Dict]) -> None:
        """Rule 115: Detect navigation elements."""
        for block in blocks:
            content = str(block.get('content', '')).strip()
            block_id = block.get('block_id')
            
            # Check against navigation patterns
            is_navigation = any(
                re.match(pattern, content, re.IGNORECASE) 
                for pattern in self.NAVIGATION_PATTERNS
            )
            
            if is_navigation:
                self.issues.append(ValidationIssue(
                    rule_id=115,
                    rule_name="Navigation Element",
                    severity=IssueSeverity.MEDIUM,
                    issue_type=IssueType.SEMANTIC,
                    description=f"Navigation element detected: '{content[:50]}'",
                    block_id=block_id,
                    impact="Medium - Navigation poison for RAG",
                    recommended_action="Remove navigation elements before embedding",
                    context={"content": content}
                ))
    
    def _detect_footer(self, blocks: List[Dict]) -> None:
        """Rule 116: Detect footer content."""
        footer_indicators = {
            'copyright', '©', 'all rights reserved', 'privacy policy',
            'terms of service', 'last updated', 'generated by',
            'powered by', 'built with', 'hosted on'
        }
        
        # Check last few blocks for footer
        for block in blocks[-5:]:
            content = str(block.get('content', '')).lower()
            block_id = block.get('block_id')
            
            found_indicators = [
                ind for ind in footer_indicators if ind in content
            ]
            
            if len(found_indicators) >= 2:
                self.warnings.append(ValidationIssue(
                    rule_id=116,
                    rule_name="Footer Content",
                    severity=IssueSeverity.LOW,
                    issue_type=IssueType.SEMANTIC,
                    description=f"Footer content detected: {found_indicators}",
                    block_id=block_id,
                    impact="Low - Footer rarely useful for retrieval",
                    recommended_action="Mark footer content as low-priority",
                    context={"indicators": found_indicators}
                ))
    
    def _detect_header_noise(self, blocks: List[Dict]) -> None:
        """Rule 117: Detect header noise."""
        # Check first few blocks for header noise
        for block in blocks[:5]:
            content = str(block.get('content', '')).lower()
            block_id = block.get('block_id')
            block_type = block.get('type')
            
            # Skip actual headings
            if block_type == 'heading':
                continue
            
            header_noise_patterns = [
                r'^skip to (main )?content',
                r'^skip navigation',
                r'^menu',
                r'^search',
            ]
            
            is_noise = any(
                re.match(pattern, content) 
                for pattern in header_noise_patterns
            )
            
            if is_noise:
                self.warnings.append(ValidationIssue(
                    rule_id=117,
                    rule_name="Header Noise",
                    severity=IssueSeverity.LOW,
                    issue_type=IssueType.SEMANTIC,
                    description=f"Header noise detected: '{content[:50]}'",
                    block_id=block_id,
                    impact="Low - Accessibility/navigation text not useful for RAG",
                    recommended_action="Remove header noise before processing",
                    context={"content": content}
                ))
    
    def _detect_badges(self, blocks: List[Dict]) -> None:
        """Rule 118: Detect badge images."""
        for block in blocks:
            content = str(block.get('content', ''))
            block_id = block.get('block_id')
            
            # Check for badge patterns
            has_badge = any(
                re.search(pattern, content) 
                for pattern in self.BADGE_PATTERNS
            )
            
            if has_badge:
                self.warnings.append(ValidationIssue(
                    rule_id=118,
                    rule_name="Badge Detected",
                    severity=IssueSeverity.LOW,
                    issue_type=IssueType.SEMANTIC,
                    description="Badge image detected in content",
                    block_id=block_id,
                    impact="Low - Badges are visual noise for RAG",
                    recommended_action="Filter badges during normalization",
                    context={"content_preview": content[:100]}
                ))
    
    def _detect_repeated_content(self, blocks: List[Dict]) -> None:
        """Rules 119 & 120: Detect repeated and duplicate content."""
        content_to_blocks: Dict[str, List[str]] = {}
        
        for block in blocks:
            content = str(block.get('content', '')).strip()
            block_id = block.get('block_id')
            
            if not content or len(content) < 50:
                continue
            
            # Normalize content for comparison
            normalized = ' '.join(content.lower().split())
            
            # Simple hash-based deduplication
            content_key = normalized[:200]  # Use first 200 chars for key
            
            if content_key not in content_to_blocks:
                content_to_blocks[content_key] = []
            content_to_blocks[content_key].append(block_id)
        
        # Find duplicates
        for content_key, block_ids in content_to_blocks.items():
            if len(block_ids) > 1:
                self.warnings.append(ValidationIssue(
                    rule_id=119 if len(block_ids) == 2 else 120,
                    rule_name="Repeated Content" if len(block_ids) == 2 else "Duplicate Section",
                    severity=IssueSeverity.MEDIUM if len(block_ids) > 2 else IssueSeverity.LOW,
                    issue_type=IssueType.SEMANTIC,
                    description=f"Content appears {len(block_ids)} times in document",
                    impact="Medium - Wastes embedding budget and confuses retrieval",
                    recommended_action="Keep single instance or consolidate repeated content",
                    context={"occurrence_count": len(block_ids), "block_ids": block_ids[:5]}
                ))
