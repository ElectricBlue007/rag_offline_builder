"""
Layer A: Structural Validation Agent
Validates document structure, heading hierarchy, and markdown integrity.
Implements Rules 101-110.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from agents.models import (
    ValidationIssue, IssueSeverity, IssueType, 
    SectionAnalysis, DocumentQualificationResult
)


class StructureAgent:
    """
    Validates structural integrity of markdown documents.
    
    Rules implemented:
    - Rule 101: Heading level gaps detection
    - Rule 102: Heading hierarchy validation
    - Rule 103: Empty heading detection
    - Rule 104: Section length validation (too small)
    - Rule 105: Section length outlier detection (too large)
    - Rule 106: Document order validation
    - Rule 107: Broken markdown code blocks
    - Rule 108: Broken table detection
    - Rule 109: Broken link detection
    - Rule 110: Broken image reference detection
    """
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []
        
    def validate(self, structured_doc: Any) -> tuple[List[ValidationIssue], List[ValidationIssue]]:
        """
        Perform structural validation on a structured document.
        
        Args:
            structured_doc: Document with typed blocks from Phase 1
            
        Returns:
            Tuple of (issues, warnings)
        """
        self.issues = []
        self.warnings = []
        
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
            
        # Rule 101 & 102: Heading hierarchy validation
        self._validate_heading_hierarchy(blocks)
        
        # Rule 103: Empty headings
        self._detect_empty_headings(blocks)
        
        # Rule 104 & 105: Section length validation
        self._validate_section_lengths(blocks)
        
        # Rule 106: Document order validation
        self._validate_document_order(blocks)
        
        # Rule 107: Broken code blocks
        self._detect_broken_code_blocks(structured_doc)
        
        # Rule 108: Broken tables
        self._detect_broken_tables(blocks)
        
        # Rule 109: Broken links
        self._detect_broken_links(blocks)
        
        # Rule 110: Broken image references
        self._detect_broken_images(blocks)
        
        return self.issues, self.warnings
    
    def _validate_heading_hierarchy(self, blocks: List[Dict]) -> None:
        """Rules 101 & 102: Validate heading levels and hierarchy."""
        headings = [b for b in blocks if b.get('type') == 'heading']
        
        if not headings:
            return
            
        prev_level = 0
        for i, heading in enumerate(headings):
            level = heading.get('metadata', {}).get('level', 0)
            title = heading.get('content', '').strip()
            block_id = heading.get('block_id')
            
            # Rule 101: Check for skipped levels
            if level > prev_level + 1 and prev_level > 0:
                self.issues.append(ValidationIssue(
                    rule_id=101,
                    rule_name="Heading Level Gap",
                    severity=IssueSeverity.MEDIUM,
                    issue_type=IssueType.STRUCTURAL,
                    description=f"Heading level jumps from H{prev_level} to H{level}",
                    location=title,
                    block_id=block_id,
                    impact="Medium - May confuse hierarchical navigation",
                    recommended_action=f"Add intermediate H{prev_level+1} heading or adjust level to H{prev_level+1}",
                    context={"previous_level": prev_level, "current_level": level}
                ))
            
            # Rule 102: First heading should be H1 or H2
            if i == 0 and level > 2:
                self.warnings.append(ValidationIssue(
                    rule_id=102,
                    rule_name="Heading Hierarchy Start",
                    severity=IssueSeverity.LOW,
                    issue_type=IssueType.STRUCTURAL,
                    description=f"Document starts with H{level} instead of H1 or H2",
                    location=title,
                    block_id=block_id,
                    impact="Low - Document structure may be unclear",
                    recommended_action="Start document with H1 or H2 heading",
                    context={"level": level}
                ))
            
            prev_level = level
    
    def _detect_empty_headings(self, blocks: List[Dict]) -> None:
        """Rule 103: Detect headings without content."""
        headings = [b for b in blocks if b.get('type') == 'heading']
        
        for i, heading in enumerate(headings):
            block_id = heading.get('block_id')
            title = heading.get('content', '').strip()
            heading_index = blocks.index(heading)
            
            # Check if next block is another heading (no content in between)
            if heading_index + 1 < len(blocks):
                next_block = blocks[heading_index + 1]
                if next_block.get('type') == 'heading':
                    self.warnings.append(ValidationIssue(
                        rule_id=103,
                        rule_name="Empty Heading",
                        severity=IssueSeverity.MEDIUM,
                        issue_type=IssueType.STRUCTURAL,
                        description=f"Heading '{title}' has no content before next heading",
                        location=title,
                        block_id=block_id,
                        impact="Medium - Section appears empty",
                        recommended_action="Add content to section or merge with adjacent section",
                        context={"heading": title}
                    ))
    
    def _validate_section_lengths(self, blocks: List[Dict]) -> None:
        """Rules 104 & 105: Validate section token counts."""
        MIN_SECTION_TOKENS = 50
        MAX_SECTION_TOKENS = 2000
        
        sections = self._group_blocks_by_section(blocks)
        
        for section_id, section_blocks in sections.items():
            if not section_blocks:
                continue
                
            # Estimate token count (rough approximation)
            total_tokens = sum(
                len(str(b.get('content', ''))) // 4 
                for b in section_blocks
            )
            
            heading_block = next(
                (b for b in section_blocks if b.get('type') == 'heading'), 
                None
            )
            
            if heading_block:
                title = heading_block.get('content', '')
                block_id = heading_block.get('block_id')
                
                # Rule 104: Too small
                if total_tokens < MIN_SECTION_TOKENS and total_tokens > 0:
                    self.warnings.append(ValidationIssue(
                        rule_id=104,
                        rule_name="Section Too Small",
                        severity=IssueSeverity.LOW,
                        issue_type=IssueType.STRUCTURAL,
                        description=f"Section '{title}' has only ~{total_tokens} tokens",
                        location=title,
                        block_id=block_id,
                        impact="Low - Section may lack sufficient context",
                        recommended_action="Consider merging with parent or sibling section",
                        context={"token_count": total_tokens, "min_threshold": MIN_SECTION_TOKENS}
                    ))
                
                # Rule 105: Too large
                if total_tokens > MAX_SECTION_TOKENS:
                    self.warnings.append(ValidationIssue(
                        rule_id=105,
                        rule_name="Section Too Large",
                        severity=IssueSeverity.MEDIUM,
                        issue_type=IssueType.STRUCTURAL,
                        description=f"Section '{title}' has ~{total_tokens} tokens",
                        location=title,
                        block_id=block_id,
                        impact="Medium - Section may need splitting for better retrieval",
                        recommended_action="Consider splitting into subsections",
                        context={"token_count": total_tokens, "max_threshold": MAX_SECTION_TOKENS}
                    ))
    
    def _validate_document_order(self, blocks: List[Dict]) -> None:
        """Rule 106: Validate document order IDs are sequential."""
        orders = [b.get('order', 0) for b in blocks if 'order' in b]
        
        if not orders:
            return
            
        for i in range(1, len(orders)):
            if orders[i] <= orders[i-1]:
                self.issues.append(ValidationIssue(
                    rule_id=106,
                    rule_name="Document Order Broken",
                    severity=IssueSeverity.HIGH,
                    issue_type=IssueType.STRUCTURAL,
                    description=f"Block order is not sequential: {orders[i-1]} -> {orders[i]}",
                    impact="High - Document structure may be corrupted",
                    recommended_action="Re-process document through Phase 1 normalizer",
                    context={"previous_order": orders[i-1], "current_order": orders[i]}
                ))
    
    def _detect_broken_code_blocks(self, structured_doc: Any) -> None:
        """Rule 107: Detect unclosed code blocks."""
        # In structured doc, code blocks should already be parsed
        # Check for any raw markdown that suggests broken code
        raw_content = structured_doc.get('raw_content', '') if isinstance(structured_doc, dict) else ''
        
        # Count opening and closing code fences
        open_fences = raw_content.count('```')
        
        if open_fences % 2 != 0:
            self.issues.append(ValidationIssue(
                rule_id=107,
                rule_name="Broken Code Block",
                severity=IssueSeverity.HIGH,
                issue_type=IssueType.STRUCTURAL,
                description="Document contains unclosed code block(s)",
                impact="High - Code extraction will fail",
                recommended_action="Fix markdown syntax by closing all code blocks",
                context={"fence_count": open_fences}
            ))
    
    def _detect_broken_tables(self, blocks: List[Dict]) -> None:
        """Rule 108: Detect malformed tables."""
        table_blocks = [b for b in blocks if b.get('type') == 'table']
        
        for table in table_blocks:
            metadata = table.get('metadata', {})
            headers = metadata.get('headers', [])
            rows = metadata.get('rows', [])
            
            # Check if all rows have same column count as headers
            if headers:
                expected_cols = len(headers)
                for i, row in enumerate(rows):
                    if len(row) != expected_cols:
                        self.warnings.append(ValidationIssue(
                            rule_id=108,
                            rule_name="Broken Table",
                            severity=IssueSeverity.MEDIUM,
                            issue_type=IssueType.STRUCTURAL,
                            description=f"Table row {i} has {len(row)} columns, expected {expected_cols}",
                            block_id=table.get('block_id'),
                            impact="Medium - Table parsing may be incorrect",
                            recommended_action="Fix table markdown alignment",
                            context={"row_index": i, "actual_cols": len(row), "expected_cols": expected_cols}
                        ))
    
    def _detect_broken_links(self, blocks: List[Dict]) -> None:
        """Rule 109: Detect broken or malformed links."""
        link_blocks = [b for b in blocks if b.get('type') == 'link']
        
        for link in link_blocks:
            metadata = link.get('metadata', {})
            url = metadata.get('url', '')
            text = metadata.get('text', '')
            
            # Check for empty URLs
            if not url:
                self.warnings.append(ValidationIssue(
                    rule_id=109,
                    rule_name="Broken Link",
                    severity=IssueSeverity.LOW,
                    issue_type=IssueType.STRUCTURAL,
                    description=f"Link '{text}' has no URL",
                    block_id=link.get('block_id'),
                    impact="Low - Link is non-functional",
                    recommended_action="Add valid URL or remove link",
                    context={"text": text, "url": url}
                ))
            
            # Check for placeholder URLs
            if url in ['#', 'javascript:void(0)', 'TODO', 'FIXME']:
                self.warnings.append(ValidationIssue(
                    rule_id=109,
                    rule_name="Placeholder Link",
                    severity=IssueSeverity.LOW,
                    issue_type=IssueType.STRUCTURAL,
                    description=f"Link '{text}' uses placeholder URL: {url}",
                    block_id=link.get('block_id'),
                    impact="Low - Link is not functional",
                    recommended_action="Replace placeholder with actual URL",
                    context={"text": text, "url": url}
                ))
    
    def _detect_broken_images(self, blocks: List[Dict]) -> None:
        """Rule 110: Detect broken or missing image references."""
        image_blocks = [b for b in blocks if b.get('type') == 'image']
        
        for image in image_blocks:
            metadata = image.get('metadata', {})
            src = metadata.get('src', '')
            alt = metadata.get('alt', '')
            
            # Check for empty source
            if not src:
                self.warnings.append(ValidationIssue(
                    rule_id=110,
                    rule_name="Broken Image Reference",
                    severity=IssueSeverity.LOW,
                    issue_type=IssueType.STRUCTURAL,
                    description=f"Image '{alt}' has no source path",
                    block_id=image.get('block_id'),
                    impact="Low - Image cannot be displayed",
                    recommended_action="Add valid image source path",
                    context={"alt": alt, "src": src}
                ))
    
    def _group_blocks_by_section(self, blocks: List[Dict]) -> Dict[str, List[Dict]]:
        """Group blocks by their parent section."""
        sections = {}
        current_section = "root"
        current_heading_path = []
        
        for block in blocks:
            if block.get('type') == 'heading':
                level = block.get('metadata', {}).get('level', 0)
                title = block.get('content', '')
                
                # Adjust heading path
                while len(current_heading_path) >= level:
                    current_heading_path.pop()
                current_heading_path.append(title)
                current_section = " > ".join(current_heading_path)
                sections[current_section] = [block]
            else:
                if current_section not in sections:
                    current_section = "root"
                sections.setdefault(current_section, []).append(block)
        
        return sections
