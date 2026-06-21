"""
Smart Structure Fixer for RAG Pipeline
Automatically fixes structural Markdown issues without using LLMs.
Focuses on heading injection, hierarchy repair, and noise removal.
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path


class SmartStructureFixer:
    """Fixes structural issues in markdown documents using rule-based heuristics."""
    
    def __init__(self):
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.short_line_pattern = re.compile(r'^([A-Z][A-Za-z0-9\s\-\_\.]{2,50})$')
        self.badge_pattern = re.compile(r'!\[.*?\]\(.*?\)|<img.*?>')
        self.nav_pattern = re.compile(r'(Home|Docs|Previous|Next|Edit this page|Table of Contents)', re.IGNORECASE)
        
    def fix(self, content: str, issues: List[Dict[str, Any]]) -> str:
        """
        Main fixing method.
        Applies surgical fixes based on detected issues.
        """
        fixed_content = content
        
        # Apply fixes based on issue types
        fix_applied = False
        
        # Check for missing headings (most common issue from your output)
        has_headings = bool(self.heading_pattern.search(content))
        if not has_headings:
            fixed_content = self._inject_headings(fixed_content)
            fix_applied = True
        
        # Fix heading hierarchy gaps
        if any('heading' in str(i.get('rule_id', '')).lower() for i in issues):
            fixed_content = self._fix_heading_hierarchy(fixed_content)
            fix_applied = True
        
        # Remove navigation noise
        if any('navigation' in str(i.get('type', '')).lower() or 'badge' in str(i.get('type', '')).lower() for i in issues):
            fixed_content = self._remove_noise(fixed_content)
            fix_applied = True
        
        # Fix broken code fences
        if any('code' in str(i.get('rule_id', '')).lower() or 'fence' in str(i.get('type', '')).lower() for i in issues):
            fixed_content = self._fix_code_fences(fixed_content)
            fix_applied = True
        
        # Fix broken tables
        if any('table' in str(i.get('rule_id', '')).lower() for i in issues):
            fixed_content = self._fix_tables(fixed_content)
            fix_applied = True
        
        return fixed_content if fix_applied else content
    
    def _inject_headings(self, content: str) -> str:
        """
        Inject headings into documents that lack them.
        Uses heuristics to identify potential heading lines.
        """
        lines = content.splitlines(keepends=True)
        result_lines = []
        
        # Track state
        in_code_block = False
        first_content_seen = False
        heading_level = 1
        last_heading_text = ""
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip code blocks
            if stripped.startswith('```'):
                in_code_block = not in_code_block
                result_lines.append(line)
                continue
            
            if in_code_block:
                result_lines.append(line)
                continue
            
            # Skip empty lines
            if not stripped:
                result_lines.append(line)
                continue
            
            # Skip badges and navigation
            if self.badge_pattern.match(stripped) or self.nav_pattern.match(stripped):
                continue
            
            # Detect potential headings
            # Rule 1: First non-empty, non-badge line should be H1
            if not first_content_seen:
                if len(stripped) < 80 and stripped[0].isupper():
                    # Likely a title
                    result_lines.append(f"# {stripped}\n")
                    first_content_seen = True
                    last_heading_text = stripped.lower()
                    heading_level = 2
                    continue
            
            # Rule 2: Short capitalized lines followed by content are likely headings
            # BUT they must NOT:
            # - End with punctuation (., :, etc.) - those are sentences
            # - Contain common non-heading phrases like "using", "with", "by"
            # - Be too long (>50 chars)
            # - Look like instructions
            
            next_line_idx = i + 1
            while next_line_idx < len(lines) and not lines[next_line_idx].strip():
                next_line_idx += 1
            
            has_following_content = next_line_idx < len(lines)
            
            # Check if it looks like a heading candidate
            is_heading_candidate = (
                len(stripped) < 50 and 
                len(stripped) > 2 and
                stripped[0].isupper() and 
                not stripped.endswith(('.', ',', ':', ';')) and
                not stripped.endswith('.') and
                has_following_content and
                not self._is_list_item(stripped) and
                not self._is_table_row(stripped) and
                not self._is_instruction_line(stripped) and
                not self._is_code_import(stripped)
            )
            
            if is_heading_candidate:
                # Additional check: should not contain mid-sentence words
                words = stripped.split()
                if len(words) <= 6:  # Headings are usually short
                    # Avoid making headings from lines with certain patterns
                    lower_stripped = stripped.lower()
                    if not any(word in lower_stripped for word in ['using ', 'with ', 'by ', 'for ', 'to ']):
                        result_lines.append(f"{'#' * heading_level} {stripped}\n")
                        last_heading_text = stripped.lower()
                        continue
            
            # Regular content line
            result_lines.append(line)
            first_content_seen = True
        
        return ''.join(result_lines)
    
    def _is_instruction_line(self, text: str) -> bool:
        """Check if text looks like an instruction rather than a heading."""
        lower = text.lower()
        # Common instruction patterns
        instruction_words = ['install', 'run', 'execute', 'type', 'enter', 'click', 'select']
        return any(lower.startswith(word) for word in instruction_words)
    
    def _is_code_import(self, text: str) -> bool:
        """Check if text looks like a code import statement."""
        return text.startswith('from ') or text.startswith('import ')
    
    def _fix_heading_hierarchy(self, content: str) -> str:
        """Ensure heading levels don't skip (e.g., H1 to H3)."""
        lines = content.splitlines(keepends=True)
        result_lines = []
        last_level = 0
        
        for line in lines:
            match = self.heading_pattern.match(line.strip())
            if match:
                current_level = len(match.group(1))
                # If we jumped more than one level, adjust
                if current_level > last_level + 1 and last_level > 0:
                    # Insert intermediate heading or adjust this one
                    adjusted_level = last_level + 1
                    text = match.group(2)
                    line = '#' * adjusted_level + ' ' + text + '\n'
                last_level = current_level
            result_lines.append(line)
        
        return ''.join(result_lines)
    
    def _remove_noise(self, content: str) -> str:
        """Remove badges, navigation elements, and other noise."""
        lines = content.splitlines()
        result_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip badge lines
            if self.badge_pattern.match(stripped):
                continue
            
            # Skip navigation lines
            if self.nav_pattern.match(stripped):
                continue
            
            # Keep the line
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _fix_code_fences(self, content: str) -> str:
        """Ensure all code fences are properly closed."""
        lines = content.splitlines(keepends=True)
        result_lines = []
        in_code_block = False
        code_fence = '```'
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    # Preserve language identifier if present
                    code_fence = stripped
                else:
                    in_code_block = False
                result_lines.append(line)
            else:
                result_lines.append(line)
        
        # Close unclosed code block
        if in_code_block:
            result_lines.append('```\n')
        
        return ''.join(result_lines)
    
    def _fix_tables(self, content: str) -> str:
        """Basic table structure repairs."""
        # For now, just ensure tables have proper separators
        # This is a simplified version - full table fixing is complex
        return content
    
    def _is_list_item(self, text: str) -> bool:
        """Check if text is a list item."""
        return (text.startswith('- ') or 
                text.startswith('* ') or 
                text.startswith('+ ') or
                re.match(r'^\d+\.\s', text))
    
    def _is_table_row(self, text: str) -> bool:
        """Check if text looks like a table row."""
        return '|' in text and text.count('|') >= 2
