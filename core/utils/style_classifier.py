"""Robust heading classification based on styles and heuristics."""

from __future__ import annotations

import re


class StyleClassifier:
    """Centralized utility for determining if a text block is a heading."""
    
    HEADING_KEYWORDS = {"heading", "title", "subtitle", "toc", "chapter", "header"}
    
    @classmethod
    def classify(cls, style_name: str, text_content: str, outline_level: int | None = None) -> tuple[bool, int]:
        """
        Returns (is_heading: bool, heading_level: int).
        Level 0 means not a heading. Levels 1-6 are valid heading levels.
        """
        style_lower = style_name.lower() if style_name else ""
        text_stripped = text_content.strip() if text_content else ""
        
        # Rule 1: Explicit outline level (common in DOCX)
        if outline_level is not None and outline_level >= 0:
            return True, outline_level + 1
            
        # Rule 2: Keyword match in style name
        for keyword in cls.HEADING_KEYWORDS:
            if keyword in style_lower:
                level_match = re.search(r'\d+', style_lower)
                level = int(level_match.group()) if level_match else 2
                return True, min(level, 6)
                
        # Rule 3: Heuristic fallback for unstyled but obvious headings
        # Short text, no terminal punctuation, no multiple sentences
        if len(text_stripped) < 100 and text_stripped:
            if not re.search(r'[.!?;:]$', text_stripped):
                if text_stripped.count('.') == 0 and text_stripped.count('!') == 0 and text_stripped.count('?') == 0:
                    return True, 2
                    
        return False, 0