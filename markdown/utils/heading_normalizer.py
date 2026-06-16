"""Heading normalization utilities for strict hierarchical Markdown."""
import copy
from typing import Any
from core.contracts.stream import OrderedDocumentBlock
from core.logging import configure_logging, log_event

class HeadingNormalizer:
    """Normalizes heading levels to be strictly sequential (no skipped levels)."""
    
    def __init__(self):
        self.logger = configure_logging()

    def normalize(self, blocks: list[OrderedDocumentBlock]) -> list[OrderedDocumentBlock]:
        """Normalize heading levels in a list of document blocks."""
        log_event(self.logger, 10, "normalize_start", "normalizing headings", block_count=len(blocks))
        
        if not blocks:
            return []

        normalized_blocks = []
        current_max_level = 0

        for block in blocks:
            new_block = copy.deepcopy(block)
            
            if new_block.block_type == "heading":
                original_level = new_block.level or 1
                
                # If the level jumps too high (e.g., H1 to H3), cap it at current_max_level + 1
                if original_level > current_max_level + 1:
                    new_block.level = current_max_level + 1
                    log_event(self.logger, 30, "heading_normalized", "skipped heading level corrected", 
                              original_level=original_level, new_level=new_block.level)
                else:
                    new_block.level = original_level
                
                # Update the max level seen so far
                current_max_level = max(current_max_level, new_block.level)
                
            normalized_blocks.append(new_block)

        log_event(self.logger, 10, "normalize_end", "headings normalized")
        return normalized_blocks

    def validate(self) -> dict[str, Any]:
        return {"status": "ok", "module": "HeadingNormalizer"}

    def debug(self) -> dict[str, Any]:
        return {"module": "HeadingNormalizer"}

    def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "module": "HeadingNormalizer"}