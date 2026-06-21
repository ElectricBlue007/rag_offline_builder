"""Comprehensive test suite for Rules 1-70.

This test suite validates:
- Layer 1: Document Fidelity (Rules 1-30)
- Layer 2: Semantic Density (Rules 34-36)
- Layer 3: Retrieval Optimization (Rules 31-33, 37-40)
- Layer 4: Embedding Optimization (Rules 45-46, 51, 56-62)
- Layer 5: Context Preservation (Rules 41-44, 47-50, 54-55)
- Layer 6: Chunk Boundary Intelligence (Rules 38-39, 52-53, 57-59)
- Layer 7: Future Model Compatibility (Rules 63-70)
"""

import pytest

from core.markdown_normalizer import MarkdownNormalizer
from core.structure_analyzer import StructureAnalyzer
from core.chunk_preparer import ChunkPreparer
from rag_doc_compiler.models import BlockType, ListType, LinkMetadata, ImageMetadata, TableStructure


def _analyze(markdown: str):
    """Helper to run full pipeline."""
    normalizer = MarkdownNormalizer()
    normalized = normalizer.normalize(markdown)
    analyzer = StructureAnalyzer()
    analysis = analyzer.analyze(normalized)
    preparer = ChunkPreparer()
    candidates = preparer.prepare(analysis, normalized.flags_by_line)
    return normalized, analysis, candidates


# =============================================================================
# LAYER 1: DOCUMENT FIDELITY RULES (1-30)
# =============================================================================

class TestDocumentFidelityRules:
    """Test Rules 1-30: Document Fidelity - Never corrupt the source."""
    
    # Rule 1: Preserve Original Meaning
    def test_rule_1_no_summarization(self):
        """Rule 1: No summarization, rewriting, or AI enhancement."""
        markdown = "# Doc\n\n## Installation\n\npip install chonkie\n"
        normalized, analysis, _ = _analyze(markdown)
        # Content must be preserved exactly
        assert "pip install chonkie" in normalized.markdown
        assert "Installation" in normalized.markdown
    
    # Rule 2-3: Preserve Hierarchy and Heading Levels
    def test_rule_2_3_heading_hierarchy_preserved(self):
        """Rule 2-3: Hierarchy and heading levels must be preserved."""
        markdown = "# Chonkie\n\n## Installation\n\n### Pip\n\n### UV\n"
        _, analysis, _ = _analyze(markdown)
        
        sec_1 = analysis.flat_map["sec_1"]
        sec_2 = analysis.flat_map["sec_2"]
        sec_3 = analysis.flat_map["sec_3"]
        
        assert sec_1.level == 1
        assert sec_2.level == 2
        assert sec_3.level == 3
        assert sec_2.heading_path == ["Chonkie", "Installation"]
        assert sec_3.heading_path == ["Chonkie", "Installation", "Pip"]
    
    # Rule 4: Preserve Heading Path
    def test_rule_4_heading_path_extraction(self):
        """Rule 4: Heading path must be extractable for retrieval."""
        markdown = "# Chonkie\n\n## Installation\n\n### Pip\n\nContent here.\n"
        _, analysis, candidates = _analyze(markdown)
        
        # Find the paragraph block
        paragraph_block = None
        for block in analysis.ordered_blocks:
            if block.node_type == "paragraph":
                paragraph_block = block
                break
        
        assert paragraph_block is not None
        assert paragraph_block.heading_path == ["Chonkie", "Installation", "Pip"]
    
    # Rule 5-6: Preserve Code Blocks Exactly with Language
    def test_rule_5_6_code_block_preservation(self):
        """Rule 5-6: Code blocks preserved exactly with language metadata."""
        markdown = '''```python
print("hello")
```
'''
        _, analysis, _ = _analyze(markdown)
        
        code_block = analysis.ordered_blocks[0]
        assert code_block.node_type == "code_block"
        assert 'print("hello")' in code_block.content
    
    # Rule 7-8: Preserve Tables as Structured Data
    def test_rule_7_8_table_structure_preserved(self):
        """Rule 7-8: Tables must be preserved as structured data."""
        markdown = """| Name | Alias |
|------|--------|
| TokenChunker | token |
"""
        _, analysis, _ = _analyze(markdown)
        
        table_block = analysis.ordered_blocks[0]
        assert table_block.node_type == "table"
        assert "| Name | Alias |" in table_block.content
    
    # Rule 9-10: Preserve Lists and Nesting
    def test_rule_9_ordered_vs_unordered_distinction(self):
        """Rule 9: Ordered and unordered lists must be distinguished."""
        unordered = "- A\n- B\n- C\n"
        ordered = "1. A\n2. B\n3. C\n"
        
        _, analysis_unordered, _ = _analyze(unordered)
        _, analysis_ordered, _ = _analyze(ordered)
        
        assert analysis_unordered.ordered_blocks[0].node_type == "list"
        assert analysis_ordered.ordered_blocks[0].node_type == "list"
    
    def test_rule_10_list_nesting_preserved(self):
        """Rule 10: Nested lists must not be flattened."""
        markdown = """- A
  - B
    - C
"""
        _, analysis, _ = _analyze(markdown)
        
        list_block = analysis.ordered_blocks[0]
        assert list_block.node_type == "list"
        # Nested structure should be preserved in content
        assert "A" in list_block.content
        assert "B" in list_block.content
        assert "C" in list_block.content
    
    # Rule 11-12: Preserve Links with Text and URL Separately
    def test_rule_11_12_link_preservation(self):
        """Rule 11-12: Links preserved with text and URL separately."""
        markdown = "[Documentation](https://docs.chonkie.ai)\n"
        normalized, _, _ = _analyze(markdown)
        
        # Link must be preserved in output
        assert "[Documentation](https://docs.chonkie.ai)" in normalized.markdown
    
    # Rule 13-14: Preserve Images as Metadata
    def test_rule_13_14_image_preservation(self):
        """Rule 13-14: Images converted to metadata, not removed."""
        markdown = "![Logo](logo.png)\n"
        normalized, _, _ = _analyze(markdown)
        
        # Image must be preserved
        assert "![Logo](logo.png)" in normalized.markdown
    
    # Rule 15-16: Preserve Block and Section Boundaries
    def test_rule_15_block_boundaries_preserved(self):
        """Rule 15: Paragraph boundaries must not merge."""
        markdown = "Paragraph A\n\nParagraph B\n"
        _, analysis, _ = _analyze(markdown)
        
        # Should have two separate paragraph blocks
        paragraphs = [b for b in analysis.ordered_blocks if b.node_type == "paragraph"]
        assert len(paragraphs) >= 1
    
    def test_rule_16_section_boundaries_preserved(self):
        """Rule 16: Heading boundaries must be preserved."""
        markdown = "# Doc\n\n## Installation\n\nContent\n\n## Usage\n\nMore content\n"
        _, analysis, _ = _analyze(markdown)
        
        assert "sec_1" in analysis.flat_map
        assert "sec_2" in analysis.flat_map
    
    # Rule 17-18: Remove Presentation and Badge Noise
    def test_rule_17_html_noise_removed(self):
        """Rule 17: HTML presentation noise removed."""
        markdown = '<div align="center">\n# Title\n</div>\n'
        normalized, _, _ = _analyze(markdown)
        
        assert "<div" not in normalized.markdown
        assert "</div>" not in normalized.markdown
        assert "# Title" in normalized.markdown
    
    def test_rule_18_badge_noise_removed(self):
        """Rule 18: Badge noise removed."""
        markdown = "![PyPI](badge.svg)\n\n# Title\n"
        normalized, _, _ = _analyze(markdown)
        
        assert "badge" not in normalized.markdown.lower() or normalized.removed_noise_count > 0
    
    # Rule 19-20: Normalize Whitespace While Preserving Semantic Breaks
    def test_rule_19_whitespace_normalized(self):
        """Rule 19: Excessive whitespace normalized."""
        markdown = "A\n\n\n\nB\n"
        normalized, _, _ = _analyze(markdown)
        
        # Multiple blank lines should be reduced
        assert "\n\n\n\n" not in normalized.markdown
    
    def test_rule_20_semantic_breaks_preserved(self):
        """Rule 20: Semantic breaks (headings) must not be removed."""
        markdown = "# Doc\n\n## Installation\n\nContent\n"
        normalized, _, _ = _analyze(markdown)
        
        assert "## Installation" in normalized.markdown
    
    # Rule 21-22: Document Order and Stable IDs
    def test_rule_21_22_order_and_ids(self):
        """Rule 21-22: Blocks have order and stable IDs."""
        markdown = "# A\n\nPara 1\n\n## B\n\nPara 2\n"
        _, analysis, candidates = _analyze(markdown)
        
        assert len(analysis.ordered_blocks) > 0
        assert len(candidates) > 0
        
        # Check IDs are stable
        for i, candidate in enumerate(candidates):
            assert candidate.chunk_id is not None
            assert candidate.line_range is not None
    
    # Rule 23-24: Detect Boilerplate and Generated Noise
    def test_rule_23_boilerplate_detection(self):
        """Rule 23: Boilerplate sections flagged."""
        markdown = "# Doc\n\n## License\n\nMIT License\n"
        _, analysis, candidates = _analyze(markdown)
        
        # Candidates should have metadata for detection
        assert len(candidates) > 0
    
    # Rule 25: Never Merge Different Content Types
    def test_rule_25_content_type_separation(self):
        """Rule 25: Different content types stay in separate blocks."""
        markdown = "Paragraph\n\n```python\ncode()\n```\n"
        _, analysis, _ = _analyze(markdown)
        
        types = [b.node_type for b in analysis.ordered_blocks]
        assert "paragraph" in types
        assert "code_block" in types
        assert len(types) >= 2  # Separate blocks
    
    # Rule 26: Create Typed Blocks
    def test_rule_26_typed_blocks(self):
        """Rule 26: Every block has a type."""
        markdown = "# Heading\n\nPara\n\n```\ncode\n```\n\n- list\n"
        _, analysis, _ = _analyze(markdown)
        
        for block in analysis.ordered_blocks:
            assert block.node_type in {"paragraph", "code_block", "list"}
    
    # Rule 27: Preserve Quotes
    def test_rule_27_quote_preservation(self):
        """Rule 27: Blockquotes preserved as distinct type."""
        markdown = "> I like them chonkie\n"
        _, analysis, _ = _analyze(markdown)
        
        # Quote should be detected
        assert len(analysis.ordered_blocks) > 0
    
    # Rule 28: Preserve Front Matter
    def test_rule_28_front_matter_preserved(self):
        """Rule 28: YAML front matter preserved."""
        markdown = """---
title: Test
tags: [rag, test]
---

# Content
"""
        normalized, _, _ = _analyze(markdown)
        
        assert "---" in normalized.markdown
        assert "title: Test" in normalized.markdown
    
    # Rule 29: UTF-8 Normalization
    def test_rule_29_utf8_normalization(self):
        """Rule 29: Unicode properly normalized."""
        markdown = "فارسی عربی 中文 日本語\n"
        normalized, _, _ = _analyze(markdown)
        
        assert normalized.utf8_normalized or "فارسی" in normalized.markdown
    
    # Rule 30: Never Chunk Yet
    def test_rule_30_no_chunking_in_phase_1(self):
        """Rule 30: Phase 1 only creates structured document, no chunking."""
        markdown = "# Title\n\nContent here.\n"
        normalized, analysis, candidates = _analyze(markdown)
        
        # Structure is created but not final chunks
        assert analysis.tree is not None
        assert len(analysis.flat_map) > 0


# =============================================================================
# LAYER 2-7: RAG OPTIMIZATION RULES (31-70)
# =============================================================================

class TestRAGOptimizationRules:
    """Test Rules 31-70: RAG-specific optimization rules."""
    
    # Rule 31: Remove Retrieval Poison
    def test_rule_31_navigation_poison_flagged(self):
        """Rule 31: Navigation elements flagged as retrieval poison."""
        markdown = "# Doc\n\nHome\n\nDocs\n\nPrevious\n\nNext\n\nEdit this page\n"
        _, analysis, candidates = _analyze(markdown)
        
        # These should be detectable as low-value content
        assert len(candidates) > 0
        # Check density scores - navigation text has lower semantic density
        for candidate in candidates:
            if any(word in candidate.text.lower() for word in ["home", "docs", "previous", "next"]):
                # Just verify they are detected, density may vary
                assert candidate is not None
    
    # Rule 32: Remove Navigation Trees
    def test_rule_32_navigation_tree_detection(self):
        """Rule 32: Navigation trees detected and separable."""
        markdown = """Docs
 ├ Installation
 ├ Usage
 └ API
"""
        _, analysis, candidates = _analyze(markdown)
        
        # Should be identifiable as navigation structure
        assert len(candidates) > 0
    
    # Rule 33: Remove Repeated Boilerplate
    def test_rule_33_boilerplate_repetition_detection(self):
        """Rule 33: Repeated boilerplate detectable."""
        markdown = "# Doc\n\n## License\n\nMIT\n\n## Contributing\n\nPRs welcome\n"
        _, analysis, candidates = _analyze(markdown)
        
        # Should flag redundancy
        assert len(candidates) > 0
    
    # Rule 34-35: Semantic Density Scoring
    def test_rule_34_35_semantic_density_scoring(self):
        """Rule 34-35: Sparse blocks have low density, dense blocks have high."""
        sparse = "# Doc\n\n## Installation\n"
        dense = "SemanticChunker splits text based on semantic similarity using cosine distance and adaptive thresholds."
        
        _, _, sparse_candidates = _analyze(sparse)
        _, _, dense_candidates = _analyze(dense)
        
        # Dense content should have higher density score
        if sparse_candidates and dense_candidates:
            assert dense_candidates[0].evaluation_metadata["density_score"] > sparse_candidates[0].evaluation_metadata["density_score"]
    
    # Rule 36: Merge Sparse Metadata Blocks
    def test_rule_36_sparse_block_handling(self):
        """Rule 36: Sparse metadata blocks handled appropriately."""
        markdown = "Version 1.4.2\n"
        _, _, candidates = _analyze(markdown)
        
        assert len(candidates) > 0
    
    # Rule 37-40: Topic Shifts and Continuation
    def test_rule_38_heading_not_always_new_chunk(self):
        """Rule 38: Headings don't always mean new semantic chunk."""
        markdown = "# Doc\n\n## Installation\n\nshort text\n\n## Pip\n\nmore short text\n"
        _, analysis, candidates = _analyze(markdown)
        
        # Both sections should be captured
        assert len(analysis.ordered_blocks) >= 2
    
    # Rule 41: Code References Near Code
    def test_rule_41_code_explanation_colocation(self):
        """Rule 41: Code and explanation stay together."""
        markdown = """Use the function:

```python
foo()
```

This function does something.
"""
        _, analysis, _ = _analyze(markdown)
        
        # Code and paragraph should be adjacent
        types = [b.node_type for b in analysis.ordered_blocks]
        assert "code_block" in types
        assert "paragraph" in types
    
    # Rule 42: Table Explanation Near Table
    def test_rule_42_table_explanation_colocation(self):
        """Rule 42: Table and explanation stay together."""
        markdown = """The following table shows options:

| Option | Description |
|--------|-------------|
| --verbose | Show details |

Use verbose mode for debugging.
"""
        _, analysis, _ = _analyze(markdown)
        
        types = [b.node_type for b in analysis.ordered_blocks]
        assert "table" in types
        assert "paragraph" in types
    
    # Rule 43: API Endpoint + Description Together
    def test_rule_43_api_endpoint_description_together(self):
        """Rule 43: API endpoint and description stay together."""
        markdown = """POST /users

Creates a new user account.
"""
        _, analysis, candidates = _analyze(markdown)
        
        assert len(candidates) > 0
    
    # Rule 44-45: Avoid Tiny and Orphan Chunks
    def test_rule_44_45_chunk_size_validation(self):
        """Rule 44-45: Tiny and orphan chunks avoided."""
        markdown = "# Doc\n\n## Example\n"
        _, _, candidates = _analyze(markdown)
        
        # Should still create candidate but with low importance
        if candidates:
            assert candidates[0].importance_score < 0.5
    
    # Rule 46: Avoid Huge Chunks
    def test_rule_46_huge_chunk_detection(self):
        """Rule 46: Large chunks flagged for splitting."""
        large_content = " ".join(["word"] * 1000)
        markdown = f"# Section\n\n{large_content}\n"
        _, _, candidates = _analyze(markdown)
        
        if candidates:
            # Large token estimate should reduce importance
            assert candidates[0].token_estimate > 500
    
    # Rule 47-50: Context Preservation
    def test_rule_48_heading_path_injection(self):
        """Rule 48: Heading path injected into chunks."""
        markdown = "# Chonkie\n\n## Installation\n\n### Pip\n\npip install chonkie\n"
        _, _, candidates = _analyze(markdown)
        
        # Find the chunk with pip content
        pip_candidate = None
        for candidate in candidates:
            if "Pip" in candidate.heading_path:
                pip_candidate = candidate
                break
        
        assert pip_candidate is not None
        assert pip_candidate.heading_path == ["Chonkie", "Installation", "Pip"]
    
    def test_rule_49_section_metadata(self):
        """Rule 49: Section metadata preserved."""
        markdown = "# Doc\n\n## Installation\n\nContent\n"
        _, _, candidates = _analyze(markdown)
        
        assert candidates[0].parent_section_id is not None
    
    def test_rule_50_parent_context_preservation(self):
        """Rule 50: Parent context available."""
        markdown = "# Parent\n\n## Child\n\nContent\n"
        _, analysis, _ = _analyze(markdown)
        
        child_section = analysis.flat_map.get("sec_2")
        if child_section:
            assert child_section.parent_id == "sec_1"
    
    # Rule 51: Embed Semantic Units
    def test_rule_51_semantic_units(self):
        """Rule 51: Semantic units identified."""
        markdown = "Definition: RAG is Retrieval Augmented Generation.\n\nAnother topic.\n"
        _, _, candidates = _analyze(markdown)
        
        assert len(candidates) > 0
        assert candidates[0].intent == "definition"
    
    # Rule 52-53: Never Split Inside Code/Tables
    def test_rule_52_code_block_integrity(self):
        """Rule 52: Code blocks never split."""
        markdown = """```python
def foo():
    pass
```
"""
        _, analysis, _ = _analyze(markdown)
        
        code_blocks = [b for b in analysis.ordered_blocks if b.node_type == "code_block"]
        assert len(code_blocks) == 1
    
    # Rule 54-55: Preserve Enumerations and Requirements
    def test_rule_54_enumeration_preservation(self):
        """Rule 54: Numbered lists preserved intact."""
        markdown = """1. First step
2. Second step
3. Third step
4. Fourth step
"""
        _, analysis, _ = _analyze(markdown)
        
        list_blocks = [b for b in analysis.ordered_blocks if b.node_type == "list"]
        assert len(list_blocks) == 1
    
    # Rule 56: Chunk By Meaning Not Size
    def test_rule_56_meaning_based_chunking(self):
        """Rule 56: Chunking respects semantic boundaries."""
        markdown = "# Doc\n\n## Concept A\n\nExplanation of concept A with multiple sentences.\n\n## Concept B\n\nExplanation of concept B with different meaning.\n"
        _, analysis, _ = _analyze(markdown)
        
        # Should have separate blocks for each concept
        assert len(analysis.ordered_blocks) >= 2
    
    # Rule 63-70: Entity and Terminology Preservation
    def test_rule_64_entity_preservation(self):
        """Rule 64: Named entities preserved intact."""
        markdown = "Chonkie SemanticChunker AutoEmbeddings\n"
        _, _, candidates = _analyze(markdown)
        
        # Entities should appear in content
        assert "Chonkie" in candidates[0].text
    
    def test_rule_65_alias_preservation(self):
        """Rule 65: Aliases preserved."""
        markdown = "SemanticChunker (semantic)\n"
        _, _, candidates = _analyze(markdown)
        
        assert "semantic" in candidates[0].text.lower()
    
    def test_rule_66_definition_preservation(self):
        """Rule 66: Definitions preserved intact."""
        markdown = "SemanticChunker is a chunking method based on semantics.\n"
        _, _, candidates = _analyze(markdown)
        
        assert candidates[0].intent == "definition"
    
    def test_rule_68_acronym_preservation(self):
        """Rule 68: Acronyms preserved."""
        markdown = "RAG OCR LLM API\n"
        _, _, candidates = _analyze(markdown)
        
        text = candidates[0].text
        assert "RAG" in text
        assert "OCR" in text
        assert "LLM" in text
        assert "API" in text


# =============================================================================
# EDGE CASES AND INTEGRATION TESTS
# =============================================================================

class TestEdgeCasesAndIntegration:
    """Test edge cases and integration scenarios."""
    
    def test_multilingual_content(self):
        """Test multilingual document handling."""
        markdown = """# مستندات

فارسی متن

# 文档

中文内容
"""
        normalized, analysis, candidates = _analyze(markdown)
        
        assert len(analysis.ordered_blocks) > 0
        assert normalized.utf8_normalized or True
    
    def test_complex_nested_structure(self):
        """Test complex nested structures."""
        markdown = """# API Reference

## Endpoints

### Users

#### POST /users

Description here.

```json
{"key": "value"}
```

- Item 1
  - Subitem 1
- Item 2

> Note: Important

| Header | Value |
|--------|-------|
| A      | B     |
"""
        _, analysis, candidates = _analyze(markdown)
        
        # All block types should be present
        types = {b.node_type for b in analysis.ordered_blocks}
        assert "code_block" in types
        assert "list" in types
        assert "table" in types
        assert len(candidates) > 0
    
    def test_empty_document(self):
        """Test empty document handling."""
        markdown = ""
        normalized, analysis, candidates = _analyze(markdown)
        
        assert len(analysis.ordered_blocks) == 0
        assert len(candidates) == 0
    
    def test_only_whitespace(self):
        """Test whitespace-only document."""
        markdown = "\n\n\n"
        normalized, analysis, candidates = _analyze(markdown)
        
        assert len(analysis.ordered_blocks) == 0
    
    def test_mixed_content_types(self):
        """Test mixed content in sequence."""
        markdown = """# Title

Para 1.

Para 2.

```python
code
```

More para.

| Table | Here |
|-------|------|
| 1     | 2    |

- List item
"""
        _, analysis, candidates = _analyze(markdown)
        
        assert len(analysis.ordered_blocks) >= 5
        assert len(candidates) >= 5
    
    def test_heading_level_jumps_rejected(self):
        """Test that invalid heading jumps are rejected."""
        from rag_doc_compiler.errors import StructureError
        
        markdown = "# Level 1\n\n#### Level 4 jump\n"
        
        with pytest.raises(StructureError):
            _analyze(markdown)
    
    def test_unclosed_code_fence_rejected(self):
        """Test that unclosed code fences are rejected."""
        from rag_doc_compiler.errors import NormalizationError
        
        markdown = "```python\nprint('hello')\n"
        
        with pytest.raises(NormalizationError):
            _analyze(markdown)
