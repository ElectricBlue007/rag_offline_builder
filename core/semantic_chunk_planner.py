"""Phase 3 semantic chunk planning.

This module does not create final chunks. It builds semantic regions and emits
planning decisions for split, merge, anchor, and context injection before any
embedding model or external semantic chunker is allowed to operate.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from core.markdown_normalizer import MarkdownNormalizer
from core.structure_analyzer import StructureAnalyzer
from rag_doc_compiler.models import DocumentNode, StructureAnalysis


@dataclass
class PlannedBlock:
    block_id: str
    node_type: str
    text: str
    line_range: list[int]
    token_estimate: int
    density_score: float
    semantic_tags: list[str]
    preserve_reasons: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)
    cross_references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticRegion:
    region_id: str
    heading_path: list[str]
    heading_level: int
    parent_path: list[str]
    line_range: list[int]
    blocks: list[PlannedBlock]
    token_estimate: int
    density_score: float
    boundary_strength: str
    merge_policy: str
    split_candidates: list[int] = field(default_factory=list)
    merge_candidates: list[str] = field(default_factory=list)
    anchor_blocks: list[str] = field(default_factory=list)
    context_to_inject: list[str] = field(default_factory=list)
    confidence_score: float = 0.0
    retrieval_score: float = 0.0
    self_sufficient: bool = False
    validation_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["blocks"] = [block.to_dict() for block in self.blocks]
        return data


@dataclass
class ChunkPlan:
    source: str
    regions: list[SemanticRegion]
    passes: list[dict[str, Any]]
    global_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "regions": [region.to_dict() for region in self.regions],
            "passes": self.passes,
            "global_warnings": self.global_warnings,
        }


class SemanticChunkPlanner:
    """Plan semantic chunking boundaries without producing final chunks."""

    IDEAL_MIN_TOKENS = 200
    IDEAL_MAX_TOKENS = 500
    HARD_MAX_TOKENS = 700
    TINY_SECTION_TOKENS = 80
    MIN_DENSITY = 0.25

    _DEFINITION = re.compile(r"\b([A-Z][A-Za-z0-9_]*(?:Chunker|Embeddings|Pipeline|API)?)\s+(?:is|means|refers to|are)\b")
    _REQUIREMENT = re.compile(r"\b(requirements?|requires?|must|required|python\s*\d|node\s*\d)\b", re.IGNORECASE)
    _PROCEDURE = re.compile(r"\b(step\s*\d+|first|then|next|finally|install|run|execute|configure)\b", re.IGNORECASE)
    _API = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\s+/|\bapi\b|endpoint", re.IGNORECASE)
    _CONFIG = re.compile(r"\b(config|configuration|\.ya?ml|\.json|toml|env|settings?)\b", re.IGNORECASE)
    _EXAMPLE = re.compile(r"\b(example|for example|sample)\b", re.IGNORECASE)
    _FAQ = re.compile(r"^\s*(q:|question:|#+\s+q:)|\?\s*$", re.IGNORECASE | re.MULTILINE)
    _CROSS_REFERENCE = re.compile(r"\b(?:see|refer to|as described in|more in)\s+([A-Z][A-Za-z0-9 &/_-]+)", re.IGNORECASE)
    _ALIAS = re.compile(r"\b([A-Z][A-Za-z0-9_]+)\s+\(([^)]+)\)")
    _ENTITY = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:Chunker|Embeddings|Pipeline|API|Config|Model)?\b")

    def __init__(self) -> None:
        self.normalizer = MarkdownNormalizer()
        self.analyzer = StructureAnalyzer()

    def plan(self, cleaned_markdown: str, source: str = "cleaned_markdown") -> ChunkPlan:
        """Build a multi-pass semantic chunking plan from cleaned Markdown."""
        normalized = self.normalizer.normalize(cleaned_markdown)
        analysis = self.analyzer.analyze(normalized)

        regions = self._build_semantic_regions(analysis)
        passes = [{"pass": 1, "name": "structural_regions", "region_count": len(regions)}]

        self._apply_anchor_rules(regions)
        passes.append({"pass": 2, "name": "anchor_detection", "anchored_regions": self._count_anchored(regions)})

        self._detect_semantic_boundaries(regions)
        passes.append({"pass": 3, "name": "semantic_boundaries", "split_candidates": sum(len(r.split_candidates) for r in regions)})

        self._optimize_merge_and_context(regions)
        passes.append({"pass": 4, "name": "optimization", "merge_candidates": sum(len(r.merge_candidates) for r in regions)})

        self._validate_regions(regions)
        passes.append({"pass": 5, "name": "validation", "warnings": sum(len(r.validation_flags) for r in regions)})

        return ChunkPlan(source=source, regions=regions, passes=passes)

    def _build_semantic_regions(self, analysis: StructureAnalysis) -> list[SemanticRegion]:
        regions = []
        sections = [
            node for node in analysis.flat_map.values()
            if node.node_type == "section" and any(child.node_type != "section" for child in node.children)
        ]

        for index, section in enumerate(sorted(sections, key=lambda node: node.line_range[0]), start=1):
            blocks = [
                self._planned_block(child)
                for child in section.children
                if child.node_type != "section"
            ]
            if not blocks:
                continue

            token_estimate = sum(block.token_estimate for block in blocks)
            density = self._weighted_density(blocks)
            heading_level = section.level
            boundary_strength = self._boundary_strength(heading_level)
            merge_policy = self._merge_policy(heading_level, token_estimate)
            context_to_inject = list(section.heading_path)

            regions.append(
                SemanticRegion(
                    region_id=f"region_{index}",
                    heading_path=list(section.heading_path),
                    heading_level=heading_level,
                    parent_path=list(section.heading_path[:-1]),
                    line_range=list(section.line_range),
                    blocks=blocks,
                    token_estimate=token_estimate,
                    density_score=density,
                    boundary_strength=boundary_strength,
                    merge_policy=merge_policy,
                    context_to_inject=context_to_inject,
                    self_sufficient=bool(section.heading_path),
                )
            )

        return regions

    def _planned_block(self, block: DocumentNode) -> PlannedBlock:
        text = block.content
        return PlannedBlock(
            block_id=block.node_id,
            node_type=block.node_type,
            text=text,
            line_range=list(block.line_range),
            token_estimate=self._estimate_tokens(text),
            density_score=self._density(text),
            semantic_tags=self._semantic_tags(block, text),
            preserve_reasons=self._preserve_reasons(block, text),
            entities=self._entities(text),
            aliases=self._aliases(text),
            cross_references=self._cross_references(text),
        )

    def _apply_anchor_rules(self, regions: list[SemanticRegion]) -> None:
        for region in regions:
            for index, block in enumerate(region.blocks):
                if block.preserve_reasons:
                    region.anchor_blocks.append(block.block_id)

                    if index > 0:
                        previous = region.blocks[index - 1]
                        if previous.node_type == "paragraph":
                            previous.preserve_reasons.append("anchor_explanation_before")

                    if index + 1 < len(region.blocks):
                        following = region.blocks[index + 1]
                        if following.node_type == "paragraph":
                            following.preserve_reasons.append("anchor_explanation_after")

    def _detect_semantic_boundaries(self, regions: list[SemanticRegion]) -> None:
        for region in regions:
            if len(region.blocks) < 3:
                continue

            similarities = [
                self._jaccard(region.blocks[index].text, region.blocks[index + 1].text)
                for index in range(len(region.blocks) - 1)
            ]

            for index, score in enumerate(similarities):
                previous_score = similarities[index - 1] if index > 0 else score
                next_score = similarities[index + 1] if index + 1 < len(similarities) else score
                valley_drop = previous_score - score
                is_local_minimum = score <= previous_score and score <= next_score

                if is_local_minimum and (valley_drop >= 0.25 or score < 0.18):
                    split_after_line = region.blocks[index].line_range[1]
                    if self._split_allowed(region, index):
                        region.split_candidates.append(split_after_line)

    def _optimize_merge_and_context(self, regions: list[SemanticRegion]) -> None:
        for index, region in enumerate(regions):
            if region.token_estimate < self.TINY_SECTION_TOKENS:
                region.validation_flags.append("tiny_section_expansion_needed")

            if region.heading_path:
                region.context_to_inject = list(region.heading_path)

            if self._needs_parent_expansion(region):
                region.context_to_inject = list(region.heading_path)
                region.validation_flags.append("parent_context_required")

            if index == 0:
                continue

            previous = regions[index - 1]
            if self._can_merge_regions(previous, region):
                previous.merge_candidates.append(region.region_id)
                region.merge_candidates.append(previous.region_id)

    def _validate_regions(self, regions: list[SemanticRegion]) -> None:
        for region in regions:
            region.self_sufficient = self._self_sufficient(region)
            region.confidence_score = self._confidence(region)
            region.retrieval_score = self._retrieval_score(region)

            if not region.self_sufficient:
                region.validation_flags.append("not_self_sufficient")
            if not region.context_to_inject:
                region.validation_flags.append("contextless_region")
            if region.token_estimate < self.TINY_SECTION_TOKENS and not region.merge_candidates:
                region.validation_flags.append("orphan_tiny_region")
            if region.density_score < self.MIN_DENSITY or self._low_semantic_mass(region):
                region.validation_flags.append("low_density_region")
            if region.confidence_score <= 0.45:
                region.validation_flags.append("low_confidence_rechunk_required")

    def _split_allowed(self, region: SemanticRegion, block_index: int) -> bool:
        left = region.blocks[block_index]
        right = region.blocks[block_index + 1]
        if left.preserve_reasons or right.preserve_reasons:
            return False
        if region.heading_level == 1:
            return True
        if left.token_estimate + right.token_estimate < self.TINY_SECTION_TOKENS:
            return False
        return True

    def _can_merge_regions(self, left: SemanticRegion, right: SemanticRegion) -> bool:
        if left.heading_level == 1 or right.heading_level == 1:
            return False
        if left.parent_path != right.parent_path:
            return False
        if left.heading_level == 2 and right.heading_level == 2:
            return left.token_estimate < self.TINY_SECTION_TOKENS or right.token_estimate < self.TINY_SECTION_TOKENS
        if left.heading_level >= 3 or right.heading_level >= 3:
            return left.token_estimate + right.token_estimate <= self.IDEAL_MAX_TOKENS
        return False

    def _needs_parent_expansion(self, region: SemanticRegion) -> bool:
        return (
            region.token_estimate < self.TINY_SECTION_TOKENS
            or region.heading_level >= 3
            or not region.self_sufficient
        )

    def _boundary_strength(self, heading_level: int) -> str:
        if heading_level == 1:
            return "hard"
        if heading_level == 2:
            return "soft"
        return "merge_candidate"

    def _merge_policy(self, heading_level: int, token_estimate: int) -> str:
        if heading_level == 1:
            return "never_merge"
        if heading_level == 2:
            return "usually_separate" if token_estimate >= self.TINY_SECTION_TOKENS else "merge_if_tiny"
        return "merge_candidate"

    def _preserve_reasons(self, block: DocumentNode, text: str) -> list[str]:
        reasons = []
        lower = text.lower()
        if block.node_type == "code_block":
            reasons.append("code_anchor")
        if block.node_type == "table":
            reasons.append("table_anchor")
        if block.node_type == "list":
            reasons.append("list_anchor")
        if block.node_type == "quote":
            reasons.append("quote_anchor")
        if self._DEFINITION.search(text):
            reasons.append("definition_block")
        if self._REQUIREMENT.search(text):
            reasons.append("requirement_block")
        if self._PROCEDURE.search(text):
            reasons.append("procedure_block")
        if self._API.search(text):
            reasons.append("api_block")
        if self._CONFIG.search(text):
            reasons.append("config_block")
        if self._EXAMPLE.search(text):
            reasons.append("example_block")
        if self._FAQ.search(text) or lower.startswith(("q:", "question:")):
            reasons.append("faq_block")
        return reasons

    def _semantic_tags(self, block: DocumentNode, text: str) -> list[str]:
        tags = [block.node_type]
        tags.extend(f"h{index + 1}:{heading.lower().replace(' ', '-')}" for index, heading in enumerate(block.heading_path))
        tags.extend(self._preserve_reasons(block, text))
        return sorted(set(tags))

    def _confidence(self, region: SemanticRegion) -> float:
        score = 0.35
        score += min(0.25, region.density_score * 0.25)
        score += 0.15 if region.context_to_inject else 0.0
        score += 0.15 if region.self_sufficient else 0.0
        score += 0.1 if self.IDEAL_MIN_TOKENS <= region.token_estimate <= self.IDEAL_MAX_TOKENS else 0.0
        score -= 0.15 if region.token_estimate > self.HARD_MAX_TOKENS else 0.0
        score -= 0.25 if region.token_estimate < self.TINY_SECTION_TOKENS and not region.merge_candidates else 0.0
        return round(max(0.0, min(1.0, score)), 3)

    def _retrieval_score(self, region: SemanticRegion) -> float:
        text = "\n".join(block.text for block in region.blocks)
        score = 0.25
        if region.context_to_inject:
            score += 0.2
        if any(block.preserve_reasons for block in region.blocks):
            score += 0.2
        if re.search(r"\b(is|are|how|why|when|where|step|install|configure|api|example)\b", text, re.IGNORECASE):
            score += 0.2
        if region.token_estimate >= self.TINY_SECTION_TOKENS:
            score += 0.1
        if any(block.cross_references for block in region.blocks):
            score += 0.05
        return round(max(0.0, min(1.0, score)), 3)

    def _self_sufficient(self, region: SemanticRegion) -> bool:
        if not region.heading_path:
            return False
        if region.token_estimate >= self.TINY_SECTION_TOKENS:
            return True
        return bool(region.merge_candidates or any(block.preserve_reasons for block in region.blocks))

    def _low_semantic_mass(self, region: SemanticRegion) -> bool:
        if region.token_estimate >= self.TINY_SECTION_TOKENS:
            return False
        if any(block.preserve_reasons for block in region.blocks):
            return False
        return True

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        words = len(re.findall(r"\S+", text))
        chars = max(1, math.ceil(len(text) / 4))
        return max(words, chars)

    @staticmethod
    def _density(text: str) -> float:
        tokens = re.findall(r"[A-Za-z0-9_]+", text)
        if not tokens:
            return 0.0
        unique_ratio = len(set(token.lower() for token in tokens)) / len(tokens)
        symbol_ratio = min(0.2, len(re.findall(r"[{}()[\]|`:/]", text)) / max(len(text), 1))
        return round(min(1.0, unique_ratio * 0.8 + symbol_ratio), 3)

    @staticmethod
    def _weighted_density(blocks: list[PlannedBlock]) -> float:
        total_tokens = sum(block.token_estimate for block in blocks)
        if total_tokens == 0:
            return 0.0
        score = sum(block.density_score * block.token_estimate for block in blocks) / total_tokens
        return round(score, 3)

    @staticmethod
    def _jaccard(left: str, right: str) -> float:
        left_terms = set(re.findall(r"[A-Za-z0-9_]+", left.lower()))
        right_terms = set(re.findall(r"[A-Za-z0-9_]+", right.lower()))
        if not left_terms or not right_terms:
            return 0.0
        return round(len(left_terms & right_terms) / len(left_terms | right_terms), 3)

    def _entities(self, text: str) -> list[str]:
        return sorted(set(match.group(0) for match in self._ENTITY.finditer(text)))

    def _aliases(self, text: str) -> dict[str, str]:
        return {match.group(1): match.group(2) for match in self._ALIAS.finditer(text)}

    def _cross_references(self, text: str) -> list[str]:
        return sorted(set(match.group(1).strip() for match in self._CROSS_REFERENCE.finditer(text)))

    def _count_anchored(self, regions: list[SemanticRegion]) -> int:
        return sum(1 for region in regions if region.anchor_blocks)
