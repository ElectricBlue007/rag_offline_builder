"""Rule-specific LLM repair prompt builder.

This agent turns validation and chunk-review rule ids into targeted operator
instructions. It does not call an LLM and it does not edit files; healing agents
use it to build precise prompts for the markdown or chunk text they replace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuleRepairStrategy:
    rule_id: int
    name: str
    repair_target: str
    operator_goal: str
    llm_instruction: str
    replacement_policy: str


class RuleRepairPromptAgent:
    """Build human-operator style prompts for every known repair rule."""

    def __init__(self) -> None:
        self._strategies = self._build_registry()

    def strategy_for(self, rule_id: Any) -> RuleRepairStrategy:
        try:
            numeric_rule = int(rule_id)
        except (TypeError, ValueError):
            numeric_rule = 0
        return self._strategies.get(numeric_rule, self._fallback_strategy(numeric_rule))

    def markdown_prompt(
        self,
        issue: dict[str, Any],
        context: str,
        target_text: str,
        line_num: int,
        scope: str,
        feedback: str = "",
    ) -> str:
        rule_id = issue.get("rule_id", 0)
        strategy = self.strategy_for(rule_id)
        issue_json = json.dumps(issue, ensure_ascii=False, indent=2)
        feedback_block = f"\nPREVIOUS ATTEMPT WAS REJECTED:\n{feedback}\n" if feedback else ""
        return f"""
You are a senior RAG Markdown repair operator.
Fix exactly one validation issue with the smallest safe edit.

RULE-SPECIFIC AGENT:
- Rule: {strategy.rule_id}
- Name: {strategy.name}
- Repair target: {strategy.repair_target}
- Operator goal: {strategy.operator_goal}
- Instruction: {strategy.llm_instruction}
- Replacement policy: {strategy.replacement_policy}

ISSUE JSON:
```json
{issue_json}
```

TARGET LINE:
{line_num}

REPLACEMENT SCOPE:
{scope}

CONTEXT SNIPPET:
```markdown
{context}
```

TARGET TEXT TO REPLACE:
```markdown
{target_text}
```
{feedback_block}
GLOBAL SAFETY RULES:
1. Return only replacement Markdown for the requested scope.
2. Do not explain the fix and do not wrap the answer in code fences.
3. Preserve factual meaning, named entities, numbers, links, code, tables, examples, and headings unless the rule directly requires changing them.
4. Do not invent content. If content must be expanded for semantic mass, expand only from visible context.
5. Keep Markdown valid: balanced code fences, complete tables, complete lists, complete HTML tags.
6. For line scope, return one line only.

FIXED MARKDOWN:
"""

    def chunk_prompt(
        self,
        chunk: dict[str, Any],
        review: dict[str, Any],
        embedding_model: str,
        feedback: str = "",
    ) -> str:
        actionable_issues = [
            issue for issue in review.get("issues", [])
            if issue.get("action") != "NONE"
        ]
        strategy_lines = []
        for issue in actionable_issues:
            strategy = self.strategy_for(issue.get("rule_id"))
            strategy_lines.append(
                {
                    "rule_id": strategy.rule_id,
                    "rule_name": strategy.name,
                    "severity": issue.get("severity"),
                    "action": issue.get("action"),
                    "message": issue.get("message"),
                    "operator_goal": strategy.operator_goal,
                    "llm_instruction": strategy.llm_instruction,
                    "replacement_policy": strategy.replacement_policy,
                }
            )

        heading_path = " > ".join(chunk.get("heading_path") or [])
        context = {
            "chunk_id": chunk.get("chunk_id"),
            "heading_path": heading_path,
            "parent_section_id": chunk.get("parent_section_id"),
            "semantic_tags": chunk.get("semantic_tags", []),
            "embedding_model": embedding_model,
            "before_decision": review.get("decision"),
            "before_score": review.get("final_score"),
            "rule_repair_plan": strategy_lines,
        }
        feedback_block = f"\nPrevious repair attempt was rejected: {feedback}\n" if feedback else ""
        return f"""
You are a senior RAG chunk repair operator.
Repair this chunk for embedding by following the rule-specific repair plan.

Context JSON:
```json
{json.dumps(context, ensure_ascii=False, indent=2)}
```

ORIGINAL CHUNK TEXT:
```markdown
{chunk.get("text", "")}
```
{feedback_block}
GLOBAL CHUNK SAFETY RULES:
1. Return only the repaired chunk text.
2. Do not explain the fix and do not wrap the answer in code fences.
3. Preserve factual meaning, named entities, code, tables, lists, examples, API paths, configuration values, and relationships.
4. Do not invent new facts. If adding context, use only heading path, visible text, or metadata shown above.
5. Make the chunk standalone and retrievable.
6. For E5 embeddings, start the returned text with "passage:".
7. Keep Markdown valid and embedding-safe.

REPAIRED CHUNK:
"""

    def covered_rule_ids(self) -> set[int]:
        return set(self._strategies)

    @classmethod
    def _build_registry(cls) -> dict[int, RuleRepairStrategy]:
        registry: dict[int, RuleRepairStrategy] = {}
        for rule_id, name in cls._phase1_rules().items():
            registry[rule_id] = cls._strategy(
                rule_id,
                name,
                "clean.md",
                "Repair Markdown normalization while preserving original meaning.",
                cls._phase1_instruction(rule_id, name),
                "Replace only the affected markdown line or context snippet.",
            )
        for rule_id, name in cls._qualification_rules().items():
            registry[rule_id] = cls._strategy(
                rule_id,
                name,
                "clean.md",
                "Repair validation issue in the source Markdown before chunking.",
                cls._qualification_instruction(rule_id, name),
                "Replace the mapped paragraph/block in clean markdown.",
            )
        for rule_id, name in cls._chunk_planning_rules().items():
            registry[rule_id] = cls._strategy(
                rule_id,
                name,
                "index.json",
                "Repair chunk planning metadata or chunk text while preserving semantic boundaries.",
                cls._chunk_planning_instruction(rule_id, name),
                "Repair the affected chunk text or metadata, then rerun chunk review.",
            )
        for rule_id, name in cls._chunk_review_rules().items():
            registry[rule_id] = cls._strategy(
                rule_id,
                name,
                "index.json",
                "Repair embedding readiness issue in a chunk.",
                cls._chunk_review_instruction(rule_id, name),
                "Replace only the affected chunk text in flat_chunk_view.",
            )
        return registry

    @staticmethod
    def _strategy(
        rule_id: int,
        name: str,
        repair_target: str,
        operator_goal: str,
        llm_instruction: str,
        replacement_policy: str,
    ) -> RuleRepairStrategy:
        return RuleRepairStrategy(
            rule_id=rule_id,
            name=name,
            repair_target=repair_target,
            operator_goal=operator_goal,
            llm_instruction=llm_instruction,
            replacement_policy=replacement_policy,
        )

    @staticmethod
    def _fallback_strategy(rule_id: int) -> RuleRepairStrategy:
        return RuleRepairStrategy(
            rule_id=rule_id,
            name="Unknown Rule",
            repair_target="clean.md or index.json",
            operator_goal="Repair only the reported issue using the provided context.",
            llm_instruction="Make the smallest valid Markdown or chunk repair that resolves the issue without inventing content.",
            replacement_policy="Replace only the supplied target text.",
        )

    @staticmethod
    def _phase1_rules() -> dict[int, str]:
        return {
            1: "Preserve Original Meaning",
            2: "Preserve Hierarchy",
            3: "Preserve Heading Levels",
            4: "Preserve Heading Path",
            5: "Preserve Code Blocks",
            6: "Preserve Code Language",
            7: "Preserve Tables",
            8: "Preserve Table Structure",
            9: "Preserve Lists",
            10: "Preserve Nested Lists",
            11: "Preserve Links",
            12: "Preserve Link Metadata",
            13: "Preserve Images As Metadata",
            14: "Preserve Image References",
            15: "Preserve Block Boundaries",
            16: "Preserve Section Boundaries",
            17: "Remove Presentation Noise",
            18: "Remove Badge Noise",
            19: "Normalize Whitespace",
            20: "Preserve Semantic Breaks",
            21: "Assign Document Order",
            22: "Assign Stable IDs",
            23: "Detect Boilerplate",
            24: "Detect Generated Noise",
            25: "Do Not Merge Different Content Types",
            26: "Create Typed Blocks",
            27: "Preserve Quotes",
            28: "Preserve Front Matter",
            29: "Normalize UTF-8",
            30: "Do Not Chunk In Phase 1",
            31: "Remove Retrieval Poison",
            32: "Remove Navigation Trees",
            33: "Remove Repeated Boilerplate",
            34: "Semantic Density Scoring",
            35: "Dense Block Detection",
            36: "Merge Sparse Metadata Blocks",
            37: "Topic Shift Detection",
            38: "Heading Soft Boundary",
            39: "Continuation Detection",
            40: "Section Continuity",
            41: "Code References Near Code",
            42: "Table Explanation Near Table",
            43: "API Endpoint With Description",
            44: "Avoid Tiny Chunks",
            45: "Avoid Orphan Chunks",
            46: "Avoid Huge Chunks",
            47: "Context Preservation",
            48: "Heading Path Injection",
            49: "Section Metadata Preservation",
            50: "Parent Context Available",
            51: "Embed Semantic Units",
            52: "Never Split Code",
            53: "Never Split Tables",
            54: "Preserve Enumerations",
            55: "Preserve Requirements",
            56: "Chunk By Meaning",
            57: "Preserve Procedures",
            58: "Preserve Examples",
            59: "Preserve Definitions",
            60: "Preserve API Blocks",
            61: "Preserve Config Blocks",
            62: "Preserve FAQ Blocks",
            63: "Entity Preservation",
            64: "Named Entity Preservation",
            65: "Alias Preservation",
            66: "Definition Preservation",
            67: "Terminology Preservation",
            68: "Acronym Preservation",
            69: "Reference Preservation",
            70: "Cross Reference Preservation",
        }

    @staticmethod
    def _qualification_rules() -> dict[int, str]:
        names = {
            101: "Heading Level Gaps",
            102: "Heading Hierarchy",
            103: "Empty Heading",
            104: "Section Too Small",
            105: "Section Too Large",
            106: "Document Order",
            107: "Broken Code Block",
            108: "Broken Table",
            109: "Broken Link",
            110: "Broken Image Reference",
            111: "Empty Semantic Block",
            112: "Near Empty Block",
            113: "Low Information Density",
            114: "Boilerplate Content",
            115: "Navigation Element",
            116: "Footer Content",
            117: "Header Noise",
            118: "Badge Image",
            119: "Repeated Content",
            120: "Duplicate Section",
            121: "Question Answerability",
            122: "Definition Detection",
            123: "Procedure Detection",
            124: "API Documentation",
            125: "Requirements Detection",
            126: "Configuration Detection",
            127: "Troubleshooting Detection",
            128: "FAQ Detection",
            129: "Reference Detection",
            130: "Cross Reference Detection",
            131: "Code Anchoring",
            132: "Table Anchoring",
            133: "List Anchoring",
            134: "Quote Anchoring",
            135: "Example Anchoring",
            136: "Heading Context",
            137: "Parent Context",
            138: "Sibling Context",
            139: "Continuation Detection",
            140: "Topic Boundary",
            141: "Minimum Semantic Mass",
            142: "Maximum Semantic Mass",
            143: "Entity Density",
            144: "Keyword Density",
            145: "Definition Presence",
            146: "Reference Integrity",
            147: "Self Sufficiency",
            148: "Context Dependency",
            149: "Retrieval Utility",
            150: "Embedding Readiness",
        }
        return names

    @staticmethod
    def _chunk_planning_rules() -> dict[int, str]:
        names = {
            201: "Build Semantic Regions First",
            202: "Heading Soft Boundary",
            203: "H1 Never Merge",
            204: "H2 Usually Separate",
            205: "H3 H4 Merge Candidates",
            206: "Tiny Section Expansion",
            207: "Parent Expansion",
            208: "Heading Path Injection",
            209: "Preserve Definition Blocks",
            210: "Preserve Requirement Blocks",
            211: "Preserve Procedure Blocks",
            212: "Preserve API Blocks",
            213: "Preserve Config Blocks",
            214: "Preserve Examples",
            215: "Code Anchor Rule",
            216: "Table Anchor Rule",
            217: "List Anchor Rule",
            218: "Quote Anchor Rule",
            219: "FAQ Anchor Rule",
            220: "Cross Reference Preservation",
            221: "Semantic Cohesion First",
            222: "Adaptive Chunk Size",
            223: "Similarity Valley Detection",
            224: "Local Minimum Detection",
            225: "Topic Drift Detection",
            226: "Chunk Self Sufficiency",
            227: "No Orphan Chunks",
            228: "No Contextless Chunks",
            229: "Entity Preservation",
            230: "Alias Preservation",
            231: "Semantic Density Scoring",
            232: "Merge Low Density Blocks",
            233: "Chunk Confidence Score",
            234: "Retrieval Utility Score",
            235: "Reject Bad Chunks",
            236: "Multi Pass Chunking",
        }
        return names

    @staticmethod
    def _chunk_review_rules() -> dict[int, str]:
        return {
            301: "Chunk Empty",
            302: "Only Symbols",
            303: "Only Heading",
            304: "Broken Markdown",
            305: "Broken Code Block",
            306: "Broken Table",
            307: "Broken List",
            308: "Heading Path Exists",
            309: "Parent Context Exists",
            310: "Chunk Metadata Exists",
            311: "Too Small Chunk",
            312: "Dangerously Small",
            313: "Too Large",
            314: "Very Large",
            315: "Token Density",
            316: "Semantic Density",
            317: "Definition Score",
            318: "Procedure Score",
            319: "Instruction Score",
            320: "Knowledge Score",
            321: "Question Answerability",
            322: "Standalone Understandability",
            323: "Context Dependency",
            324: "Topic Consistency",
            325: "Semantic Cohesion",
            326: "Contains Retrievable Information",
            327: "Contains Actionable Information",
            328: "Contains Definition",
            329: "Contains Entities",
            330: "Contains Keywords",
            331: "Contains Searchable Concepts",
            332: "Contains Relationships",
            333: "Navigation Detection",
            334: "Badge Detection",
            335: "Footer Detection",
            336: "Boilerplate Detection",
            337: "License Detection",
            338: "Repeated Chunk Detection",
            339: "Near Duplicate Chunk Detection",
            340: "Heading Injection Exists",
            341: "Section Context Exists",
            342: "Document Context Exists",
            343: "Cross Reference Metadata Exists",
            344: "Prefix Validation",
            345: "Language Validation",
            346: "Unicode Validation",
            347: "Encoding Validation",
            348: "Mixed Language Detection",
            349: "Embedding Cost Score",
            350: "Embedding Readiness Score",
        }

    @staticmethod
    def _phase1_instruction(rule_id: int, name: str) -> str:
        if rule_id in {17, 18, 23, 24, 31, 32, 33}:
            return "Remove or isolate retrieval poison, navigation, badges, decorative boilerplate, and generated noise without touching real content."
        if rule_id in {5, 6, 7, 8, 9, 10, 41, 42, 52, 53, 54, 55}:
            return "Preserve the complete block exactly enough that code, tables, lists, requirements, and examples remain intact."
        if rule_id in {19, 20, 29}:
            return "Normalize whitespace or encoding artifacts while preserving semantic breaks and readable text."
        if rule_id in {44, 45, 46, 47, 48, 49, 50, 56}:
            return "Improve semantic chunk readiness by preserving context and preventing tiny, orphan, or oversized units."
        return f"Apply {name} with the smallest Markdown repair that preserves meaning."

    @staticmethod
    def _qualification_instruction(rule_id: int, name: str) -> str:
        if 101 <= rule_id <= 110:
            return "Repair structural Markdown: headings, document order, code fences, tables, links, or images."
        if 111 <= rule_id <= 120:
            return "Remove or repair low-value semantic noise while preserving useful domain content."
        if 121 <= rule_id <= 130:
            return "Make the block answerable and retrievable without inventing information."
        if 131 <= rule_id <= 140:
            return "Keep anchored content with its explanation and preserve context boundaries."
        if 141 <= rule_id <= 150:
            return "Adjust semantic mass and context so the block is suitable for embedding and retrieval."
        return f"Repair validation rule {name} using only visible context."

    @staticmethod
    def _chunk_planning_instruction(rule_id: int, name: str) -> str:
        if rule_id in {203, 204, 205, 206, 207, 232}:
            return "Merge or expand only when it improves self-sufficiency without crossing major topic boundaries."
        if rule_id in {209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219}:
            return "Keep the protected semantic block together with its explanation and context."
        if rule_id in {208, 220, 226, 227, 228, 229, 230}:
            return "Inject heading path, parent context, entities, aliases, and references so the chunk can stand alone."
        return f"Repair chunk planning issue {name} while preserving semantic cohesion."

    @staticmethod
    def _chunk_review_instruction(rule_id: int, name: str) -> str:
        if rule_id in {301, 302, 303, 311, 312}:
            return "Do not fabricate facts. Expand only from neighboring context, heading path, or visible text; otherwise leave repair to rechunking."
        if rule_id in {304, 305, 306, 307}:
            return "Fix Markdown syntax exactly: balance links, code fences, tables, and list markers."
        if rule_id in {308, 309, 340, 341, 342, 343, 344}:
            return "Inject compact heading, parent, document, cross-reference, or E5 prefix context without changing facts."
        if rule_id in {313, 314, 324, 325, 349}:
            return "Split or condense by semantic topic while preserving complete protected blocks."
        if rule_id in {315, 316, 321, 322, 323, 326, 329, 330, 331, 332, 350}:
            return "Improve standalone retrievability by adding visible context, entities, relationships, and searchable concepts."
        if rule_id in {333, 334, 335, 336, 337, 338, 339}:
            return "Remove or reduce noise, duplicates, navigation, badges, footer, license, or boilerplate unless it is the document subject."
        if rule_id in {345, 346, 347, 348}:
            return "Normalize language, Unicode, encoding, and mixed-language artifacts while preserving meaning."
        return f"Repair chunk review issue {name} using the supplied chunk context."
