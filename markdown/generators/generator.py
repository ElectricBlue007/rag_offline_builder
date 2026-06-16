"""RAG-optimized Markdown generation from unified document stream."""
from __future__ import annotations
import json
import yaml
from dataclasses import asdict
from pathlib import Path
from typing import Any
from core.contracts.stream import DocumentStream, OrderedDocumentBlock
from core.contracts.markdown import MarkdownGenerationResult, MarkdownSection
from core.logging import configure_logging, log_event
from markdown.utils.text_sanitizer import TextSanitizer
from markdown.utils.heading_normalizer import HeadingNormalizer
from markdown.utils.table_formatter import TableFormatter

class RAGMarkdownGenerator:
    """Generates strictly hierarchical, RAG-optimized Markdown from a DocumentStream."""
    generator_name = "rag_markdown_generator"

    def __init__(self):
        self.logger = configure_logging()
        self.sanitizer = TextSanitizer()
        self.normalizer = HeadingNormalizer()
        self.table_formatter = TableFormatter()

    def generate(self, stream: DocumentStream, metadata: dict[str, Any] | None = None) -> MarkdownGenerationResult:
        """Generate RAG-optimized Markdown with YAML frontmatter."""
        log_event(self.logger, 20, "markdown_gen_start", "generating RAG markdown", source_path=stream.source_path)
        
        # 1. Normalize headings to prevent chunker failures
        normalized_blocks = self.normalizer.normalize(stream.blocks)
        
        # 2. Build Markdown and Sections
        markdown_body, sections = self._build_markdown(normalized_blocks)
        
        # 3. Inject YAML Frontmatter (Critical for Parent-Child RAG retrieval)
        frontmatter = dict(metadata or {})
        frontmatter["source_path"] = stream.source_path
        frontmatter_yaml = yaml.dump(frontmatter, sort_keys=False, allow_unicode=True)
        final_markdown = f"---\n{frontmatter_yaml}---\n\n{markdown_body}"

        result = MarkdownGenerationResult(
            source_path=stream.source_path,
            markdown=final_markdown,
            sections=sections,
            frontmatter=frontmatter,
            footnotes={}, # Placeholder for future footnote extractor integration
            stream=stream,
            warnings=stream.warnings,
            debug_info={"generator": self.generator_name, "blocks_processed": len(normalized_blocks)},
        )
        result.validate()
        log_event(self.logger, 20, "markdown_gen_end", "markdown generation completed", sections=len(sections))
        return result

    def _build_markdown(self, blocks: list[OrderedDocumentBlock]) -> tuple[str, list[MarkdownSection]]:
        lines: list[str] = []
        sections: list[MarkdownSection] = []
        section_index = 0

        for block in blocks:
            if block.block_type == "heading":
                heading_prefix = "#" * max(1, block.level or 1)
                text = self.sanitizer.sanitize(block.text or "")
                lines.append(f"{heading_prefix} {text}")
                sections.append(
                    MarkdownSection(
                        section_id=f"section-{section_index}",
                        title=text,
                        level=block.level or 1,
                        markdown=f"{heading_prefix} {text}",
                    )
                )
                section_index += 1
            elif block.block_type == "paragraph":
                lines.append(self.sanitizer.sanitize(block.text or ""))
            elif block.block_type == "table":
                if block.table_cells:
                    lines.append("")
                    lines.extend(self.table_formatter.format_table(block.table_cells))
                    lines.append("")
            elif block.block_type == "image":
                if block.image:
                    lines.append("")
                    img_ref = f"![{self.sanitizer.sanitize(block.image.alt_text or block.image.id)}]({block.image.id})"
                    lines.append(img_ref)
                    if block.image.description:
                        lines.append(f"*Visual Context: {self.sanitizer.sanitize(block.image.description)}*")
                    if block.image.caption:
                        lines.append(f"*Caption: {self.sanitizer.sanitize(block.image.caption)}*")
                    lines.append("")

        markdown = "\n".join(lines).strip()
        if not sections:
            sections.append(MarkdownSection(section_id="section-0", title="Document", level=1, markdown=markdown or "# Document"))
        return markdown, sections

    def validate(self) -> dict[str, Any]: return {"status": "ok", "generator": self.generator_name}
    def debug(self) -> dict[str, Any]: return {"generator": self.generator_name}
    def health_check(self) -> dict[str, Any]: return {"status": "ok", "generator": self.generator_name}

# Backward compatibility alias
MarkdownGenerator = RAGMarkdownGenerator

def generate_markdown(stream: DocumentStream, metadata: dict[str, Any] | None = None) -> MarkdownGenerationResult:
    return RAGMarkdownGenerator().generate(stream, metadata)

def result_to_json(result: MarkdownGenerationResult) -> str:
    return json.dumps(asdict(result), indent=2, sort_keys=True, default=str)