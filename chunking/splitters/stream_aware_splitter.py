"""Stream-aware chunking that preserves table integrity and section hierarchy."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from core.contracts.chunking import ChunkArtifact, ChunkingResult
from core.contracts.stream import DocumentStream
from core.logging import configure_logging, log_event


class StreamAwareChunker:
    chunker_name = "stream_aware_splitter"

    def chunk_stream(
        self, 
        stream: DocumentStream, 
        document_id: str, 
        max_chunk_size: int = 1000
    ) -> ChunkingResult:
        logger = configure_logging()
        log_event(logger, 20, "chunker_start", "stream-aware chunking started", document_id=document_id)
        
        chunks: list[ChunkArtifact] = []
        current_text_lines: list[str] = []
        section_stack: list[str] = []
        chunk_index = 0
        current_contains_image = False
        
        for block in stream.blocks:
            if block.block_type == "heading":
                level = block.level or 1
                text = block.text or ""
                
                # Maintain section hierarchy stack
                while len(section_stack) >= level:
                    section_stack.pop()
                section_stack.append(text)
                
                # Finalize current chunk if it has content
                if current_text_lines:
                    chunk = self._create_chunk(
                        text_lines=current_text_lines,
                        section_path=list(section_stack),
                        document_id=document_id,
                        chunk_index=chunk_index,
                        contains_image=current_contains_image,
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_index += 1
                    current_text_lines = []
                    current_contains_image = False
                    
            elif block.block_type == "paragraph":
                if block.text:
                    current_text_lines.append(block.text)
                    if len("\n\n".join(current_text_lines)) > max_chunk_size:
                        chunk = self._create_chunk(
                            text_lines=current_text_lines,
                            section_path=list(section_stack),
                            document_id=document_id,
                            chunk_index=chunk_index,
                            contains_image=current_contains_image,
                        )
                        if chunk:
                            chunks.append(chunk)
                            chunk_index += 1
                        current_text_lines = []
                        current_contains_image = False
                        
            elif block.block_type == "table":
                # CRITICAL: Never split a table. Finalize current text first.
                if current_text_lines:
                    chunk = self._create_chunk(
                        text_lines=current_text_lines,
                        section_path=list(section_stack),
                        document_id=document_id,
                        chunk_index=chunk_index,
                        contains_image=current_contains_image,
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_index += 1
                    current_text_lines = []
                    current_contains_image = False
                
                # Create a dedicated, intact table chunk
                table_md = self._table_to_markdown(block.table_cells)
                chunk = self._create_chunk(
                    text_lines=[table_md],
                    section_path=list(section_stack),
                    document_id=document_id,
                    chunk_index=chunk_index,
                    contains_table=True,
                    table_markdown=table_md
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1
                    
            elif block.block_type == "image":
                if block.image and block.image.description:
                    current_text_lines.append(f"[Image Description: {block.image.description}]")
                    current_contains_image = True

        # Finalize any remaining text
        if current_text_lines:
            chunk = self._create_chunk(
                text_lines=current_text_lines,
                section_path=list(section_stack),
                document_id=document_id,
                chunk_index=chunk_index,
                contains_image=current_contains_image,
            )
            if chunk:
                chunks.append(chunk)

        result = ChunkingResult(
            source_path=stream.source_path,
            document_id=document_id,
            chunks=chunks,
            debug_info={"chunker": self.chunker_name, "total_chunks": len(chunks)}
        )
        result.validate()
        log_event(logger, 20, "chunker_end", "stream-aware chunking completed", chunks=len(chunks))
        return result

    def _create_chunk(
        self,
        text_lines: list[str],
        section_path: list[str],
        document_id: str,
        chunk_index: int,
        contains_table: bool = False,
        table_markdown: str | None = None,
        contains_image: bool = False
    ) -> ChunkArtifact | None:
        text = "\n\n".join(line.strip() for line in text_lines if line.strip()).strip()
        if not text:
            return None
            
        chunk_id = f"{document_id}-chunk-{chunk_index}"
        
        metadata: dict[str, Any] = {
            "section_path": section_path,
            "contains_table": contains_table,
            "contains_image": contains_image,
        }
        if table_markdown:
            metadata["table_markdown"] = table_markdown
            
        return ChunkArtifact(
            chunk_id=chunk_id,
            document_id=document_id,
            text=text,
            section_path=section_path,
            chunk_index=chunk_index,
            metadata=metadata
        )

    def _table_to_markdown(self, table_cells: list[Any]) -> str:
        rows: dict[int, list[Any]] = {}
        for cell in table_cells:
            rows.setdefault(getattr(cell, "row_index", 0), []).append(cell)

        if not rows:
            return ""

        ordered_rows = [rows[index] for index in sorted(rows)]
        normalized_rows = [
            sorted(row, key=lambda cell: getattr(cell, "column_index", 0)) for row in ordered_rows
        ]
        
        header_cells = [getattr(cell, "text", "").strip() for cell in normalized_rows[0]]
        if not header_cells:
            return ""

        markdown_rows = [
            "| " + " | ".join(header_cells) + " |",
            "| " + " | ".join("---" for _ in header_cells) + " |",
        ]
        for row in normalized_rows[1:]:
            markdown_rows.append(
                "| " + " | ".join(getattr(cell, "text", "").strip() for cell in row) + " |"
            )
        return "\n".join(markdown_rows)

    def validate(self) -> dict[str, Any]: return {"status": "ok", "chunker": self.chunker_name}
    def debug(self) -> dict[str, Any]: return {"chunker": self.chunker_name}
    def health_check(self) -> dict[str, Any]: return {"status": "ok", "chunker": self.chunker_name}


def chunk_stream(stream: DocumentStream, document_id: str, max_chunk_size: int = 1000) -> ChunkingResult:
    return StreamAwareChunker().chunk_stream(stream, document_id, max_chunk_size)

def result_to_json(result: ChunkingResult) -> str:
    return json.dumps(asdict(result), indent=2, sort_keys=True, default=str)