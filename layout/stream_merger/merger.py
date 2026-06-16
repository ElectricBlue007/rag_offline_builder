"""Merges independent extraction results into a unified document stream."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from core.contracts.content import ContentExtractionResult, ContentElement
from core.contracts.document import ImageArtifact
from core.contracts.layout import ReadingOrderResult
from core.contracts.stream import DocumentStream, OrderedDocumentBlock
from core.contracts.tables import TableExtractionResult
from core.logging import configure_logging, log_event


class StreamMerger:
    merger_name = "stream_merger"

    def merge(
        self,
        source_path: str,
        reading_order: ReadingOrderResult | None,
        content: ContentExtractionResult | None,
        tables: TableExtractionResult | None,
        images: list[ImageArtifact] | None,
    ) -> DocumentStream:
        logger = configure_logging()
        log_event(logger, 20, "merger_start", "stream merger started", source_path=source_path)
        
        blocks: list[OrderedDocumentBlock] = []
        warnings: list[str] = []
        
        # Fallback if no reading order exists
        if reading_order is None or not reading_order.nodes:
            if content and content.elements:
                for idx, element in enumerate(content.elements):
                    blocks.append(self._element_to_block(element, idx))
            else:
                warnings.append("No reading order or content elements provided.")
        else:
            # Queue-based matching to preserve exact chronological order
            content_queue = list(content.elements) if content else []
            table_queue = list(tables.tables) if tables else []
            image_queue = list(images) if images else []
            
            for idx, node in enumerate(reading_order.nodes):
                if node.node_type == "image":
                    img_index = node.metadata.get("image_index")
                    if img_index is not None and img_index < len(image_queue):
                        img = image_queue[img_index]
                        blocks.append(OrderedDocumentBlock(
                            block_id=f"block-{idx}",
                            block_type="image",
                            image=img,
                            page_number=node.page_number,
                            order_index=idx,
                            metadata=node.metadata
                        ))
                    else:
                        warnings.append(f"Image node at index {idx} missing image artifact.")
                        
                elif node.node_type == "table":
                    if table_queue:
                        table_cells = table_queue.pop(0)
                        blocks.append(OrderedDocumentBlock(
                            block_id=f"block-{idx}",
                            block_type="table",
                            table_cells=table_cells,
                            page_number=node.page_number,
                            order_index=idx,
                            metadata=node.metadata
                        ))
                    else:
                        warnings.append(f"Table node at index {idx} missing table artifact.")
                        
                else:  # paragraph or heading
                    matched_element = None
                    if node.text:
                        # Match by exact text to handle out-of-order content extractors
                        for i, el in enumerate(content_queue):
                            if el.text.strip() == node.text.strip():
                                matched_element = content_queue.pop(i)
                                break
                    if matched_element is None and content_queue:
                        matched_element = content_queue.pop(0)
                        
                    if matched_element:
                        blocks.append(self._element_to_block(matched_element, idx, node.page_number))
                    else:
                        warnings.append(f"Text node at index {idx} missing content element.")

        stream = DocumentStream(
            source_path=source_path,
            blocks=blocks,
            warnings=warnings,
            debug_info={"merger": self.merger_name}
        )
        stream.validate()
        log_event(logger, 20, "merger_end", "stream merger completed", blocks=len(blocks))
        return stream

    def _element_to_block(self, element: ContentElement, idx: int, page_number: int | None = None) -> OrderedDocumentBlock:
        block_type = "paragraph"
        level = None
        if element.element_type == "heading":
            block_type = "heading"
            level = element.level
            
        return OrderedDocumentBlock(
            block_id=f"block-{idx}",
            block_type=block_type,
            text=element.text,
            level=level,
            page_number=page_number or element.page_number,
            order_index=idx,
            metadata=element.metadata
        )

    def validate(self) -> dict[str, Any]:
        return {"status": "ok", "merger": self.merger_name}

    def debug(self) -> dict[str, Any]:
        return {"merger": self.merger_name}

    def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "merger": self.merger_name}


def merge_stream(
    source_path: str,
    reading_order: ReadingOrderResult | None,
    content: ContentExtractionResult | None,
    tables: TableExtractionResult | None,
    images: list[ImageArtifact] | None,
) -> DocumentStream:
    return StreamMerger().merge(source_path, reading_order, content, tables, images)


def result_to_json(result: DocumentStream) -> str:
    return json.dumps(asdict(result), indent=2, sort_keys=True, default=str)