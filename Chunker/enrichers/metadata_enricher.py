"""
enrichers/metadata_enricher.py

Attaches rich metadata to each SemanticChunk before it is stored.

Metadata added (all optional via MetadataConfig):
  source_file     — relative path of the .md file
  breadcrumb      — full header chain ("# A > ## B > ### C")
  section_header  — the deepest / most specific header title
  chunk_type      — "text" | "table" | "image_description"
  chunk_index     — sequential index within the document
  language        — "en" | "es" | "unknown"
  page_numbers    — list of page numbers this chunk likely falls on
                    (approximated from the _metadata.json if available)
  total_pages     — total pages in the source PDF
  source_pdf      — original PDF filename from metadata.json
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from Chunker.config import MetadataConfig
from Chunker.preprocessors.language_detector import detect_language
from Chunker.splitters.semantic_chunker import SemanticChunk


@dataclass
class EnrichedChunk:
    """Final output unit: content + all metadata."""
    content: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"content": self.content, "metadata": self.metadata}


def _load_page_metadata(md_path: Path) -> Optional[dict]:
    """
    Look for a sibling _metadata.json file and load it.
    E.g. for  outputs/Binder2/Binder2.md  →  outputs/Binder2/Binder2_metadata.json
    """
    stem = md_path.stem  # "Binder2"
    meta_path = md_path.parent / f"{stem}_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _estimate_page_numbers(
    chunk_index: int,
    total_chunks: int,
    page_metadata: dict,
) -> list[int]:
    """
    Approximate which PDF page(s) a chunk falls on.

    Strategy: distribute chunks evenly across pages weighted by each
    page's num_chunks count from the metadata.
    """
    pages = page_metadata.get("pages", [])
    if not pages or total_chunks == 0:
        return []

    # Build a mapping: chunk_index → page_num
    cumulative = 0
    for page in pages:
        page_chunks = page.get("num_chunks", 0)
        if cumulative + page_chunks > chunk_index:
            return [page["page_num"] + 1]   # 1-indexed for human readability
        cumulative += page_chunks

    # If beyond all known chunks, assign last page
    return [pages[-1]["page_num"] + 1]


class MetadataEnricher:
    """
    Attaches metadata to a list of SemanticChunks.

    Usage:
        enricher = MetadataEnricher(config)
        enriched = enricher.enrich(chunks, md_path)
    """

    def __init__(self, config: Optional[MetadataConfig] = None):
        self.cfg = config or MetadataConfig()

    def enrich(
        self,
        chunks: list[SemanticChunk],
        md_path: Path,
    ) -> list[EnrichedChunk]:
        """
        Enrich all chunks from a single markdown file.
        md_path is used to locate the sibling _metadata.json.
        """
        # Load page metadata once per file
        page_meta = _load_page_metadata(md_path) if self.cfg.include_page_metadata else None
        total_chunks = len(chunks)

        enriched: list[EnrichedChunk] = []
        for chunk in chunks:
            meta = self._build_metadata(chunk, md_path, page_meta, total_chunks)
            enriched.append(EnrichedChunk(content=chunk.content, metadata=meta))

        return enriched

    def _build_metadata(
        self,
        chunk: SemanticChunk,
        md_path: Path,
        page_meta: Optional[dict],
        total_chunks: int,
    ) -> dict:
        meta: dict = {}

        if self.cfg.include_source_file:
            meta["source_file"] = str(md_path)

        if self.cfg.include_breadcrumb:
            meta["breadcrumb"] = chunk.breadcrumb
            # Most specific header (deepest level)
            if chunk.header_texts:
                meta["section_header"] = chunk.header_texts[-1]
            else:
                meta["section_header"] = ""

        if self.cfg.include_chunk_type:
            meta["chunk_type"] = chunk.section_type

        if self.cfg.include_chunk_index:
            meta["chunk_index"] = chunk.chunk_index

        if self.cfg.include_language:
            meta["language"] = detect_language(chunk.content)

        if self.cfg.include_page_metadata and page_meta:
            meta["source_pdf"] = page_meta.get("file_name", "")
            meta["total_pages"] = page_meta.get("num_pages", 0)
            meta["page_numbers"] = _estimate_page_numbers(
                chunk.chunk_index, total_chunks, page_meta
            )

        return meta
