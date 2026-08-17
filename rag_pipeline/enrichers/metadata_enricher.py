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

Also attaches (always, not config-gated — this is foundational plumbing
rather than descriptive metadata):
  chunk_id        — stable, deterministic identifier (16-char hex hash
                    of source_file + chunk_index). Re-running the pipeline
                    on unchanged input reproduces the exact same IDs,
                    which is what makes ChromaDB .upsert() safe to call
                    repeatedly (roadmap step 4) — same chunk always maps
                    to the same vector-store entry rather than duplicating.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from rag_pipeline.config import MetadataConfig
from rag_pipeline.document_reference import clean_heading, reference_fields
from rag_pipeline.enrichers.page_mapper import (
    MarkerInfo,
    assign_pages_by_length,
    build_page_index,
    resolve_marker_pages,
)
from rag_pipeline.preprocessors.language_detector import detect_language
from rag_pipeline.splitters.semantic_chunker import SemanticChunk


# ---------------------------------------------------------------------------
# Inline page-marker detection
#
# Different OCR/markdown-conversion tools mark page boundaries differently.
# This is a registry, not a single hardcoded pattern — each entry is tried
# independently, so a document from any one of these tools gets its page
# numbers extracted correctly, and documents that use none of them simply
# fall through with page_numbers=[] (handled gracefully everywhere
# downstream, never a crash).
#
# Known conventions covered so far:
#   - Chandra OCR:         <span id="page-4-0"></span>            (0-indexed)
#   - Docling / many MD
#     exporters:            <!-- page: 5 -->  or  <!-- Page 5 -->  (1-indexed)
#   - marker-pdf style:     {4}------------------------------      (0-indexed)
#   - Generic plain-text:   --- Page 5 ---   or   [Page 5]          (1-indexed)
#
# To add support for another tool: run it on one sample doc, look at the
# raw markdown for whatever marks a page break, and add one more
# (regex, page_number_from_match) tuple below — no other code needs to
# change.
# ---------------------------------------------------------------------------
_PAGE_MARKER_HANDLERS: list[tuple[re.Pattern, Callable[[re.Match], int]]] = [
    (re.compile(r'<span id="page-(\d+)-\d+"></span>'), lambda m: int(m.group(1)) + 1),
    (re.compile(r'<!--\s*page:?\s*(\d+)\s*-->', re.IGNORECASE), lambda m: int(m.group(1))),
    (re.compile(r'^\{(\d+)\}-+\s*$', re.MULTILINE), lambda m: int(m.group(1)) + 1),
    (re.compile(r'-{2,}\s*page\s*(\d+)\s*-{2,}', re.IGNORECASE), lambda m: int(m.group(1))),
    (re.compile(r'\[page\s*(\d+)\]', re.IGNORECASE), lambda m: int(m.group(1))),
]


def _extract_and_strip_pages(text: str) -> tuple[str, list[int]]:
    """
    Tries every known marker convention against `text`. Returns
    (cleaned_text, page_numbers): page_numbers is the sorted, de-duplicated,
    1-indexed list of pages this text touches — empty if none of the known
    conventions match (e.g. a tool we haven't seen a sample of yet, or a
    chunk with no page boundary inside it). Markers that DO match are
    stripped out of the text so they never leak into embeddings or the
    LLM's context; markers that don't match are left completely untouched.
    """
    cleaned, pages, _starts_with, _ends_with = _extract_pages_with_positions(text)
    return cleaned, pages


def _extract_pages_with_positions(text: str) -> tuple[str, list[int], bool, bool]:
    """
    Like _extract_and_strip_pages, but also reports WHERE the markers sat:

        (cleaned_text, pages, starts_with_marker, ends_with_marker)

    Position matters because a page marker is a BOUNDARY, not a label:

      * starts_with_marker=False means text appears before the first marker,
        and that text belongs to the PREVIOUS page.
      * ends_with_marker=True means the last marker has no text after it, so
        it announces the NEXT chunk's page rather than describing this one.

    Ignoring this is what made pages drift by one at every page break.
    Offsets are measured against the original `text` so that stripping one
    pattern cannot shift the offsets recorded for another.
    """
    if not text:
        return text, [], False, False

    pages: set[int] = set()
    first_starts: list[int] = []
    last_ends: list[int] = []
    cleaned = text

    for pattern, get_page in _PAGE_MARKER_HANDLERS:
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        for m in matches:
            pages.add(get_page(m))
        first_starts.append(matches[0].start())
        last_ends.append(matches[-1].end())
        cleaned = pattern.sub("", cleaned)

    if not pages:
        return cleaned.strip(), [], False, False

    starts_with_marker = not text[: min(first_starts)].strip()
    ends_with_marker = not text[max(last_ends):].strip()

    return cleaned.strip(), sorted(pages), starts_with_marker, ends_with_marker


def compute_chunk_id(source_file: str, chunk_index: int) -> str:
    """
    Deterministic, stable chunk identifier. Same (source_file, chunk_index)
    pair always produces the same ID — this is what makes downstream
    ChromaDB upserts idempotent across repeated pipeline runs on
    unchanged input, rather than accumulating duplicate entries.
    """
    raw = f"{source_file}::{chunk_index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class EnrichedChunk:
    """Final output unit: content + embed-ready text + all metadata."""
    content: str
    metadata: dict = field(default_factory=dict)
    # Stable identifier — see compute_chunk_id() above. Always present;
    # not config-gated since downstream storage (ChromaDB) requires it.
    chunk_id: str = ""
    # Embedding-ready text (see SemanticChunk.normalized_text docstring).
    # Defaults to content for backward compatibility with any code
    # constructing EnrichedChunk directly without this field.
    normalized_text: str = ""
    # Contextual-retrieval text: normalized_text with a situating blurb
    # prepended (see enrichers/contextual_enricher.py). Defaults to
    # normalized_text for backward compatibility.
    contextualized_text: str = ""
    # Final embedding input: contextualized_text with the E5 "passage: "
    # prefix applied (see enrichers/embed_text_builder.py). THIS is what
    # actually gets passed to the embedding model. Defaults to
    # contextualized_text for backward compatibility.
    embed_text: str = ""

    def __post_init__(self):
        if not self.normalized_text:
            self.normalized_text = self.content
        if not self.contextualized_text:
            self.contextualized_text = self.normalized_text
        if not self.embed_text:
            self.embed_text = self.contextualized_text

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "normalized_text": self.normalized_text,
            "contextualized_text": self.contextualized_text,
            "embed_text": self.embed_text,
            "metadata": self.metadata,
        }


def _load_page_metadata(md_path: Path) -> Optional[dict]:
    """
    Look for a sibling sidecar JSON file and load it. Naming conventions
    vary by tool — Chandra writes "<stem>_meta.json", other setups use
    "<stem>_metadata.json", "<stem>.metadata.json", or "<stem>.json".
    Tries them in order and returns the first one found.

    NOTE: this sidecar file is only used for supplementary info
    (total_pages / source_pdf fallback) — actual per-chunk page numbers
    come from inline markers via _extract_and_strip_pages(), which is
    more precise and doesn't depend on any sidecar file existing, or on
    which OCR tool produced it (see the marker-handler registry above).
    """
    stem = md_path.stem  # "big test"
    candidates = [
        md_path.parent / f"{stem}_metadata.json",
        md_path.parent / f"{stem}_meta.json",
        md_path.parent / f"{stem}.metadata.json",
        md_path.parent / f"{stem}.json",
    ]
    for meta_path in candidates:
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
    Fallback ONLY — used when a chunk has no inline page-span markers
    (e.g. sidecar-only metadata from an OCR tool that doesn't emit
    <span id="page-N-M"> tags). Approximates which PDF page(s) a chunk
    falls on by distributing chunks evenly across pages, weighted by
    each page's num_chunks count from the metadata.

    Expects the older "{'pages': [{'num_chunks':.., 'page_num':..}]}"
    schema. Chandra's own "_meta.json" does NOT have this shape (it has
    "page_stats" instead, with no chunk-to-page mapping) — for Chandra
    input this will simply return [] and the caller falls back further.
    """
    pages = page_metadata.get("pages", [])
    if not pages or total_chunks == 0:
        return []

    cumulative = 0
    for page in pages:
        page_chunks = page.get("num_chunks", 0)
        if cumulative + page_chunks > chunk_index:
            return [page["page_num"] + 1]   # 1-indexed for human readability
        cumulative += page_chunks

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

        # ── Pass 1: clean each chunk and record what its markers say ─────
        # Page attribution is a DOCUMENT-level question, not a per-chunk
        # one. Markers are boundaries, so a chunk containing no marker
        # still sits on whatever page was open when it began, and a chunk's
        # position along the whole text is what locates it when there are
        # no markers at all. Neither can be answered while looking at one
        # chunk in isolation -- hence two passes.
        cleaned_texts: list[tuple[str, str, str, str]] = []
        marker_infos: list[MarkerInfo] = []
        for chunk in chunks:
            clean_content, pages, starts_with, ends_with = _extract_pages_with_positions(
                chunk.content
            )
            clean_normalized, _ = _extract_and_strip_pages(chunk.normalized_text)
            clean_contextualized, _ = _extract_and_strip_pages(chunk.contextualized_text)
            clean_embed, _ = _extract_and_strip_pages(chunk.embed_text)
            cleaned_texts.append(
                (clean_content, clean_normalized, clean_contextualized, clean_embed)
            )
            marker_infos.append(MarkerInfo(
                pages=pages,
                starts_with_marker=starts_with,
                ends_with_marker=ends_with,
            ))

        # ── Pass 2: resolve pages for the document, then build metadata ──
        page_numbers, page_source = self._resolve_pages(
            marker_infos,
            [texts[0] and len(texts[0]) or 0 for texts in cleaned_texts],
            page_meta,
        )

        enriched: list[EnrichedChunk] = []
        for position, chunk in enumerate(chunks):
            clean_content, clean_normalized, clean_contextualized, clean_embed = (
                cleaned_texts[position]
            )
            meta = self._build_metadata(
                chunk,
                md_path,
                page_meta,
                total_chunks,
                page_numbers[position],
                page_source,
            )
            enriched.append(EnrichedChunk(
                content=clean_content,
                metadata=meta,
                chunk_id=compute_chunk_id(str(md_path), chunk.chunk_index),
                normalized_text=clean_normalized,
                contextualized_text=clean_contextualized,
                embed_text=clean_embed,
            ))

        return enriched

    def _resolve_pages(
        self,
        marker_infos: list,
        lengths: list,
        page_meta: Optional[dict],
    ) -> tuple[list, str]:
        """
        Decide the page numbers for every chunk in one document, and report
        how they were obtained. Best method available wins:

          1. "markers"   -- any inline page boundary exists anywhere in the
                            document. Exact, and the running cursor gives a
                            page even to chunks with no marker inside them.
          2. "estimated" -- no markers, but the sidecar describes page
                            sizes, so position within the text maps onto a
                            page. Approximate, and labelled as such.
          3. "none"      -- nothing to go on. Better to show no page than
                            to invent one.
        """
        if not self.cfg.include_page_metadata:
            return [[] for _ in marker_infos], "none"

        if any(info.pages for info in marker_infos):
            return resolve_marker_pages(marker_infos), "markers"

        index = build_page_index(page_meta)
        if index is not None:
            return assign_pages_by_length(lengths, index), "estimated"

        return [[] for _ in marker_infos], "none"

    def _build_metadata(
        self,
        chunk: SemanticChunk,
        md_path: Path,
        page_meta: Optional[dict],
        total_chunks: int,
        page_numbers: Optional[list[int]] = None,
        page_source: str = "none",
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

        if self.cfg.include_page_metadata:
            # Pages are resolved once for the whole document in enrich();
            # see _resolve_pages(). page_source records HOW they were found
            # so a citation can be honest about its own precision:
            #   "markers"   exact, from inline page boundaries
            #   "estimated" approximate, from sidecar page sizes
            #   "none"      unknown; no markers and no usable sidecar
            meta["page_numbers"] = list(page_numbers or [])
            meta["page_source"] = page_source if meta["page_numbers"] else "none"

            # source_pdf / total_pages: sidecar-file schemas vary a lot by
            # OCR tool (Chandra's has "page_stats" and no "file_name"/
            # "num_pages" at all; other tools use "pages", "elements",
            # "page_count", etc). Try the common key names in order rather
            # than assuming one fixed schema, and fall back to deriving a
            # sensible value directly when nothing matches.
            pm = page_meta or {}
            source_pdf = pm.get("file_name") or pm.get("source_file") or pm.get("filename") or ""
            if not source_pdf:
                source_pdf = md_path.stem + ".pdf"
            meta["source_pdf"] = source_pdf

            total_pages = pm.get("num_pages") or pm.get("total_pages") or pm.get("page_count") or 0
            if not total_pages:
                for list_key in ("page_stats", "pages", "elements"):
                    if isinstance(pm.get(list_key), list):
                        total_pages = len(pm[list_key])
                        break
            meta["total_pages"] = total_pages

        # ── Structured document reference ────────────────────────────────
        # The splitter already knows the heading chain a chunk lives under
        # (header_texts / header_levels), but until now only the flattened
        # "breadcrumb" string survived onto the chunk. That was enough to
        # DISPLAY a location, but not enough to say "chapter X, section Y"
        # in a citation, and not enough to filter by chapter later.
        #
        # So the chain is split into named levels here, once, and stored as
        # flat scalar fields (ChromaDB rejects list metadata):
        #   chapter / section / subsection / heading_path / heading_depth
        #   page_display  — "12" or "12-14", collapsed from page_numbers
        #   reference     — the ready-to-read one-liner every consumer shows
        #
        # Labels come from position in the chain, not from '#' depth — many
        # OCR'd documents start at '##' or skip levels, and the outermost
        # heading a chunk sits under is its chapter regardless of depth.
        heading_titles = [clean_heading(t) for t in (chunk.header_texts or [])]
        meta.update(reference_fields({
            **meta,
            "header_texts": heading_titles,
            "header_levels": list(chunk.header_levels or []),
        }))

        return meta
