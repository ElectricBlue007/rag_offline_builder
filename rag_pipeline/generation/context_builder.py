"""
generation/context_builder.py

Formats retrieved (or reranked) chunks into the context block an LLM
actually reads — the final formatting step before generation.

Accepts both RetrievedChunk (hybrid retrieval only) and RerankedChunk
(post cross-encoder) via duck-typing on the fields both share
(.chunk_id, .text, .metadata), matching the same pattern used in
run_query.py's format_results().

Per-chunk format:
    [1] Table — Binder2.pdf, page 1 — SERVICE: Great circle > Operator actions
    Create SOI — logical entry: CMI track; real entry: Enter "NT"...

The numbered index doubles as a citation anchor: the system prompt
(generation/rag_answerer.py) asks the LLM to reference these numbers
("According to [1]...") so an answer's claims can be traced back to
a specific source chunk.
"""

import re
from dataclasses import dataclass
from typing import Optional, Union

from rag_pipeline.config import ContextBuilderConfig
from rag_pipeline.document_reference import heading_levels


# Defensive fallback only — table_normalizer.py (roadmap step 1)
# already guarantees normalized_text is clean prose for table chunks
# by the time anything reaches retrieval. This regex-based strip is a
# second line of defense for the unusual case where table
# normalization was disabled in config, not a real HTML parser.
_RE_HTML_TAG = re.compile(r'<[^>]+>')

_TYPE_LABELS = {
    "table": "Table",
    "image_description": "Screenshot",
    "text": "Text",
}


def _get_field(chunk: Union[dict, object], field: str, default=""):
    if isinstance(chunk, dict):
        return chunk.get(field, default)
    return getattr(chunk, field, default)


def _strip_hashes(breadcrumb: str) -> str:
    """Same breadcrumb-cleaning logic as contextual_enricher.py — strips
    markdown '#' markers for a clean readable display."""
    segments = breadcrumb.split(" > ")
    cleaned = [re.sub(r'^#+\s*', '', s).strip() for s in segments]
    return " > ".join(cleaned)


def _defensive_strip_html(text: str) -> str:
    """Fallback cleanup — see module docstring. A no-op on the normal
    (table-normalization-enabled) path since there's nothing to strip."""
    if "<" not in text:
        return text
    return _RE_HTML_TAG.sub(" ", text).strip()


def _format_chunk(chunk: Union[dict, object], index: int, cfg: ContextBuilderConfig) -> str:
    metadata = _get_field(chunk, "metadata", {}) or {}
    text = _defensive_strip_html(_get_field(chunk, "text", ""))

    if cfg.max_chars_per_chunk is not None and len(text) > cfg.max_chars_per_chunk:
        text = text[: cfg.max_chars_per_chunk].rstrip() + " ...[truncated]"

    header_parts = []

    if cfg.include_type_label:
        chunk_type = metadata.get("chunk_type", "text")
        header_parts.append(_TYPE_LABELS.get(chunk_type, "Text"))

    if cfg.include_source:
        source = metadata.get("source_pdf") or metadata.get("source_file", "?")
        page_numbers = metadata.get("page_numbers", "")
        # Normally this arrives as a comma-separated STRING — ChromaDB's
        # metadata sanitization (_sanitize_metadata_for_chroma) converts
        # the original list before it ever reaches retrieval. But this
        # function may also be called on chunks that bypassed ChromaDB
        # entirely (e.g. direct pipeline output in tests, or a future
        # retrieval path) — handle the raw list form defensively too,
        # rather than let Python's default list repr ("[1]") leak into
        # the LLM-facing context.
        if isinstance(page_numbers, list):
            page_numbers = ",".join(str(p) for p in page_numbers)
        source_str = source
        if page_numbers:
            source_str += f", page {page_numbers}"
        header_parts.append(source_str)

    if cfg.include_breadcrumb:
        # The heading chain is what tells the model WHERE in the document a
        # passage lives. Rendering it as labelled levels
        # ("Chapter: Prompting > Section: Few-shot") rather than a bare
        # "Prompting > Few-shot" is what makes the model able to repeat the
        # location in its answer: it can only cite a chapter/section by name
        # if the words "Chapter"/"Section" appear next to the titles here.
        levels = heading_levels(metadata)
        if levels and cfg.label_heading_levels:
            header_parts.append(
                " > ".join(f"{lvl['label']}: {lvl['title']}" for lvl in levels)
            )
        elif levels:
            header_parts.append(" > ".join(lvl["title"] for lvl in levels))
        else:
            breadcrumb = metadata.get("breadcrumb", "")
            if breadcrumb and breadcrumb != "(root)":
                header_parts.append(_strip_hashes(breadcrumb))

    header = " — ".join(header_parts)
    prefix = f"[{index}] {header}" if header else f"[{index}]"

    return f"{prefix}\n{text}"


def build_context(
    chunks: list[Union[dict, object]],
    config: Optional[ContextBuilderConfig] = None,
) -> str:
    """
    Format a list of retrieved/reranked chunks into a single context
    string, ready to embed in an LLM prompt. Returns "" for an empty
    chunk list — callers should check for this and respond
    appropriately (e.g. "no relevant information found") rather than
    sending an empty context block to the LLM.
    """
    cfg = config or ContextBuilderConfig()
    if not chunks:
        return ""

    capped = chunks[: cfg.max_chunks]
    parts = [_format_chunk(c, i, cfg) for i, c in enumerate(capped, start=1)]
    return "\n\n".join(parts)
