"""
document_reference.py

ONE place that turns chunk metadata into a human document reference like:

    Binder2.pdf, page 12 - Chapter: Prompt Engineering > Section: One-shot

Why this module exists
----------------------
Before this, the document location of a chunk was rebuilt independently in
five places (context_builder, rag_answerer, run_query, run_answer, the web
UI), each with slightly different formatting. That meant the reference the
LLM saw, the reference printed in the CLI and the reference shown in the
browser could disagree with each other. When a user asks "which page is
this from?", three different answers is worse than none.

So: every component now imports from here.

What a reference is made of
---------------------------
1. WHERE the file is        -> source_pdf (falls back to source_file)
2. WHICH page               -> page_numbers, from the OCR page markers
                               (metadata_enricher._extract_and_strip_pages)
3. WHERE in the structure   -> the markdown heading chain, split into
                               Chapter / Section / Subsection

How the heading chain becomes Chapter/Section/Subsection
--------------------------------------------------------
The splitter records the ancestor heading chain for every chunk, e.g.

    header_texts  = ["Prompt Engineering", "Few-shot", "Worked example"]
    header_levels = [1, 2, 3]

Labels are assigned by POSITION IN THE CHAIN, not by the absolute number of
'#' characters. This is deliberate: plenty of OCR'd documents start at '##'
because '#' was used for the book title, or skip from '#' straight to '###'.
Using position means the top-most heading a chunk lives under is always its
"Chapter", whatever depth the source markdown happened to use.

Backward compatibility (important)
----------------------------------
Chunks that were ingested BEFORE this module existed only carry the flat
"breadcrumb" string ("# A > ## B"). Every function here falls back to
parsing that string, so references work on an existing chroma_db WITHOUT
re-running ingestion. Re-ingesting simply makes the fields explicit.

ChromaDB also flattens list metadata into comma-separated strings, so each
reader here accepts both the list form and the string form.
"""

import re
from typing import Any, Optional

# Position-based labels. Index 0 = outermost heading a chunk lives under.
# Anything deeper than the list re-uses the last label.
LEVEL_LABELS_EN: tuple[str, ...] = ("Chapter", "Section", "Subsection")

# Persian equivalents, used by the web UI and the bilingual reports.
# فصل = chapter, بخش = section, زیربخش = subsection.
LEVEL_LABELS_FA: tuple[str, ...] = ("\u0641\u0635\u0644", "\u0628\u062e\u0634", "\u0632\u06cc\u0631\u0628\u062e\u0634")

_RE_LEADING_HASHES = re.compile(r"^#+\s*")
_ROOT_MARKERS = {"", "(root)", "(no heading)"}


def clean_heading(text: str) -> str:
    """
    '### System reports' -> 'System reports'.

    The outer strip() comes FIRST on purpose: breadcrumb strings are split
    on ' > ', so segments arrive as ' ## Few-shot' with a leading space and
    an anchored ^#+ pattern would not match.
    """
    if not text:
        return ""
    return _RE_LEADING_HASHES.sub("", str(text).strip()).strip()


def _as_list(value: Any) -> list[str]:
    """
    Accept the three shapes metadata can arrive in:
      - a real list (straight from the pipeline)
      - a comma-separated string (after ChromaDB metadata sanitization)
      - None / "" (field absent)
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def heading_chain(metadata: dict) -> list[str]:
    """
    The chunk's ancestor headings, outermost first, cleaned of '#' marks.

    Preference order:
      1. header_texts      - explicit, set by the splitter
      2. heading_path      - explicit, set by metadata_enricher
      3. breadcrumb        - the legacy flat string, parsed as a fallback
         so already-ingested corpora keep working.
    """
    metadata = metadata or {}

    explicit = _as_list(metadata.get("header_texts"))
    if explicit:
        return [clean_heading(h) for h in explicit]

    path = str(metadata.get("heading_path") or "").strip()
    if path and path not in _ROOT_MARKERS:
        return [clean_heading(p) for p in path.split(">") if clean_heading(p)]

    breadcrumb = str(metadata.get("breadcrumb") or "").strip()
    if breadcrumb and breadcrumb not in _ROOT_MARKERS:
        return [clean_heading(p) for p in breadcrumb.split(">") if clean_heading(p)]

    return []


def label_for_depth(index: int, persian: bool = False) -> str:
    """Chapter / Section / Subsection for chain position `index`."""
    labels = LEVEL_LABELS_FA if persian else LEVEL_LABELS_EN
    if index < 0:
        index = 0
    return labels[index] if index < len(labels) else labels[-1]


def heading_levels(metadata: dict) -> list[dict]:
    """
    The heading chain as labelled levels:

        [{"index": 0, "label": "Chapter", "label_fa": "فصل",
          "title": "Prompt Engineering", "markdown_level": 1}, ...]

    markdown_level is the original '#' depth when known (it is only useful
    for display/debugging - labelling never depends on it).
    """
    chain = heading_chain(metadata)
    raw_levels = _as_list((metadata or {}).get("header_levels"))

    levels = []
    for index, title in enumerate(chain):
        markdown_level: Optional[int] = None
        if index < len(raw_levels):
            try:
                markdown_level = int(raw_levels[index])
            except (TypeError, ValueError):
                markdown_level = None
        levels.append({
            "index": index,
            "label": label_for_depth(index),
            "label_fa": label_for_depth(index, persian=True),
            "title": title,
            "markdown_level": markdown_level,
        })
    return levels


def pages_display(metadata: dict) -> str:
    """
    Page numbers as a compact human string. Consecutive runs collapse:
    [4] -> "4", [4, 5, 6] -> "4-6", [4, 9] -> "4, 9". Empty when the OCR
    output carried no page markers.
    """
    raw = _as_list((metadata or {}).get("page_numbers"))
    numbers = []
    for item in raw:
        try:
            numbers.append(int(item))
        except (TypeError, ValueError):
            # A non-numeric page label (e.g. roman numerals) - keep as-is.
            return ", ".join(raw)
    if not numbers:
        return ""

    numbers = sorted(set(numbers))
    runs: list[list[int]] = [[numbers[0]]]
    for number in numbers[1:]:
        if number == runs[-1][-1] + 1:
            runs[-1].append(number)
        else:
            runs.append([number])

    parts = []
    for run in runs:
        parts.append(str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}")
    return ", ".join(parts)


def page_is_estimated(metadata: dict) -> bool:
    """
    True when the page number was inferred from document position rather
    than read from a real page marker (see enrichers/page_mapper.py).

    Callers use this to render "~page 4" instead of "page 4". A page the
    reader cannot verify must not look identical to one they can: an
    approximate page shown as exact is the kind of error that quietly
    destroys trust in every other citation on the screen.
    """
    return str((metadata or {}).get("page_source") or "").strip() == "estimated"


def source_display(metadata: dict) -> str:
    """The original document name, preferring the real PDF filename."""
    metadata = metadata or {}
    source = metadata.get("source_pdf") or metadata.get("source_file") or ""
    source = str(source).strip()
    if not source:
        return ""
    # source_file is a path; only the filename is useful in a citation.
    return source.replace("\\", "/").split("/")[-1]


def structured_reference(metadata: dict) -> dict:
    """
    Everything a caller needs to display or verify a citation, already
    split into named parts so no consumer has to parse strings:

        {
          "source": "Binder2.pdf",
          "pages": "12",
          "chapter": "Prompt Engineering",
          "section": "Few-shot",
          "subsection": "Worked example",
          "levels": [ ...heading_levels()... ],
          "heading_path": "Prompt Engineering > Few-shot > Worked example",
          "text": "Binder2.pdf, page 12 - Chapter: ... > Section: ...",
          "short": "Binder2.pdf p.12 - Few-shot",
        }

    Missing pieces are simply empty strings - a chunk with no page markers
    and no headings still produces a valid (source-only) reference.
    """
    metadata = metadata or {}
    levels = heading_levels(metadata)
    titles = [level["title"] for level in levels]

    return {
        "source": source_display(metadata),
        "pages": pages_display(metadata),
        # How the page was obtained: "markers" (exact) | "estimated"
        # (inferred from position) | "none". Consumers that show a page
        # should also show this, or show nothing at all.
        "page_source": str(metadata.get("page_source") or "none"),
        "pages_estimated": page_is_estimated(metadata),
        "chapter": titles[0] if len(titles) > 0 else "",
        "section": titles[1] if len(titles) > 1 else "",
        "subsection": titles[2] if len(titles) > 2 else "",
        "levels": levels,
        "heading_path": " > ".join(titles),
        "text": format_reference(metadata),
        "short": format_reference(metadata, style="short"),
    }


def format_reference(
    metadata: dict,
    *,
    style: str = "full",
    include_source: bool = True,
    include_pages: bool = True,
    include_headings: bool = True,
) -> str:
    """
    Render the reference as one line.

    style="full"  -> Binder2.pdf, page 12 - Chapter: Prompting > Section: Few-shot
    style="short" -> Binder2.pdf p.12 - Few-shot

    "full" is what goes into the LLM context: the explicit
    'Chapter:'/'Section:' words are what let the model reproduce the
    location in its answer instead of inventing one. "short" is for
    cramped UI spots like source chips.
    """
    metadata = metadata or {}
    source = source_display(metadata) if include_source else ""
    pages = pages_display(metadata) if include_pages else ""
    levels = heading_levels(metadata) if include_headings else []

    # "~" marks a page inferred from position rather than read from a marker.
    approx = "~" if (pages and page_is_estimated(metadata)) else ""

    if style == "short":
        head = source
        if pages:
            head = f"{head} {approx}p.{pages}" if head else f"{approx}p.{pages}"
        deepest = levels[-1]["title"] if levels else ""
        if deepest:
            return f"{head} - {deepest}" if head else deepest
        return head

    left = source
    if pages:
        page_word = "pages" if ("-" in pages or "," in pages) else "page"
        rendered = f"{approx}{page_word} {pages}"
        left = f"{left}, {rendered}" if left else rendered

    right = " > ".join(f"{level['label']}: {level['title']}" for level in levels)

    if left and right:
        return f"{left} - {right}"
    return left or right


def reference_fields(metadata: dict) -> dict:
    """
    The flat key/value pairs to store ON a chunk's metadata at ingestion
    time. Kept flat and scalar because ChromaDB only accepts str/int/
    float/bool metadata values.

    Storing these is not strictly required - every function here can
    derive them from the breadcrumb on the fly - but doing it once at
    ingestion means retrieval-time filtering ("only chapter X") and
    sorting become possible later.
    """
    reference = structured_reference(metadata)
    return {
        "chapter": reference["chapter"],
        "section": reference["section"],
        "subsection": reference["subsection"],
        "heading_path": reference["heading_path"],
        "heading_depth": len(reference["levels"]),
        "page_display": reference["pages"],
        "page_source": reference["page_source"],
        "reference": reference["text"],
    }
