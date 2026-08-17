"""
Mapping chunks to PDF page numbers.

WHY THIS MODULE EXISTS
Headings and pages come from two different places, which is why one can be
perfect while the other is wrong:

  chapter / section -> read from the Markdown heading tree. Always exact.
  page number       -> must be recovered from the ORIGINAL PDF layout,
                       which the Markdown has already thrown away.

There are only two honest ways to recover a page number:

  1. INLINE MARKERS (exact). The OCR tool left page boundaries in the
     Markdown, e.g. a <span id="page-4-0"></span> tag. If those exist we
     use them and the page is ground truth.

  2. PROPORTIONAL ESTIMATE (approximate). No markers, but a sidecar JSON
     says how much text each page holds. We map 'how far through the
     document this chunk sits' onto 'which page covers that position'.

THE BUG THIS REPLACES
The old fallback compared the pipeline's own chunk_index against the
sidecar's per-page num_chunks totals. Those count different things: the
sidecar counts the OCR tool's chunks (49 for a 6-page test document) while
the pipeline produces its own semantic chunks (11 for the same file).
Comparing them squeezed every chunk into the first pages, so that document
could only ever cite pages 1-2 and pages 3-6 were unreachable regardless
of the question.

The fix: stop counting chunks, measure POSITION. A chunk starting 70% of
the way through the text belongs on whichever page covers the 70% mark,
weighted by how many tokens each page really holds. That survives the
pipeline splitting text differently from the OCR tool, which is the point.

HONESTY RULE
An estimate is never presented as ground truth. Each assignment carries its
source ('markers' | 'estimated' | 'none') so a citation can render an
approximate page as '~page 4' rather than pretending to know it exactly.
A confidently wrong page is worse than an obviously fuzzy one, because the
reader cannot check it without re-reading the whole PDF.

Pure arithmetic: no regexes, no I/O, no dependencies. The marker patterns
themselves stay in metadata_enricher.py.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

# Sidecar schemas differ per OCR tool, so every lookup tries several names.
_PAGE_LIST_KEYS = ("pages", "page_stats", "elements")
_PAGE_NUM_KEYS = ("page_num", "page_number", "page", "index", "page_idx")
# Preferred weights: how much TEXT a page holds, since text position is
# the only thing we can actually measure in the Markdown.
_WEIGHT_KEYS = ("token_count", "tokens", "num_tokens", "char_count",
                "text_length", "length")
# Weak fallback: element counts. Correlates with text volume only loosely
# (20 tiny list items vs 3 dense paragraphs), so used only as a last resort.
_COUNT_KEYS = ("num_chunks", "chunk_count", "num_elements", "element_count")
_TOTAL_PAGE_KEYS = ("num_pages", "total_pages", "page_count", "pages_count")


@dataclass
class PageIndex:
    """
    Where each page sits along the document, as fractions of total text.

    cumulative[i] is the fraction of the document ending with page
    page_numbers[i], so page i spans (cumulative[i-1], cumulative[i]].
    Page numbers are 1-indexed for humans.
    """
    page_numbers: list = field(default_factory=list)
    cumulative: list = field(default_factory=list)
    weight_source: str = "uniform"   # tokens | counts | uniform

    @property
    def total_pages(self) -> int:
        return len(self.page_numbers)

    def page_span(self, position: int) -> tuple:
        """The (start_fraction, end_fraction) covered by page at position."""
        start = self.cumulative[position - 1] if position > 0 else 0.0
        return start, self.cumulative[position]


def _number(value) -> Optional[float]:
    """Coerce a sidecar value to a number, or None. Sidecars contain junk."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _first_number(entry: dict, keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key in entry:
            found = _number(entry.get(key))
            if found is not None:
                return found
    return None


def _page_entries(page_meta: dict) -> list:
    """The per-page list from whichever key this OCR tool used."""
    for key in _PAGE_LIST_KEYS:
        value = (page_meta or {}).get(key)
        if isinstance(value, list) and value:
            entries = [item for item in value if isinstance(item, dict)]
            if entries:
                return entries
    return []


def build_page_index(page_meta: Optional[dict]) -> Optional[PageIndex]:
    """
    Build a PageIndex from a sidecar dict, or None if it says nothing usable.

    Degrades in three steps, best first:
      1. per-page token counts   -> weight_source 'tokens'
      2. per-page element counts -> weight_source 'counts'
      3. only a total page count -> weight_source 'uniform' (equal pages)
    """
    page_meta = page_meta or {}
    entries = _page_entries(page_meta)

    page_numbers: list = []
    weights: list = []
    weight_source = "uniform"

    if entries:
        raw_numbers = []
        for position, entry in enumerate(entries):
            number = _first_number(entry, _PAGE_NUM_KEYS)
            raw_numbers.append(int(number) if number is not None else position)

        # Some tools number pages from 0, others from 1. Shift only when a
        # real 0 is present, so a document that starts at page 1 is intact.
        shift = 1 if (raw_numbers and min(raw_numbers) == 0) else 0
        page_numbers = [number + shift for number in raw_numbers]

        token_weights = [_first_number(entry, _WEIGHT_KEYS) or 0.0 for entry in entries]
        if sum(token_weights) > 0:
            weights, weight_source = token_weights, "tokens"
        else:
            count_weights = [_first_number(entry, _COUNT_KEYS) or 0.0 for entry in entries]
            if sum(count_weights) > 0:
                weights, weight_source = count_weights, "counts"
            else:
                weights, weight_source = [1.0] * len(entries), "uniform"
    else:
        total = None
        for key in _TOTAL_PAGE_KEYS:
            total = _number(page_meta.get(key))
            if total:
                break
        if not total or total < 1:
            return None
        page_numbers = list(range(1, int(total) + 1))
        weights = [1.0] * len(page_numbers)
        weight_source = "uniform"

    if not page_numbers:
        return None

    # A page recorded as empty would occupy zero width and become impossible
    # to cite. Give every page a floor so it stays reachable.
    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1.0] * len(page_numbers)
        total_weight = float(len(page_numbers))
    floor = total_weight * 0.001
    weights = [max(weight, floor) for weight in weights]
    total_weight = sum(weights)

    cumulative = []
    running = 0.0
    for weight in weights:
        running += weight
        cumulative.append(running / total_weight)
    cumulative[-1] = 1.0  # guard against float drift

    return PageIndex(
        page_numbers=page_numbers,
        cumulative=cumulative,
        weight_source=weight_source,
    )


def pages_for_span(
    index: PageIndex,
    start_fraction: float,
    end_fraction: float,
) -> list:
    """
    Every page overlapped by the span [start_fraction, end_fraction].

    A chunk straddling a page break genuinely belongs to both pages, so this
    can return more than one. The citation then reads 'pages 4-5', which is
    honest, instead of silently picking a side.
    """
    if not index or not index.page_numbers:
        return []

    start = max(0.0, min(1.0, float(start_fraction)))
    end = max(0.0, min(1.0, float(end_fraction)))
    if end < start:
        start, end = end, start

    hits: list = []
    for position, page_number in enumerate(index.page_numbers):
        page_start, page_end = index.page_span(position)
        if start == end:
            # Zero-width span (e.g. an empty chunk): treat it as a point.
            inside = page_start <= start <= page_end
        else:
            inside = (start < page_end) and (end > page_start)
        if inside:
            hits.append(page_number)

    if hits:
        return hits

    # Float edge cases only: fall back to the page holding the midpoint.
    midpoint = (start + end) / 2.0
    for position, page_number in enumerate(index.page_numbers):
        _, page_end = index.page_span(position)
        if midpoint <= page_end:
            return [page_number]
    return [index.page_numbers[-1]]


def assign_pages_by_length(lengths: Sequence[int], index: PageIndex) -> list:
    """
    Estimate pages for chunks given in document order, using each chunk's
    text length to place it along the document.

    Length rather than chunk position, because chunk sizes vary enormously
    (a one-line heading chunk vs a full table). Weighting by length stops a
    long chunk from counting as the same 'distance' as a short one.
    """
    lengths = [max(0, int(length or 0)) for length in lengths]
    if not lengths or not index or not index.page_numbers:
        return [[] for _ in lengths]

    total = sum(lengths)
    if total <= 0:
        # No text at all: place evenly by position so the result is still
        # monotonic rather than empty.
        count = len(lengths)
        return [
            pages_for_span(index, position / count, (position + 1) / count)
            for position in range(count)
        ]

    assignments: list = []
    running = 0
    for length in lengths:
        start = running / total
        running += length
        end = running / total
        assignments.append(pages_for_span(index, start, end))
    return assignments


@dataclass
class MarkerInfo:
    """
    What the page markers inside one chunk tell us.

    starts_with_marker: the chunk opens with a marker, so none of its text
        belongs to the previous page.
    ends_with_marker: the chunk ends with a marker and no text follows it,
        so that marker announces the NEXT chunk's page, not this one's.
    """
    pages: list = field(default_factory=list)
    starts_with_marker: bool = False
    ends_with_marker: bool = False


def resolve_marker_pages(markers: Sequence[MarkerInfo]) -> list:
    """
    Turn per-chunk marker findings into per-chunk page lists, carrying the
    current page forward across chunks.

    This is the second half of the reported bug. Markers are BOUNDARIES, so
    reading each chunk in isolation gets two cases wrong:

      * A chunk with no marker inside it got NO page at all, even though it
        clearly sits on whatever page was open when it started.
      * Text before the first marker in a chunk belongs to the PREVIOUS
        page, but was attributed to the page that marker announces.

    A running 'current page' cursor fixes both.
    """
    resolved: list = []
    current: Optional[int] = None

    for info in markers:
        pages = sorted({int(page) for page in (info.pages or [])})

        if not pages:
            # No boundary inside this chunk: it lies wholly on the open page.
            resolved.append([current] if current is not None else [])
            continue

        chunk_pages = list(pages)

        # A trailing marker with no text after it belongs to the next chunk.
        if info.ends_with_marker and len(chunk_pages) > 1:
            chunk_pages = chunk_pages[:-1]

        # Text before the first marker continues the previous page.
        if not info.starts_with_marker and current is not None:
            chunk_pages = [current] + chunk_pages

        resolved.append(sorted(set(chunk_pages)))
        current = pages[-1]

    return resolved
