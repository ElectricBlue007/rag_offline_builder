"""
normalizers/table_normalizer.py

Converts HTML table chunks into natural-language prose for embedding,
WITHOUT discarding the original HTML (which stays in `content` for
display/rendering).

Why this exists
----------------
Embedding models tokenize raw HTML as syntax noise, not semantics.
"<table><tr><td>Create SOI</td><td>CMI track</td>..." loses the column
relationship that makes the row meaningful ("Create SOI" is the ACTION,
"CMI track" is its LOGICAL ENTRY). Converting each row to a sentence
that names the relationship ("Create SOI: logical entry is CMI track...")
lets the embedding model actually capture what the table says.

Two real table shapes observed in Chandra OCR output
------------------------------------------------------
1. HEADER-ROW tables (Binder2.md pattern):
   <thead><tr><th>Operator action</th><th>Logical entry</th>...</thead>
   <tbody><tr><td>Create SOI</td><td>CMI track</td>...</tbody>
   → one row = one record. Normalize each row using its column headers
     as field names: "Create SOI — logical entry: CMI track; ..."

2. KEY-VALUE tables (Testing.md pattern — prompt/config examples):
   No <thead>/<th> at all. Each row is <td><b>Label</b></td><td>Value</td>,
   sometimes two label:value pairs per row (Temperature/Token Limit),
   sometimes one value spanning via colspan (Prompt/Goal/Model). The
   WHOLE table describes a single record/example, not one row = one item.
   → Normalize the entire table as ONE flat "Label: Value; Label: Value"
     paragraph, since row boundaries don't carry independent meaning here.

3. FALLBACK — anything that matches neither shape: extract cell text
   row by row, best-effort, so nothing is silently dropped.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False


# Values that mean "not applicable / nothing here" in the OCR'd tables.
# Deliberately narrow — "N/A" is a meaningful config value (e.g. Top-K: N/A)
# and must NOT be dropped, only a bare "-" placeholder is noise.
_EMPTY_MARKERS = {"-", "—", "–", ""}


@dataclass
class TableCell:
    text: str
    colspan: int = 1
    is_bold_only: bool = False


@dataclass
class ParsedRow:
    cells: list[TableCell] = field(default_factory=list)


@dataclass
class ParsedTable:
    headers: list[str] = field(default_factory=list)
    rows: list[ParsedRow] = field(default_factory=list)
    table_type: str = "unknown"  # "header_row" | "key_value" | "fallback"


class TableNormalizationError(Exception):
    """Raised when a table cannot be parsed at all (malformed HTML)."""
    pass


# Decorative boilerplate that sometimes appears INSIDE a table cell's own
# HTML (e.g. <p style="text-align: right;"><b>Continues next page...</b></p>)
# — the preprocessor's line-based stripper only catches this when it sits
# on its own markdown line, so it must be caught again here at the cell
# level after the cell's HTML has been flattened to text.
_RE_CELL_DECORATIVE = re.compile(r'\bContinues next page\.*\s*', re.IGNORECASE)


def _clean_cell_text(cell) -> str:
    """
    Extract readable text from a <td>/<th> BeautifulSoup tag.
    - <br/> becomes "; " (a clause separator, not silently dropped)
    - <p>, <pre>, and other block tags are joined with spaces
    - whitespace is collapsed
    - decorative boilerplate ("Continues next page...") is stripped
    """
    # Replace <br> tags with a text separator before extracting text,
    # otherwise BeautifulSoup just concatenates the two sides with no space.
    for br in cell.find_all("br"):
        br.replace_with("; ")

    text = cell.get_text(separator=" ", strip=True)
    text = _RE_CELL_DECORATIVE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Clean up cases where a "; " ended up right before a period/comma,
    # or has a stray space before it (from "<br/>" sitting after a space),
    # or got duplicated due to <br/> replacement at a boundary.
    text = re.sub(r"\s+;", ";", text)
    text = re.sub(r";\s*;+", ";", text)
    text = re.sub(r"\s*;\s*$", "", text)
    return text.strip()


def _is_bold_only(cell) -> bool:
    """
    True if the cell's entire content is a single <b> (or <strong>) tag
    with no other meaningful text — the signature of a key-value table's
    label cell, e.g. <td><b>Temperature</b></td>.
    """
    bold_tags = cell.find_all(["b", "strong"])
    if len(bold_tags) != 1:
        return False
    full_text = cell.get_text(strip=True)
    bold_text = bold_tags[0].get_text(strip=True)
    return bool(full_text) and full_text == bold_text


def parse_table_html(html: str) -> ParsedTable:
    """
    Parse a single <table>...</table> HTML string into a ParsedTable,
    detecting whether it's a header-row table, a key-value table, or
    an unrecognized shape (fallback).
    """
    if not _HAS_BS4:
        raise TableNormalizationError(
            "beautifulsoup4 is required for table normalization. "
            "Run: pip install beautifulsoup4 lxml"
        )

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    table = soup.find("table")
    if table is None:
        raise TableNormalizationError("No <table> element found in input HTML.")

    all_rows = table.find_all("tr")
    if not all_rows:
        raise TableNormalizationError("Table has no <tr> rows.")

    # ── Detect header-row shape: any <th> anywhere ──────────────────────────
    has_th = table.find("th") is not None

    if has_th:
        header_row = None
        data_rows = []
        for tr in all_rows:
            ths = tr.find_all("th")
            if ths and header_row is None:
                header_row = tr
            else:
                data_rows.append(tr)

        headers = [_clean_cell_text(th) for th in header_row.find_all("th")] if header_row else []
        rows = []
        for tr in data_rows:
            cells = []
            for td in tr.find_all(["td", "th"]):
                colspan = int(td.get("colspan", 1) or 1)
                cells.append(TableCell(
                    text=_clean_cell_text(td),
                    colspan=colspan,
                    is_bold_only=_is_bold_only(td),
                ))
            if cells:
                rows.append(ParsedRow(cells=cells))

        return ParsedTable(headers=headers, rows=rows, table_type="header_row")

    # ── Detect key-value shape: first row's first cell is bold-only ────────
    first_row_cells = all_rows[0].find_all(["td", "th"])
    looks_key_value = bool(first_row_cells) and _is_bold_only(first_row_cells[0])

    if looks_key_value:
        rows = []
        for tr in all_rows:
            cells = []
            for td in tr.find_all(["td", "th"]):
                colspan = int(td.get("colspan", 1) or 1)
                cells.append(TableCell(
                    text=_clean_cell_text(td),
                    colspan=colspan,
                    is_bold_only=_is_bold_only(td),
                ))
            if cells:
                rows.append(ParsedRow(cells=cells))
        return ParsedTable(headers=[], rows=rows, table_type="key_value")

    # ── Fallback: unrecognized shape, extract everything best-effort ───────
    rows = []
    for tr in all_rows:
        cells = []
        for td in tr.find_all(["td", "th"]):
            colspan = int(td.get("colspan", 1) or 1)
            cells.append(TableCell(text=_clean_cell_text(td), colspan=colspan))
        if cells:
            rows.append(ParsedRow(cells=cells))
    return ParsedTable(headers=[], rows=rows, table_type="fallback")


def _normalize_header_row_table(parsed: ParsedTable) -> str:
    """
    One sentence per data row, using column headers as field names.
    The first column's value is treated as the row's subject.

    "Create SOI — logical entry: CMI track; real entry: Enter "NT"...;
     related departure: CMI; CMI Report; restrictions: Only for lost
     surface point tracks and reference positions Maximum number of CMI =10."
    """
    sentences = []
    headers = parsed.headers

    for row in parsed.rows:
        if not row.cells:
            continue

        subject = row.cells[0].text
        clauses = []

        for i in range(1, len(row.cells)):
            value = row.cells[i].text
            if value.strip() in _EMPTY_MARKERS:
                continue
            header_name = headers[i].lower() if i < len(headers) else f"field {i}"
            clauses.append(f"{header_name}: {value}")

        if clauses:
            sentence = f"{subject} — " + "; ".join(clauses) + "."
        else:
            sentence = f"{subject}."

        sentences.append(sentence)

    return "\n".join(sentences)


def _normalize_key_value_table(parsed: ParsedTable) -> str:
    """
    The whole table is one record — flatten every label:value pair found
    across all rows into a single paragraph. A row may contain one pair
    (label + colspan value) or two pairs (label, value, label2, value2).
    """
    clauses = []

    for row in parsed.rows:
        cells = row.cells
        i = 0
        while i < len(cells):
            if cells[i].is_bold_only:
                label = cells[i].text
                # Absorb all following non-bold-only cells as this label's
                # value, until the next bold-only (label) cell or row end.
                value_parts = []
                j = i + 1
                while j < len(cells) and not cells[j].is_bold_only:
                    value_parts.append(cells[j].text)
                    j += 1
                value = " ".join(v for v in value_parts if v).strip()
                if value:
                    clauses.append(f"{label}: {value}")
                elif label:
                    clauses.append(f"{label}: (no value)")
                i = j
            else:
                # Stray non-labeled cell with no preceding bold label —
                # keep its text so nothing is silently dropped.
                if cells[i].text:
                    clauses.append(cells[i].text)
                i += 1

    return "; ".join(clauses)


def _normalize_fallback_table(parsed: ParsedTable) -> str:
    """Best-effort: one line per row, cells joined with '; '."""
    lines = []
    for row in parsed.rows:
        texts = [c.text for c in row.cells if c.text]
        if texts:
            lines.append("; ".join(texts))
    return "\n".join(lines)


def normalize_table_html(html: str) -> str:
    """
    Main entry point. Convert one <table>...</table> HTML chunk into a
    natural-language string suitable for embedding.

    Returns an empty string (never raises) if the table is empty after
    parsing — callers should fall back to the raw HTML in that case
    rather than embedding nothing.
    """
    parsed = parse_table_html(html)

    if parsed.table_type == "header_row":
        return _normalize_header_row_table(parsed)
    elif parsed.table_type == "key_value":
        return _normalize_key_value_table(parsed)
    else:
        return _normalize_fallback_table(parsed)
