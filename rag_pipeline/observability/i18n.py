"""
Tiny bilingual (English / فارسی) Markdown helpers for the reports.

Why a module for this: every sentence in a report exists twice, and doing that
by hand inside the report builders would double their size and guarantee the
two languages drift apart. Here, a bilingual string is just a 2-tuple
`(english, farsi)` -- called `Bi` below -- and these functions decide how to
render it.

Stdlib only. No dependencies, no I/O, no state.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple

# A bilingual string: (English, فارسی)
Bi = Tuple[str, str]

# Render modes.
EN = "en"
FA = "fa"
BOTH = "both"


def esc(value: Any) -> str:
    """Make a value safe to put inside a Markdown table cell."""
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\r", "").replace("\n", " ")


def rtl(text: str) -> str:
    """
    Wrap Persian text so it renders right-to-left in Notion, GitHub and most
    Markdown viewers. Plain Markdown has no direction marker, so a small HTML
    wrapper is the portable way to do this.
    """
    text = (text or "").strip()
    return f'<div dir="rtl">{text}</div>' if text else ""


def _pick(bi: Bi, lang: str) -> List[str]:
    en, fa = (bi[0], bi[1]) if isinstance(bi, tuple) and len(bi) == 2 else (str(bi), str(bi))
    if lang == EN:
        return [en]
    if lang == FA:
        return [rtl(fa)]
    return [en, rtl(fa)]


def para(bi: Bi, lang: str = BOTH) -> str:
    """A paragraph, in one or both languages, followed by a blank line."""
    parts = [p for p in _pick(bi, lang) if p]
    return "\n\n".join(parts) + "\n"


def bullet(bi: Bi, lang: str = BOTH) -> str:
    """One bullet point. The Persian half becomes an indented sub-bullet."""
    if lang == EN:
        return f"- {bi[0]}"
    if lang == FA:
        return f"- {rtl(bi[1])}"
    return f"- {bi[0]}\n  - {rtl(bi[1])}"


def heading(bi: Bi, level: int = 2, lang: str = BOTH) -> str:
    """A heading. In `both` mode the two languages share one line."""
    hashes = "#" * max(1, min(6, level))
    if lang == EN:
        title = bi[0]
    elif lang == FA:
        title = bi[1]
    else:
        title = f"{bi[0]} — {bi[1]}"
    return f"\n{hashes} {title}\n"


def table(headers: Sequence[Any], rows: Iterable[Sequence[Any]]) -> str:
    """A GitHub-style Markdown table. Cells are escaped and flattened."""
    head = "| " + " | ".join(esc(h) for h in headers) + " |"
    rule = "|" + "|".join(["---"] * len(headers)) + "|"
    body = ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    return "\n".join([head, rule] + body) + "\n"


def side_by_side(rows: Sequence[Bi], headers: Bi = ("English", "فارسی"),
                 lang: str = BOTH) -> str:
    """
    Two-column table with English on the left and Persian on the right --
    used for lists of explanations where a reader wants to compare wording.
    """
    if lang == EN:
        return "\n".join(f"- {r[0]}" for r in rows) + "\n"
    if lang == FA:
        return "\n".join(f"- {rtl(r[1])}" for r in rows) + "\n"
    return table([headers[0], headers[1]], [[r[0], rtl(r[1])] for r in rows])


def fence(text: str, info: str = "text", max_chars: int = 1200) -> str:
    """
    A fenced code block, used to show real excerpts of the user's data.

    Two safety measures: the excerpt is truncated with a visible marker, and
    any ``` inside the data is neutralised so it cannot break out of the fence
    and corrupt the rest of the report.
    """
    body = "" if text is None else str(text)
    if len(body) > max_chars:
        cut = len(body) - max_chars
        body = body[:max_chars] + f"\n… [truncated: {cut} more character(s)]"
    body = body.replace("```", "'''")
    return f"```{info}\n{body}\n```\n"


def pct(part: float, whole: float) -> str:
    """'12.5%' -- or 'n/a' when there is nothing to divide by."""
    if not whole:
        return "n/a"
    return f"{(100.0 * part / whole):.1f}%"


def human_seconds(seconds: Optional[float]) -> str:
    """Durations a person can read: 45ms, 1.2s, 3m 04s."""
    if seconds is None:
        return "—"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}m {rest:04.1f}s"
