"""
retrieval/document_map.py -- making GENERAL questions work.

THE PROBLEM
-----------
Retrieval returns paragraphs. A specific question is well served by that.
A general question ('what is prompt engineering?') is not, for two separate
reasons:

1. NO BREADTH.
   Dense search ranks by similarity, and the five most similar paragraphs to
   a broad query are usually five NEIGHBOURING paragraphs of the same
   section. The model then answers 'prompt engineering is about few-shot
   examples' -- not a hallucination, it simply never saw the rest of the
   chapter.

2. NO STRUCTURE.
   'Few-shot is ONE OF SEVERAL strategies' is not written in any paragraph.
   It is a property of the heading tree. No amount of chunk retrieval can
   return it, because it does not live inside a chunk.

THE FIX, IN TWO PARTS
---------------------
1. diversify_by_section() caps how many chunks may come from the same
   section, so the limited prompt slots are spent on different parts of the
   document instead of on near-duplicates.

2. format_document_map() injects the heading tree itself -- 'Chapter: Prompt
   Engineering contains sections: Zero-shot; Few-shot; ...' -- so the
   structural fact becomes visible to the model.

Both are built from the chapter/section metadata added in 1.5.0, and both
fall back to parsing the old breadcrumb, so they also work on a chroma_db
that was built before any of this existed. Standard library only.

WHY A HEURISTIC FOR 'IS THIS BROAD?'
------------------------------------
Asking an LLM whether the question is broad would cost an extra model call
on every question and would make retrieval non-deterministic and hard to
debug. The heuristic here is deliberately conservative: it only fires on
recognisable overview phrasing in a SHORT question, it reports the reason it
fired (surfaced in the Query Inspector), and being wrong is cheap -- the map
is a few lines of orientation text, not a replacement for real passages.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rag_pipeline.document_reference import (
    clean_heading,
    heading_chain,
    source_display,
)


# ---------------------------------------------------------------------------
# Is this a broad / overview question?
# ---------------------------------------------------------------------------

# Overview phrasing, matched against the lowercased question. Word-boundary
# anchored so 'what is' does not fire inside 'somewhat isolated'.
_BROAD_PATTERNS_EN = [
    r"\bwhat\s+(is|are)\b",
    r"\bwhat's\b",
    r"\bdefine\b",
    r"\bdefinition\s+of\b",
    r"\bexplain\b",
    r"\boverview\b",
    r"\bintroduction\s+to\b",
    r"\bintro\s+to\b",
    r"\btell\s+me\s+about\b",
    r"\bsummar(y|ise|ize)\b",
    r"\bin\s+general\b",
    r"\bgenerally\b",
    r"\bbasics\s+of\b",
    r"\bkinds?\s+of\b",
    r"\btypes?\s+of\b",
    r"\bstrategies\b",
]

# Persian overview phrasing. No word-boundary anchors: boundaries behave
# differently for Persian script and these forms are distinctive enough.
_BROAD_PATTERNS_FA = [
    "چیست",
    "چیه",
    "چی هست",
    "توضیح بده",
    "توضیح دهید",
    "معرفی",
    "کلیات",
    "به طور کلی",
    "بطور کلی",
    "خلاصه",
    "انواع",
]

# Signals that the question is SPECIFIC even if it opens with 'what is'.
# 'What is the timeout value in step 3?' is not an overview question.
_SPECIFIC_PATTERNS = [
    r"\bstep\s*\d",
    r"\berror\b",
    r"\bexception\b",
    r"\bfailed?\b",
    r"\bhow\s+do\s+i\b",
    r"\bhow\s+to\b",
    r"\bexact\b",
    r"\bvalue\s+of\b",
    r"\bpage\s*\d",
]

# Broad questions are short by nature. A longer question carries its own
# qualifiers, so it is treated as specific.
_MAX_BROAD_WORDS = 9


@dataclass
class BreadthVerdict:
    """Why the question was (or was not) treated as broad -- shown in the UI."""
    is_broad: bool
    reason_en: str
    reason_fa: str
    matched: str = ""


def is_broad_question(question: str) -> BreadthVerdict:
    """
    Decide whether a question asks for an overview of a topic rather than one
    specific fact. Conservative on purpose -- see the module docstring.
    """
    text = (question or "").strip().lower()
    if not text:
        return BreadthVerdict(False, "Empty question.", "پرسش خالی است.")

    for pattern in _SPECIFIC_PATTERNS:
        found = re.search(pattern, text)
        if found:
            hit = found.group(0)
            return BreadthVerdict(
                False,
                "Looks like a specific question (matched '" + hit + "').",
                "پرسش جزئی به نظر می‌رسد (تطابق با '" + hit + "').",
                hit,
            )

    word_count = len(text.split())
    if word_count > _MAX_BROAD_WORDS:
        return BreadthVerdict(
            False,
            "Question is long (" + str(word_count) + " words), so it carries "
            "its own qualifiers and is treated as specific.",
            "پرسش طولانی است (" + str(word_count) + " کلمه) و جزئی در نظر گرفته می‌شود.",
        )

    for pattern in _BROAD_PATTERNS_EN:
        found = re.search(pattern, text)
        if found:
            hit = found.group(0)
            return BreadthVerdict(
                True,
                "Overview phrasing detected ('" + hit + "') in a short "
                "question, so the document map was added for orientation.",
                "عبارت کلی ('" + hit + "') در پرسشی کوتاه دیده شد؛ نقشه سند افزوده شد.",
                hit,
            )

    for phrase in _BROAD_PATTERNS_FA:
        if phrase in text:
            return BreadthVerdict(
                True,
                "Persian overview phrasing detected ('" + phrase + "').",
                "عبارت کلی فارسی ('" + phrase + "') دیده شد؛ نقشه سند افزوده شد.",
                phrase,
            )

    return BreadthVerdict(
        False,
        "No overview phrasing detected; treated as a specific question.",
        "عبارت کلی دیده نشد؛ پرسش جزئی در نظر گرفته شد.",
    )


# ---------------------------------------------------------------------------
# Diversity: stop near-duplicate paragraphs eating every prompt slot
# ---------------------------------------------------------------------------

def chunk_metadata(chunk: Any) -> dict:
    """Metadata of a chunk, whether it is a dataclass or a plain dict."""
    if isinstance(chunk, dict):
        return chunk.get("metadata", {}) or {}
    return getattr(chunk, "metadata", {}) or {}


def section_key(metadata: Optional[dict]) -> str:
    """
    Identity of the section a chunk belongs to.

    Chapter + section, NOT the full heading path: two subsections of the same
    section are near-duplicates for breadth purposes, which is exactly what
    we are trying to avoid stacking up.
    """
    metadata = metadata or {}
    chain = heading_chain(metadata)
    if chain:
        return " > ".join(chain[:2])
    return source_display(metadata) or "(unknown)"


def diversify_by_section(
    chunks: Sequence[Any],
    limit: int,
    max_per_section: Optional[int],
    get_metadata=chunk_metadata,
) -> List[Any]:
    """
    Choose which `limit` chunks reach the prompt, allowing at most
    `max_per_section` of them from any one section.

    Guarantees, all of which matter:
      - Rank order is preserved. The best chunk is still [1]; this reorders
        nothing, it only skips.
      - It never returns FEWER chunks than plain truncation would. Once every
        section has hit its cap, skipped chunks are added back in rank order
        to fill the remaining slots, so a single-section document behaves
        exactly as it did before.
      - max_per_section of 0 or None disables the whole thing.
    """
    if limit <= 0:
        return []
    chunks = list(chunks)
    if not max_per_section or max_per_section <= 0:
        return chunks[:limit]

    counts: Dict[str, int] = {}
    picked: List[Any] = []
    deferred: List[Any] = []

    for chunk in chunks:
        key = section_key(get_metadata(chunk))
        if counts.get(key, 0) < max_per_section:
            picked.append(chunk)
            counts[key] = counts.get(key, 0) + 1
            if len(picked) >= limit:
                return picked
        else:
            deferred.append(chunk)

    for chunk in deferred:
        if len(picked) >= limit:
            break
        picked.append(chunk)

    return picked


# ---------------------------------------------------------------------------
# The document map itself
# ---------------------------------------------------------------------------

@dataclass
class Outline:
    """(source file, chapter) -> ordered unique section titles."""
    chapters: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)

    def add(self, source: str, chapter: str, section: str) -> None:
        if not chapter:
            return
        sections = self.chapters.setdefault((source, chapter), [])
        if section and section not in sections:
            sections.append(section)

    def __bool__(self) -> bool:
        return bool(self.chapters)


def build_outline(metadatas: Sequence[dict]) -> Outline:
    """
    Fold a pile of chunk metadata into a chapter -> sections tree.

    Uses heading_chain(), so a corpus with no chapter/section fields still
    produces an outline from its older breadcrumb strings.
    """
    outline = Outline()
    for metadata in metadatas or []:
        if not isinstance(metadata, dict):
            continue
        chain = heading_chain(metadata)
        if not chain:
            continue
        source = source_display(metadata)
        chapter = clean_heading(chain[0])
        section = clean_heading(chain[1]) if len(chain) > 1 else ""
        outline.add(source, chapter, section)
    return outline


def _tokens(text: str) -> set:
    return {t for t in re.split(r"\W+", (text or "").lower()) if len(t) > 2}


def select_chapters(
    outline: Outline,
    question: str,
    retrieved_metadatas: Sequence[dict] = (),
    max_chapters: int = 3,
) -> List[Tuple[Tuple[str, str], List[str]]]:
    """
    Pick which chapters to show, in this priority order:

      1. Chapters the retrieved chunks actually came from. Retrieval already
         did the semantic work: if the few-shot chunk came back, the reader
         needs the Prompt Engineering outline around it.
      2. Chapters whose TITLE overlaps the question wording. Catches the case
         where the chapter's own intro paragraph was never retrieved.
    """
    if not outline:
        return []

    ordered: List[Tuple[str, str]] = []

    for metadata in retrieved_metadatas or []:
        if not isinstance(metadata, dict):
            continue
        chain = heading_chain(metadata)
        if not chain:
            continue
        key = (source_display(metadata), clean_heading(chain[0]))
        if key in outline.chapters and key not in ordered:
            ordered.append(key)

    question_tokens = _tokens(question)
    if question_tokens:
        for key in outline.chapters:
            if key in ordered:
                continue
            if _tokens(key[1]) & question_tokens:
                ordered.append(key)

    return [(key, outline.chapters[key]) for key in ordered[:max_chapters]]


def format_document_map(
    selected: Sequence[Tuple[Tuple[str, str], List[str]]],
    max_sections_per_chapter: int = 12,
) -> str:
    """
    Render the map that goes at the top of the context block.

    The wording is deliberate. The map carries no [n] numbers and explicitly
    tells the model not to cite it, because a citation pointing at the map
    would be a citation the reader cannot verify against a page -- the exact
    failure this whole reference feature exists to remove.
    """
    if not selected:
        return ""

    lines = [
        "Document map -- how the source document is organised. Orientation "
        "only: use it to explain how topics relate and what a chapter covers, "
        "but do NOT cite it and do not state facts from it beyond the "
        "structure shown. Cite only the numbered sources below."
    ]

    for (source, chapter), sections in selected:
        head = "- " + source + " > Chapter: " + chapter if source else "- Chapter: " + chapter
        shown = [s for s in sections if s][:max_sections_per_chapter]
        hidden = max(0, len([s for s in sections if s]) - len(shown))
        if shown:
            listed = "; ".join(shown)
            if hidden:
                listed += "; ... (+" + str(hidden) + " more)"
            head += "\n  Sections: " + listed
        lines.append(head)

    return "\n".join(lines)


def build_document_map_block(
    question: str,
    corpus_metadatas: Sequence[dict],
    retrieved_metadatas: Sequence[dict] = (),
    max_chapters: int = 3,
    max_sections_per_chapter: int = 12,
) -> str:
    """Convenience wrapper: metadata in, ready-to-prepend text out."""
    outline = build_outline(corpus_metadatas)
    selected = select_chapters(outline, question, retrieved_metadatas, max_chapters)
    return format_document_map(selected, max_sections_per_chapter)
