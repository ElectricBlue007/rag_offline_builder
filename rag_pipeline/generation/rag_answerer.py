"""
generation/rag_answerer.py

Ties retrieval + context assembly + LLM generation into a single
"ask a question, get an answer" call — the final piece that turns
this pipeline into an actual question-answering system rather than
just a search tool.

Dependency-injected (retriever, llm_client, reranker) rather than
constructing its own components — this keeps RagAnswerer decoupled
from exactly which retriever/LLM implementation is behind it, and
makes it trivial to test with fakes.
"""

from dataclasses import dataclass, field
from typing import Optional, Protocol

from rag_pipeline.config import ContextBuilderConfig, RuntimeConfig
from rag_pipeline import model_manager
from rag_pipeline.document_reference import structured_reference
from rag_pipeline.generation.context_builder import build_context
from rag_pipeline.retrieval.document_map import (
    build_document_map_block,
    chunk_metadata,
    diversify_by_section,
    is_broad_question,
)


class _Retriever(Protocol):
    def retrieve(self, query: str) -> list: ...


class _Reranker(Protocol):
    def rerank(self, query: str, candidates: list) -> list: ...


class _LlmClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


@dataclass
class RagAnswer:
    answer: str
    sources: list[dict] = field(default_factory=list)

    # Breadth diagnostics for general questions: the document-map text that
    # was prepended (empty string when none was) and why that decision was
    # made. Surfaced in the Query Inspector so a reader can see WHY a broad
    # question got an outline and a specific one did not.
    document_map: str = ""
    breadth: dict = field(default_factory=dict)

    # The exact context block sent to the LLM — kept for debugging and
    # for the future RAGAS evaluation step (roadmap step 9), which
    # needs to inspect what context an answer was actually grounded in.
    context_used: str = ""


_NO_RESULTS_MESSAGE = "I couldn't find any relevant information to answer that question."


class RagAnswerer:
    """
    Usage:
        answerer = RagAnswerer(retriever, llm_client, reranker=reranker)
        result = answerer.answer("What is a CMI?")
        print(result.answer)
    """

    def __init__(
        self,
        retriever: _Retriever,
        llm_client: _LlmClient,
        reranker: Optional[_Reranker] = None,
        context_config: Optional[ContextBuilderConfig] = None,
        runtime: Optional[RuntimeConfig] = None,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.reranker = reranker
        self.context_cfg = context_config or ContextBuilderConfig()
        self.runtime = runtime or RuntimeConfig()

        # Corpus outline source, read at most once per process (see
        # _corpus_metadatas). None means "not looked up yet"; an empty list
        # means "looked up, nothing available" — the difference matters, or
        # a corpus with no headings would be re-read on every question.
        self._corpus_metadatas_cache: Optional[list] = None

    def answer(self, query: str) -> RagAnswer:
        if not query or not query.strip():
            return RagAnswer(answer=_NO_RESULTS_MESSAGE)

        chunks = self.retriever.retrieve(query)
        if self.reranker is not None:
            chunks = self.reranker.rerank(query, chunks)

        if not chunks:
            return RagAnswer(answer=_NO_RESULTS_MESSAGE)

        # FIX(B3): build_context() only puts the first max_chunks chunks in
        # the prompt, so the model can only ever emit citations [1..max_chunks].
        # Reporting sources for chunks beyond that made callers display
        # sources the LLM never actually saw. Context and the source list now
        # come from the exact same slice.
        # BREADTH: choose WHICH chunks fill the max_chunks slots instead of
        # blindly taking the top N. For a broad question the top N are
        # usually neighbouring paragraphs of one section, which makes the
        # model describe that section as if it were the whole topic.
        # diversify_by_section never sends fewer chunks than truncation did
        # — see retrieval/document_map.py.
        used_chunks = diversify_by_section(
            chunks,
            self.context_cfg.max_chunks,
            getattr(self.context_cfg, "max_chunks_per_section", None),
        )
        context = build_context(used_chunks, self.context_cfg)

        # STRUCTURE: "few-shot is one of several strategies" is written in
        # no chunk — it lives in the heading tree. Prepend that tree so the
        # model can actually see it.
        document_map, breadth = self._document_map_for(query, used_chunks)
        if document_map:
            context = document_map + "\n\n" + context
        system_prompt = self._build_system_prompt(query)
        user_prompt = self._build_user_prompt(query, context)

        answer_text = self.llm_client.generate(system_prompt, user_prompt)

        # End-of-request cleanup: a single answer() call walks through
        # up to three different models (embedding -> reranker ->
        # generation), each of which already unloads itself the moment
        # its OWN step finishes (see model_manager.ensure_active calls
        # in EmbedGenerator/CrossEncoderReranker/LlmClient). This is
        # just a final safety net so nothing at all is left resident in
        # VRAM once the whole question-answer cycle is done — matters
        # most for a long-running process like webapp/app.py sitting
        # idle between chat messages.
        self._release_all_used_hosts()

        sources = [
            self._source_entry(chunk, number)
            for number, chunk in enumerate(used_chunks, start=1)
        ]

        return RagAnswer(
            answer=answer_text,
            sources=sources,
            context_used=context,
            document_map=document_map,
            breadth=breadth,
        )

    @staticmethod
    def _get_field(chunk, field: str, default=""):
        if isinstance(chunk, dict):
            return chunk.get(field, default)
        return getattr(chunk, field, default)

    def _chunk_metadata(self, chunk) -> dict:
        return self._get_field(chunk, "metadata", {}) or {}

    def _corpus_metadatas(self) -> list:
        """
        Metadata for the WHOLE corpus, read once and cached for the life of
        this answerer.

        The map has to describe the document, not just the handful of chunks
        this question happened to retrieve. If it were built from retrieved
        chunks alone, a question that only matched few-shot paragraphs would
        produce a "map" claiming the chapter contains nothing but few-shot —
        confidently wrong, which is worse than no map at all.

        Any failure (older chroma client, missing collection, an injected
        fake retriever without the method) degrades to an empty list, and
        the caller falls back to the retrieved chunks' own metadata.
        """
        if self._corpus_metadatas_cache is not None:
            return self._corpus_metadatas_cache

        metadatas: list = []
        chroma = getattr(self.retriever, "chroma", None)
        lister = getattr(chroma, "list_metadatas", None)
        if callable(lister):
            try:
                metadatas = list(lister() or [])
            except Exception:
                metadatas = []

        self._corpus_metadatas_cache = metadatas
        return metadatas

    def _document_map_for(self, query: str, used_chunks: list) -> tuple:
        """
        Build the document-map block for this question.

        Returns (map_text, diagnostics). map_text is "" whenever no map
        should be added; diagnostics always explains why, in both languages,
        so the Query Inspector can show the decision instead of leaving the
        reader guessing why one question got an outline and another did not.
        """
        cfg = self.context_cfg
        mode = str(getattr(cfg, "document_map", "off") or "off").lower()
        retrieved = [chunk_metadata(chunk) for chunk in used_chunks]
        info = {"mode": mode, "included": False}

        if mode == "off":
            info["reason_en"] = "Document map is turned off in settings."
            info["reason_fa"] = "نقشه سند در تنظیمات غیرفعال است."
            return "", info

        if mode == "auto":
            verdict = is_broad_question(query)
            info.update(
                {
                    "is_broad": verdict.is_broad,
                    "matched": verdict.matched,
                    "reason_en": verdict.reason_en,
                    "reason_fa": verdict.reason_fa,
                }
            )
            if not verdict.is_broad:
                return "", info
        else:
            info.update(
                {
                    "is_broad": None,
                    "reason_en": "document_map is set to 'always'.",
                    "reason_fa": "تنظیم document_map روی 'always' است.",
                }
            )

        block = build_document_map_block(
            query,
            self._corpus_metadatas() or retrieved,
            retrieved,
            max_chapters=getattr(cfg, "document_map_max_chapters", 3),
            max_sections_per_chapter=getattr(cfg, "document_map_max_sections", 12),
        )
        if not block:
            # Headings exist in no usable form, or none matched. Say so
            # plainly rather than silently producing nothing.
            info["reason_en"] = (
                "Broad question, but no chapter/section headings matched, so "
                "there was no map to add. Check that your documents use "
                "descriptive headings."
            )
            info["reason_fa"] = (
                "پرسش کلی بود، اما هیچ فصل/بخشی مطابقت نداشت؛ نقشه‌ای افزوده نشد."
            )
            return "", info

        info["included"] = True
        info["chars"] = len(block)
        return block, info

    def _source_entry(self, chunk, citation_number: int) -> dict:
        """
        One entry of RagAnswer.sources — everything a caller needs to show
        or verify where a bracketed [n] citation came from.

        `citation_number` is the SAME number the context block gave this
        chunk, so a UI can match "[2]" in the answer text to this entry
        without re-deriving anything.

        The chapter/section/page fields come from document_reference, which
        falls back to parsing the breadcrumb when a corpus was ingested
        before those fields existed — so references work on an existing
        chroma_db without re-running ingestion.
        """
        metadata = self._chunk_metadata(chunk)
        reference = structured_reference(metadata)
        return {
            "citation_number": citation_number,
            "chunk_id": self._get_field(chunk, "chunk_id", ""),
            "source_file": metadata.get("source_file", "?"),
            "breadcrumb": metadata.get("breadcrumb", ""),
            # Populated by metadata_enricher.py from inline OCR page
            # markers — lets callers show/verify exactly which page(s)
            # each bracketed [n] citation in the answer text points to.
            "page_numbers": metadata.get("page_numbers", []),
            # Structured location, split into named parts so no caller has
            # to parse strings to answer "which chapter was this?".
            "source": reference["source"],
            "pages": reference["pages"],
            # How the page was obtained: "markers" (exact, read from page
            # boundaries in the OCR output), "estimated" (inferred from the
            # chunk's position in the document), or "none". A UI that shows
            # a page must also show that it is approximate -- an estimate
            # dressed up as ground truth is worse than no page at all.
            "page_source": reference["page_source"],
            "pages_estimated": reference["pages_estimated"],
            "chapter": reference["chapter"],
            "section": reference["section"],
            "subsection": reference["subsection"],
            "heading_path": reference["heading_path"],
            "levels": reference["levels"],
            "reference": reference["text"],
            "reference_short": reference["short"],
        }

    def _release_all_used_hosts(self) -> None:
        """
        Best-effort final unload across every Ollama host this answer()
        call may have touched (embedder, ollama-backend reranker, LLM
        client). Silently skips anything that isn't a real Ollama-backed
        component (e.g. a test fake, or a huggingface-backend reranker,
        which was already cleaned up per-call by its own lifecycle
        helper) — this is cleanup, not something that should ever raise
        and break an answer that already succeeded.
        """
        hosts = set()
        for component in (getattr(self.retriever, "embedder", None), self.reranker, self.llm_client):
            cfg = getattr(component, "cfg", None)
            host = getattr(cfg, "host", None) or getattr(cfg, "ollama_host", None)
            if host:
                hosts.add(host)
        for host in hosts:
            try:
                model_manager.release_all(host, self.runtime)
            except Exception as e:
                # model_manager itself already warns to stderr on
                # unload/connection failures (see _safe_unload) — this
                # outer catch is only for something unexpected (e.g. a
                # bug in a caller's custom cfg object), and still
                # surfaces it rather than hiding it, while never
                # letting cleanup break an answer that already
                # succeeded.
                print(f"  [RagAnswerer] WARNING: model cleanup failed for {host}: {e}")

    def _build_system_prompt(self, query: str) -> str:
        # detect_language() only discriminates English vs. Spanish (see
        # preprocessors/language_detector.py), which isn't enough to
        # support arbitrary user languages. Rather than special-casing
        # each language detect_language() would need to learn, we ask the
        # model to mirror whatever language the question itself is in —
        # this covers any language the underlying LLM understands, not
        # just the two the heuristic detector knows about.
        base = (
            "You are a helpful assistant answering questions using ONLY the "
            "provided context. If the context doesn't contain enough "
            "information to answer, say so clearly rather than guessing or "
            "using outside knowledge. Every source in the context is "
            "numbered, e.g. '[1]', '[2]'. You MUST cite the bracketed "
            "number of every source you draw on, placed right after the "
            "claim it supports (e.g. 'Prompt engineering guides model "
            "output [1].'). Do not invent numbers that aren't in the "
            "context, and do not write an answer with zero citations if "
            "the context contains relevant sources. Always answer in the "
            "same language the question was asked in, whatever language "
            "that is, even if the provided context is in a different "
            "language."
        )

        # Each context entry's header line carries the document location it
        # came from (file, page, and the heading chain as "Chapter: ... >
        # Section: ..."). Without an explicit instruction, models cite the
        # bare number and drop that location — which is exactly the part a
        # reader needs to go and check the source document. Asking for it
        # costs a few tokens per citation and makes every claim verifiable.
        if getattr(self.context_cfg, "cite_document_location", False):
            base += (
                " Each numbered source begins with a header line giving its "
                "document location: the file name, the page number, and the "
                "position in the document as 'Chapter: ... > Section: ... > "
                "Subsection: ...'. When you cite a source, repeat that "
                "location in parentheses straight after the number, e.g. "
                "'Prompt engineering guides model output [1] (Binder2.pdf, "
                "page 12 — Chapter: Prompt Engineering > Section: "
                "Few-shot).'. Copy the file name, page number, chapter and "
                "section EXACTLY as they appear in that header line — never "
                "guess or reformat a page number, chapter or section, and "
                "omit any part that the header line does not provide."
            )

        return base

    def _build_user_prompt(self, query: str, context: str) -> str:
        return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
