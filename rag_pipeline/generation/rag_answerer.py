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
from rag_pipeline.generation.context_builder import build_context


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

    def answer(self, query: str) -> RagAnswer:
        if not query or not query.strip():
            return RagAnswer(answer=_NO_RESULTS_MESSAGE)

        chunks = self.retriever.retrieve(query)
        if self.reranker is not None:
            chunks = self.reranker.rerank(query, chunks)

        if not chunks:
            return RagAnswer(answer=_NO_RESULTS_MESSAGE)

        context = build_context(chunks, self.context_cfg)
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
            {
                "chunk_id": self._get_field(c, "chunk_id", ""),
                "source_file": self._chunk_metadata(c).get("source_file", "?"),
                "breadcrumb": self._chunk_metadata(c).get("breadcrumb", ""),
                # Populated by metadata_enricher.py from inline OCR page
                # markers — lets callers show/verify exactly which page(s)
                # each bracketed [n] citation in the answer text points to.
                "page_numbers": self._chunk_metadata(c).get("page_numbers", []),
            }
            for c in chunks
        ]

        return RagAnswer(answer=answer_text, sources=sources, context_used=context)

    @staticmethod
    def _get_field(chunk, field: str, default=""):
        if isinstance(chunk, dict):
            return chunk.get(field, default)
        return getattr(chunk, field, default)

    def _chunk_metadata(self, chunk) -> dict:
        return self._get_field(chunk, "metadata", {}) or {}

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
        return base

    def _build_user_prompt(self, query: str, context: str) -> str:
        return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
