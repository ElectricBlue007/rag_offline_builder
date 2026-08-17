"""
observability/query_trace.py -- "Why did I get THIS answer?"

When you ask a question in the web app, a lot happens before the LLM writes
one word:

    your question
      -> language detection            (which language is this?)
      -> query prefix + embedding      (turned into a vector)
      -> dense search in ChromaDB      (meaning-based, top N)
      -> sparse search in BM25         (keyword-based, top N)
      -> Reciprocal Rank Fusion        (merge the two ranked lists)
      -> optional cross-encoder rerank (precision pass)
      -> context builder               (only the first max_chunks survive!)
      -> prompt                        (system + context + question)
      -> LLM                           (the answer you see)

This module records ALL of that for a single question and returns it as a
plain dict, so the web UI can draw it.

HOW IT WORKS -- and why this design was chosen
----------------------------------------------
It does NOT re-implement the answering logic. Re-implementing it would mean
two code paths that slowly drift apart, and then the picture would lie.

Instead it temporarily wraps the methods of the live objects
(embedder.embed_texts, chroma.query, bm25.search, retriever.retrieve,
reranker.rerank, llm_client.generate), calls the ORIGINAL
RagAnswerer.answer(), and puts every method back afterwards in a finally
block -- even if the call raised.

Consequences, on purpose:
  * The answer the user gets is byte-for-byte the answer they would have got
    without tracing. This layer only watches.
  * The prompt shown in the UI is not a reconstruction: it is the exact
    string that was handed to llm_client.generate().
  * If anything in the recording itself goes wrong, the recording is dropped
    and the answer still returns. Observability must never cost you an answer.

No new dependencies. Standard library only.
"""

import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from rag_pipeline.document_reference import structured_reference

# A rough characters-per-token ratio, only used for the "how full is the
# context window" bar. Labelled as an estimate everywhere it is shown,
# because the real number depends on the model's tokenizer.
_CHARS_PER_TOKEN = 4.0

_CITATION_RE = re.compile(r"\[(\d+)\]")

# How much chunk text to ship to the browser per chunk. Enough to read and
# judge, small enough that a 30-candidate trace stays a reasonable payload.
_MAX_TEXT_TO_UI = 4000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get(obj: Any, name: str, default=None):
    """Field access that works for dicts, dataclasses and plain objects."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _tokens(text: str) -> int:
    return max(0, int(round(len(text or "") / _CHARS_PER_TOKEN)))


def _pages_to_str(pages) -> str:
    if isinstance(pages, list):
        return ",".join(str(p) for p in pages)
    return str(pages or "")


class _MethodPatch:
    """
    Temporarily replaces one method on one live object and always restores
    it -- including the subtle case where the object had no instance
    attribute of its own and was using the class method (then the correct
    restore is to delete the instance attribute, not to assign it back).
    """

    def __init__(self, obj: Any, attr: str, factory):
        self.obj = obj
        self.attr = attr
        self.factory = factory
        self.original = None
        self.had_own_attribute = False
        self.applied = False

    def apply(self) -> bool:
        if self.obj is None:
            return False
        original = getattr(self.obj, self.attr, None)
        if original is None or not callable(original):
            return False
        self.original = original
        self.had_own_attribute = self.attr in getattr(self.obj, "__dict__", {})
        try:
            setattr(self.obj, self.attr, self.factory(original))
            self.applied = True
        except Exception:
            # Some objects forbid attribute assignment (e.g. __slots__).
            # That stage simply goes unrecorded; everything else still works.
            self.applied = False
        return self.applied

    def restore(self) -> None:
        if not self.applied:
            return
        try:
            if self.had_own_attribute:
                setattr(self.obj, self.attr, self.original)
            else:
                delattr(self.obj, self.attr)
        except Exception:
            try:
                setattr(self.obj, self.attr, self.original)
            except Exception:
                pass


class QueryTracer:
    """
    Usage:
        tracer = QueryTracer(answerer)
        result, trace = tracer.run("What is a CMI?")

    `result` is exactly what answerer.answer(question) returns.
    `trace` is a JSON-serialisable dict describing how it got there.
    """

    def __init__(self, answerer: Any):
        self.answerer = answerer
        self.retriever = _get(answerer, "retriever")
        self.reranker = _get(answerer, "reranker")
        self.llm_client = _get(answerer, "llm_client")
        self.context_cfg = _get(answerer, "context_cfg")
        self.embedder = _get(self.retriever, "embedder")
        self.chroma = _get(self.retriever, "chroma")
        self.bm25 = _get(self.retriever, "bm25")
        self.retriever_cfg = _get(self.retriever, "cfg")

        self._rec: Dict[str, Any] = {}
        self._stages: List[Dict[str, Any]] = []

    # -- recording helpers ---------------------------------------------------

    def _stage(self, name: str, en: str, fa: str, seconds: float,
               items_in=None, items_out=None, note_en="", note_fa="") -> None:
        self._stages.append({
            "name": name,
            "label": {"en": en, "fa": fa},
            "seconds": round(seconds, 4),
            "items_in": items_in,
            "items_out": items_out,
            "note": {"en": note_en, "fa": note_fa},
        })

    # -- the wrappers --------------------------------------------------------

    def _patches(self) -> List[_MethodPatch]:
        rec = self._rec

        def embed_factory(original):
            def wrapper(texts, *a, **kw):
                t0 = time.time()
                vectors = original(texts, *a, **kw)
                elapsed = time.time() - t0
                try:
                    sent = list(texts) if texts else []
                    # The retriever embeds exactly one query text here.
                    rec["embedded_query_text"] = sent[0] if sent else ""
                    dim = len(vectors[0]) if vectors else 0
                    rec["vector_dim"] = dim
                    self._stage(
                        "embed_query", "Embed the question", "تبدیل سوال به بردار",
                        elapsed, items_in=len(sent), items_out=dim,
                        note_en="The question is turned into a vector of numbers so it can be "
                                "compared by meaning against every stored chunk.",
                        note_fa="سوال به بردار عددی تبدیل می‌شود تا بر اساس معنا مقایسه شود.",
                    )
                except Exception:
                    pass
                return vectors
            return wrapper

        def chroma_query_factory(original):
            def wrapper(query_vector, *a, **kw):
                t0 = time.time()
                raw = original(query_vector, *a, **kw)
                elapsed = time.time() - t0
                try:
                    def unwrap(key):
                        value = raw.get(key) if isinstance(raw, dict) else None
                        if not value:
                            return []
                        return value[0] if isinstance(value[0], list) else value

                    ids = unwrap("ids")
                    distances = unwrap("distances")
                    requested = kw.get("n_results", a[0] if a else None)
                    where = kw.get("where")
                    rec["dense"] = {
                        "requested": requested,
                        "returned": len(ids),
                        "ids": list(ids),
                        "distances": [round(float(d), 5) for d in distances] if distances else [],
                        "where": where,
                    }
                    self._stage(
                        "dense_search", "Dense search (meaning)", "جستجوی معنایی",
                        elapsed, items_in=requested, items_out=len(ids),
                        note_en="ChromaDB returns the chunks whose vectors are closest to the "
                                "question's vector. This finds text that MEANS the same thing "
                                "even when it uses different words.",
                        note_fa="قطعاتی که معنای نزدیک به سوال دارند — حتی اگر کلمات متفاوت باشد.",
                    )
                except Exception:
                    pass
                return raw
            return wrapper

        def bm25_search_factory(original):
            def wrapper(query, *a, **kw):
                t0 = time.time()
                results = original(query, *a, **kw)
                elapsed = time.time() - t0
                try:
                    requested = kw.get("n_results", a[0] if a else None)
                    rec["bm25_query_text"] = query
                    rec["sparse"] = {
                        "requested": requested,
                        "returned": len(results),
                        "ids": [_get(r, "chunk_id", "") for r in results],
                        "scores": [round(float(_get(r, "score", 0.0)), 4) for r in results],
                    }
                    self._stage(
                        "sparse_search", "Sparse search (keywords)", "جستجوی کلمه‌ای",
                        elapsed, items_in=requested, items_out=len(results),
                        note_en="BM25 matches the actual words of your question. This catches exact "
                                "terms, codes and names that a vector search can miss.",
                        note_fa="BM25 دقیقاً کلمات سوال را پیدا می‌کند؛ مناسب برای کدها و نام‌ها.",
                    )
                except Exception:
                    pass
                return results
            return wrapper

        def retrieve_factory(original):
            def wrapper(query, *a, **kw):
                t0 = time.time()
                chunks = original(query, *a, **kw)
                elapsed = time.time() - t0
                try:
                    rec["fused"] = [self._chunk_row(c) for c in chunks]
                    self._stage(
                        "fuse_rrf", "Merge both lists (RRF)", "ادغام دو لیست (RRF)",
                        elapsed, items_out=len(chunks),
                        note_en="Reciprocal Rank Fusion merges the meaning list and the keyword "
                                "list using POSITION only, never the raw scores (they are on "
                                "different scales). A chunk found by both rises to the top.",
                        note_fa="ادغام بر اساس رتبه، نه امتیاز خام. قطعه‌ای که هر دو پیدا کرده‌اند بالا می‌آید.",
                    )
                except Exception:
                    pass
                return chunks
            return wrapper

        def rerank_factory(original):
            def wrapper(query, candidates, *a, **kw):
                t0 = time.time()
                reranked = original(query, candidates, *a, **kw)
                elapsed = time.time() - t0
                try:
                    rec["reranked"] = [self._chunk_row(c) for c in reranked]
                    self._stage(
                        "rerank", "Rerank (precision pass)", "بازرتبه‌بندی",
                        elapsed, items_in=len(candidates or []), items_out=len(reranked),
                        note_en="A cross-encoder reads the question TOGETHER WITH each candidate "
                                "and scores real relevance, then keeps only the best ones. This is "
                                "slower but much more accurate than vector distance.",
                        note_fa="مدل سوال و متن را با هم می‌خواند و مرتبط‌ترین‌ها را نگه می‌دارد.",
                    )
                except Exception:
                    pass
                return reranked
            return wrapper

        def generate_factory(original):
            def wrapper(system_prompt, user_prompt, *a, **kw):
                t0 = time.time()
                answer = original(system_prompt, user_prompt, *a, **kw)
                elapsed = time.time() - t0
                try:
                    rec["system_prompt"] = system_prompt
                    rec["user_prompt"] = user_prompt
                    self._stage(
                        "generate", "Ask the LLM", "پرسیدن از مدل زبانی",
                        elapsed,
                        items_in=_tokens(system_prompt) + _tokens(user_prompt),
                        items_out=_tokens(answer or ""),
                        note_en="Everything below is literally what the model received. The model "
                                "cannot see any chunk that is not in this text.",
                        note_fa="متن زیر دقیقاً همان چیزی است که مدل دریافت کرد. هرچه اینجا نباشد، مدل نمی‌بیند.",
                    )
                except Exception:
                    pass
                return answer
            return wrapper

        return [
            _MethodPatch(self.embedder, "embed_texts", embed_factory),
            _MethodPatch(self.chroma, "query", chroma_query_factory),
            _MethodPatch(self.bm25, "search", bm25_search_factory),
            _MethodPatch(self.retriever, "retrieve", retrieve_factory),
            _MethodPatch(self.reranker, "rerank", rerank_factory),
            _MethodPatch(self.llm_client, "generate", generate_factory),
        ]

    @staticmethod
    def _chunk_row(chunk: Any) -> Dict[str, Any]:
        metadata = _get(chunk, "metadata", {}) or {}
        text = _get(chunk, "text", "") or ""
        # Where this passage sits in the original document. Derived here so
        # the inspector can show "page 12, Chapter X > Section Y" for every
        # candidate -- including the ones that were dropped, which is how a
        # reader checks whether the RIGHT part of the document was found.
        reference = structured_reference(metadata)
        return {
            "chunk_id": _get(chunk, "chunk_id", ""),
            "text": text,
            "chars_total": len(text),
            "source_file": metadata.get("source_pdf") or metadata.get("source_file", "?"),
            "breadcrumb": metadata.get("breadcrumb", ""),
            "reference": reference["text"],
            "reference_short": reference["short"],
            "chapter": reference["chapter"],
            "section": reference["section"],
            "subsection": reference["subsection"],
            "levels": reference["levels"],
            "page_numbers": _pages_to_str(metadata.get("page_numbers", "")),
            # "markers" = exact page, read from a page boundary in the OCR
            # output. "estimated" = inferred from where the chunk sits in
            # the document. "none" = unknown. Shown in the inspector so a
            # reader can tell a verifiable page from an approximate one.
            "page_source": str(metadata.get("page_source") or "none"),
            "chunk_type": metadata.get("chunk_type", metadata.get("type", "text")),
            "language": metadata.get("language", ""),
            "rrf_score": _get(chunk, "rrf_score", _get(chunk, "original_rrf_score", None)),
            "dense_rank": _get(chunk, "dense_rank", None),
            "sparse_rank": _get(chunk, "sparse_rank", None),
            "rerank_score": _get(chunk, "rerank_score", None),
            "original_rank": _get(chunk, "original_rank", None),
        }

    # -- main entry point ----------------------------------------------------

    def run(self, question: str) -> Tuple[Any, Optional[Dict[str, Any]]]:
        patches = [p for p in self._patches() if p.apply()]
        started = time.time()
        try:
            result = self.answerer.answer(question)
        finally:
            for patch in patches:
                patch.restore()
        total = time.time() - started

        try:
            trace = self._build(question, result, total)
        except Exception as exc:      # a broken picture must not cost an answer
            trace = {
                "question": question,
                "error": f"{type(exc).__name__}: {exc}",
                "stages": self._stages,
            }
        return result, trace

    # -- turning raw recordings into the picture -----------------------------

    def _build(self, question: str, result: Any, total: float) -> Dict[str, Any]:
        rec = self._rec
        cctx = self.context_cfg
        max_chunks = _get(cctx, "max_chunks", None)
        max_chars = _get(cctx, "max_chars_per_chunk", None)

        # The list the answerer actually sliced for the prompt is the reranked
        # list when a reranker ran, otherwise the fused list.
        final_list = rec.get("reranked") or rec.get("fused") or []
        fused_list = rec.get("fused") or []
        fused_rank_by_id = {c["chunk_id"]: i + 1 for i, c in enumerate(fused_list)}

        answer_text = _get(result, "answer", "") or ""
        cited_numbers = {int(n) for n in _CITATION_RE.findall(answer_text)}

        candidates: List[Dict[str, Any]] = []
        for index, row in enumerate(final_list):
            position = index + 1
            used = (max_chunks is None) or (index < max_chunks)
            chars_total = row["chars_total"]
            chars_sent = chars_total
            truncated = False
            if used and max_chars is not None and chars_total > max_chars:
                chars_sent = max_chars
                truncated = True
            if not used:
                chars_sent = 0

            fused_rank = fused_rank_by_id.get(row["chunk_id"])
            rerank_rank = position if row.get("rerank_score") is not None else None
            moved = None
            if rerank_rank is not None and fused_rank is not None:
                moved = fused_rank - rerank_rank      # +2 means it climbed 2 places

            found_by = "both"
            if row.get("dense_rank") is None and row.get("sparse_rank") is not None:
                found_by = "sparse"
            elif row.get("sparse_rank") is None and row.get("dense_rank") is not None:
                found_by = "dense"
            elif row.get("dense_rank") is None and row.get("sparse_rank") is None:
                found_by = "unknown"

            candidates.append({
                **row,
                "text": row["text"][:_MAX_TEXT_TO_UI],
                "text_truncated_for_display": len(row["text"]) > _MAX_TEXT_TO_UI,
                "final_rank": position,
                "fused_rank": fused_rank,
                "rerank_rank": rerank_rank,
                "rank_change": moved,
                "found_by": found_by,
                "used_in_prompt": used,
                "citation_number": position if used else None,
                "cited_in_answer": bool(used and position in cited_numbers),
                "chars_sent": chars_sent,
                "truncated_in_prompt": truncated,
            })

        used_rows = [c for c in candidates if c["used_in_prompt"]]
        dropped_rows = [c for c in candidates if not c["used_in_prompt"]]

        system_prompt = rec.get("system_prompt", "")
        user_prompt = rec.get("user_prompt", "")
        context_block = _get(result, "context_used", "") or ""
        prompt_chars = len(system_prompt) + len(user_prompt)
        num_ctx = _get(_get(self.llm_client, "cfg", None), "num_ctx", None)

        trace: Dict[str, Any] = {
            "question": question,
            "started_at": _now_iso(),
            "total_seconds": round(total, 3),
            "stages": self._stages,
            "query_texts": {
                "raw": question,
                "embedded": rec.get("embedded_query_text", ""),
                "bm25": rec.get("bm25_query_text", ""),
            },
            "dense": rec.get("dense", {}),
            "sparse": rec.get("sparse", {}),
            "funnel": [
                {"key": "dense", "label": {"en": "Dense (meaning)", "fa": "معنایی"},
                 "count": rec.get("dense", {}).get("returned", 0)},
                {"key": "sparse", "label": {"en": "Sparse (keywords)", "fa": "کلمه‌ای"},
                 "count": rec.get("sparse", {}).get("returned", 0)},
                {"key": "fused", "label": {"en": "After merge (RRF)", "fa": "پس از ادغام"},
                 "count": len(fused_list)},
                {"key": "reranked", "label": {"en": "After rerank", "fa": "پس از بازرتبه‌بندی"},
                 "count": len(rec.get("reranked") or []) if rec.get("reranked") is not None else None},
                {"key": "used", "label": {"en": "Sent to the model", "fa": "ارسال به مدل"},
                 "count": len(used_rows)},
                {"key": "cited", "label": {"en": "Cited in answer", "fa": "استنادشده"},
                 "count": len([c for c in used_rows if c["cited_in_answer"]])},
            ],
            "candidates": candidates,
            "prompt": {
                "system": system_prompt,
                "user": user_prompt,
                "context_block": context_block,
                "chars": prompt_chars,
                "tokens_estimate": _tokens(system_prompt) + _tokens(user_prompt),
                "context_chars": len(context_block),
                "num_ctx": num_ctx,
                "context_window_used_pct": (
                    round(100.0 * (_tokens(system_prompt) + _tokens(user_prompt)) / num_ctx, 1)
                    if num_ctx else None
                ),
            },
            "answer": {
                "text": answer_text,
                "chars": len(answer_text),
                "citations": sorted(cited_numbers),
            },
            # How the general-question handling behaved: whether the question
            # was judged broad, whether the document map was added, and the
            # map text itself. Without this the reader cannot tell why one
            # question got a chapter outline and another did not.
            "breadth": dict(_get(result, "breadth", {}) or {}),
            "document_map": _get(result, "document_map", "") or "",
            "sections_used": sorted(
                {
                    (c.get("section") or c.get("chapter") or c.get("source_file") or "?")
                    for c in used_rows
                }
            ),
            "settings": self._settings_snapshot(max_chunks, max_chars, num_ctx),
        }
        trace["warnings"] = self._warnings(trace, used_rows, dropped_rows, cited_numbers)
        return trace

    def _settings_snapshot(self, max_chunks, max_chars, num_ctx) -> Dict[str, Any]:
        rcfg = self.retriever_cfg
        rrcfg = _get(self.reranker, "cfg", None)
        lcfg = _get(self.llm_client, "cfg", None)
        ecfg = _get(self.embedder, "cfg", None)
        return {
            "embedding_model": _get(ecfg, "model_name", "?"),
            "generation_model": _get(lcfg, "model", "?"),
            "temperature": _get(lcfg, "temperature", None),
            "num_ctx": num_ctx,
            "rerank_enabled": self.reranker is not None,
            "reranker_backend": _get(rrcfg, "backend", None),
            "reranker_model": (
                _get(rrcfg, "ollama_model", None) if _get(rrcfg, "backend", None) == "ollama"
                else _get(rrcfg, "model_name", None)
            ),
            "dense_n_results": _get(rcfg, "dense_n_results", None),
            "sparse_n_results": _get(rcfg, "sparse_n_results", None),
            "final_n_results": _get(rcfg, "final_n_results", None),
            "rrf_k": _get(rcfg, "rrf_k", None),
            "language_filter": _get(rcfg, "language_filter", None),
            "where_filter": (self._rec.get("dense", {}) or {}).get("where"),
            "max_chunks": max_chunks,
            "max_chars_per_chunk": max_chars,
        }

    @staticmethod
    def _warnings(trace, used_rows, dropped_rows, cited_numbers) -> List[Dict[str, str]]:
        """
        The judgements a reviewer would make while looking at this trace.
        Each one is a concrete, checkable statement about THIS question --
        not generic advice.
        """
        out: List[Dict[str, str]] = []

        def add(level, code, en, fa):
            out.append({"level": level, "code": code, "en": en, "fa": fa})

        dense = trace["dense"] or {}
        sparse = trace["sparse"] or {}
        settings = trace["settings"]

        if not used_rows:
            add("red", "nothing_sent",
                "No chunk reached the model. The answer cannot be grounded in your documents.",
                "هیچ قطعه‌ای به مدل نرسید؛ پاسخ متکی بر اسناد شما نیست.")

        if dropped_rows:
            add("yellow", "dropped",
                f"{len(dropped_rows)} retrieved chunk(s) were thrown away before the prompt "
                f"because context_builder.max_chunks = {settings['max_chunks']}. "
                f"Retrieval paid for them; the model never saw them.",
                f"{len(dropped_rows)} قطعه بازیابی شد اما به دلیل محدودیت max_chunks "
                f"({settings['max_chunks']}) به مدل نرسید.")

        truncated = [c for c in used_rows if c["truncated_in_prompt"]]
        if truncated:
            add("yellow", "truncated",
                f"{len(truncated)} chunk(s) were cut at {settings['max_chars_per_chunk']} "
                "characters. If the answer is missing a detail, it may have been in the cut part.",
                f"{len(truncated)} قطعه در {settings['max_chars_per_chunk']} کاراکتر بریده شد؛ ممکن است "
                "بخش حذف‌شده حاوی پاسخ بوده باشد.")

        uncited = [c for c in used_rows if not c["cited_in_answer"]]
        if used_rows and uncited:
            add("yellow", "uncited",
                f"The model was given {len(used_rows)} chunk(s) but cited only "
                f"{len(used_rows) - len(uncited)}. Unused context is noise and can lower quality.",
                f"از {len(used_rows)} قطعه ارسال‌شده، فقط به {len(used_rows) - len(uncited)} مورد استناد شد.")

        bad_citations = [n for n in cited_numbers if n > len(used_rows)]
        if bad_citations:
            add("red", "invalid_citation",
                f"The answer cites {bad_citations}, but only [1]..[{len(used_rows)}] existed in "
                "the prompt. That citation is invented -- treat the answer with suspicion.",
                f"پاسخ به {bad_citations} استناد کرده ولی فقط [1] تا [{len(used_rows)}] وجود داشت؛ این استناد ساختگی است.")

        sources = {c["source_file"] for c in used_rows}
        if len(used_rows) > 1 and len(sources) == 1:
            add("yellow", "single_source",
                f"Every chunk sent came from one file ({list(sources)[0]}). The answer reflects "
                "one document only.",
                f"همه قطعات از یک فایل بودند ({list(sources)[0]}).")

        if dense.get("returned") == 0:
            where = dense.get("where")
            if where:
                add("red", "language_filter_empty",
                    f"Dense search returned nothing while filtering on {where}. If the question's "
                    "detected language does not match your documents' language, this filter hides "
                    "everything. Set retrieval.hybrid.language_filter: false to test.",
                    f"جستجوی معنایی با فیلتر {where} هیچ نتیجه‌ای نداد. برای آزمون، language_filter را false کنید.")
            else:
                add("red", "dense_empty",
                    "Dense search returned nothing. Is the Chroma collection built and non-empty?",
                    "جستجوی معنایی نتیجه‌ای نداشت. آیا پایگاه Chroma ساخته شده است؟")

        if sparse.get("returned") == 0:
            add("yellow", "sparse_empty",
                "Keyword search returned nothing. Is bm25_index.pkl built and current?",
                "جستجوی کلمه‌ای نتیجه‌ای نداشت. آیا bm25_index.pkl ساخته شده است؟")

        only_one_source_type = [c for c in used_rows if c["found_by"] == "dense"]
        if used_rows and len(only_one_source_type) == len(used_rows) and sparse.get("returned"):
            add("yellow", "keyword_ignored",
                "Every chunk in the prompt came from the meaning search only; the keyword search "
                "contributed nothing to the final list.",
                "همه قطعات نهایی فقط از جستجوی معنایی آمدند.")

        pct = trace["prompt"]["context_window_used_pct"]
        if pct is not None and pct > 90:
            add("red", "context_overflow_risk",
                f"The prompt fills about {pct}% of the model's context window "
                f"({settings['num_ctx']} tokens). Text beyond the window is silently dropped.",
                f"پرامپت حدود {pct}% پنجره مدل را پر کرده است؛ مازاد بدون هشدار حذف می‌شود.")

        if not out:
            add("green", "looks_healthy",
                "Nothing suspicious: every chunk sent to the model was cited, nothing was cut, "
                "and both retrievers contributed.",
                "مشکلی دیده نشد: به همه قطعات استناد شد، متنی بریده نشد و هر دو جستجو نقش داشتند.")
        return out


def trace_answer(answerer: Any, question: str) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """Convenience wrapper: returns (answer_result, trace_dict)."""
    return QueryTracer(answerer).run(question)


def demo_trace() -> Dict[str, Any]:
    """
    A realistic, clearly-labelled example trace.

    Purpose: let anyone open the web app and SEE what the inspector looks
    like before Ollama, Chroma and BM25 are running -- and let the UI be
    tested without any model. The numbers below are illustrative, and the UI
    marks this trace as a demo.
    """
    def chunk(i, cid, src, page, crumb, score, dense, sparse, rr, used, cited, text, ctype="text"):
        reference = structured_reference({
            "source_file": src, "breadcrumb": crumb, "page_numbers": page,
        })
        return {
            "chunk_id": cid, "text": text, "chars_total": len(text),
            "source_file": src, "breadcrumb": crumb, "page_numbers": page,
            "reference": reference["text"], "reference_short": reference["short"],
            "chapter": reference["chapter"], "section": reference["section"],
            "subsection": reference["subsection"], "levels": reference["levels"],
            # The demo shows exact pages, as if the OCR output contained
            # page markers. Real traces report "estimated" when the page was
            # inferred from position instead of read from a marker.
            "page_source": "markers",
            "chunk_type": ctype, "language": "en",
            "rrf_score": score, "dense_rank": dense, "sparse_rank": sparse,
            "rerank_score": rr, "original_rank": i,
            "final_rank": i, "fused_rank": i, "rerank_rank": i, "rank_change": 0,
            "found_by": "both" if (dense and sparse) else ("dense" if dense else "sparse"),
            "used_in_prompt": used, "citation_number": i if used else None,
            "cited_in_answer": cited, "chars_sent": len(text) if used else 0,
            "truncated_in_prompt": False, "text_truncated_for_display": False,
        }

    candidates = [
        chunk(1, "Testing.md::c12", "Testing.md", "3", "Prompt Engineering > One-shot & few-shot",
              0.0325, 1, 2, 0.94, True, True,
              "One-shot prompting gives the model a single worked example before the real "
              "question. Few-shot gives several. Both raise accuracy on formatting-sensitive "
              "tasks without any fine-tuning."),
        chunk(2, "Testing.md::c07", "Testing.md", "2", "Prompt Engineering > Zero-shot",
              0.0301, 3, 1, 0.88, True, True,
              "Zero-shot prompting asks the model to perform a task with no examples at all, "
              "relying only on the instruction."),
        chunk(3, "Binder2.md::c31", "Binder2.md", "11", "Operator actions > Create SOI",
              0.0164, 6, None, 0.42, True, False,
              "Create SOI - logical entry: CMI track; real entry: Enter NT in the action field.",
              ctype="table"),
        chunk(4, "Testing.md::c19", "Testing.md", "5", "Prompt Engineering > Temperature",
              0.0157, 8, None, 0.31, False, False,
              "Temperature controls randomness. Lower values make output more deterministic."),
        chunk(5, "Binder2.md::c02", "Binder2.md", "1", "Introduction",
              0.0151, None, 4, 0.12, False, False,
              "This binder describes the operating procedures for the system."),
    ]
    used = [c for c in candidates if c["used_in_prompt"]]
    context_block = "\n\n".join(
        f"[{c['citation_number']}] {'Table' if c['chunk_type'] == 'table' else 'Text'} "
        f"— {c['reference']}\n{c['text']}"
        for c in used
    )
    question = "What is one-shot prompting?"
    system_prompt = (
        "You are a helpful assistant answering questions using ONLY the provided context. "
        "Every source in the context is numbered, e.g. '[1]', '[2]'. You MUST cite the "
        "bracketed number of every source you draw on."
    )
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"
    answer_text = (
        "One-shot prompting means giving the model exactly one worked example before asking "
        "the real question [1]. It sits between zero-shot, which provides no examples at "
        "all [2], and few-shot, which provides several [1]."
    )

    return {
        "demo": True,
        "question": question,
        "started_at": _now_iso(),
        "total_seconds": 4.812,
        "stages": [
            {"name": "embed_query", "label": {"en": "Embed the question", "fa": "تبدیل سوال به بردار"},
             "seconds": 0.412, "items_in": 1, "items_out": 4096,
             "note": {"en": "The question becomes a vector so it can be compared by meaning.",
                      "fa": "سوال به بردار تبدیل می‌شود تا بر اساس معنا مقایسه شود."}},
            {"name": "dense_search", "label": {"en": "Dense search (meaning)", "fa": "جستجوی معنایی"},
             "seconds": 0.087, "items_in": 30, "items_out": 30,
             "note": {"en": "Finds text that means the same thing, even with different words.",
                      "fa": "متنی با معنای مشابه را پیدا می‌کند."}},
            {"name": "sparse_search", "label": {"en": "Sparse search (keywords)", "fa": "جستجوی کلمه‌ای"},
             "seconds": 0.014, "items_in": 30, "items_out": 30,
             "note": {"en": "Matches the exact words, codes and names in your question.",
                      "fa": "کلمات و کدهای دقیق سوال را تطبیق می‌دهد."}},
            {"name": "fuse_rrf", "label": {"en": "Merge both lists (RRF)", "fa": "ادغام دو لیست"},
             "seconds": 0.002, "items_in": 60, "items_out": 5,
             "note": {"en": "Merges by rank position, not by raw score.",
                      "fa": "ادغام بر اساس رتبه انجام می‌شود."}},
            {"name": "generate", "label": {"en": "Ask the LLM", "fa": "پرسیدن از مدل"},
             "seconds": 4.297, "items_in": _tokens(system_prompt + user_prompt),
             "items_out": _tokens(answer_text),
             "note": {"en": "The model can only see what is in this prompt.",
                      "fa": "مدل فقط همین پرامپت را می‌بیند."}},
        ],
        "query_texts": {
            "raw": question,
            "embedded": f"Instruct: Given a web search query, retrieve relevant passages\nQuery: {question}",
            "bm25": question,
        },
        "dense": {"requested": 30, "returned": 30, "ids": [], "distances": [], "where": {"language": "en"}},
        "sparse": {"requested": 30, "returned": 30, "ids": [], "scores": []},
        "funnel": [
            {"key": "dense", "label": {"en": "Dense (meaning)", "fa": "معنایی"}, "count": 30},
            {"key": "sparse", "label": {"en": "Sparse (keywords)", "fa": "کلمه‌ای"}, "count": 30},
            {"key": "fused", "label": {"en": "After merge (RRF)", "fa": "پس از ادغام"}, "count": 5},
            {"key": "reranked", "label": {"en": "After rerank", "fa": "پس از بازرتبه‌بندی"}, "count": 5},
            {"key": "used", "label": {"en": "Sent to the model", "fa": "ارسال به مدل"}, "count": 3},
            {"key": "cited", "label": {"en": "Cited in answer", "fa": "استنادشده"}, "count": 2},
        ],
        "candidates": candidates,
        "prompt": {
            "system": system_prompt,
            "user": user_prompt,
            "context_block": context_block,
            "chars": len(system_prompt) + len(user_prompt),
            "tokens_estimate": _tokens(system_prompt) + _tokens(user_prompt),
            "context_chars": len(context_block),
            "num_ctx": 129024,
            "context_window_used_pct": 0.3,
        },
        "answer": {"text": answer_text, "chars": len(answer_text), "citations": [1, 2]},
        "breadth": {
            "mode": "auto",
            "included": True,
            "is_broad": True,
            "matched": "what is",
            "reason_en": "Overview phrasing detected ('what is') in a short question, "
                         "so the document map was added for orientation.",
            "reason_fa": "عبارت کلی ('what is') در پرسشی کوتاه دیده شد؛ نقشه سند افزوده شد.",
            "chars": 214,
        },
        "document_map": (
            "Document map -- how the source document is organised. Orientation only: "
            "use it to explain how topics relate and what a chapter covers, but do NOT "
            "cite it and do not state facts from it beyond the structure shown. Cite "
            "only the numbered sources below.\n"
            "- Testing.pdf > Chapter: Prompt Engineering\n"
            "  Sections: Zero-shot; One-shot & few-shot; Chain-of-thought; Temperature"
        ),
        "sections_used": ["Chain-of-thought", "One-shot & few-shot", "Zero-shot"],
        "settings": {
            "embedding_model": "qwen3-embedding:8b",
            "generation_model": "qwen2.5:14b",
            "temperature": 0.1,
            "num_ctx": 129024,
            "rerank_enabled": True,
            "reranker_backend": "ollama",
            "reranker_model": "B-A-M-N/qwen3-reranker-0.6b-fp16:latest",
            "dense_n_results": 30,
            "sparse_n_results": 30,
            "final_n_results": 20,
            "rrf_k": 60,
            "language_filter": True,
            "where_filter": {"language": "en"},
            "max_chunks": 3,
            "max_chars_per_chunk": 1500,
        },
        "warnings": [
            {"level": "yellow", "code": "dropped",
             "en": "2 retrieved chunk(s) were thrown away before the prompt because "
                   "context_builder.max_chunks = 3. Retrieval paid for them; the model never saw them.",
             "fa": "2 قطعه بازیابی شد اما به دلیل محدودیت max_chunks = 3 به مدل نرسید."},
            {"level": "yellow", "code": "uncited",
             "en": "The model was given 3 chunk(s) but cited only 2. Unused context is noise "
                   "and can lower quality.",
             "fa": "از 3 قطعه ارسال‌شده، فقط به 2 مورد استناد شد."},
        ],
    }
