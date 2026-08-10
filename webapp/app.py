#!/usr/bin/env python3
"""
webapp/app.py — a small Flask server that puts a proper chat UI in front
of the existing RAG pipeline (hybrid retrieval + optional reranking +
Ollama generation), instead of the CLI's interactive prompt loop.

WHERE THIS FILE GOES
    your_project/
    ├── rag_pipeline/        <- the existing package (config.py, run_query.py, ...)
    └── webapp/              <- this folder
        ├── app.py           <- this file
        ├── chat_store.py
        └── static/index.html

RUN IT
    cd your_project
    pip install flask
    python webapp/app.py
    # then open http://localhost:5000 in a browser

    Optional flags mirror run_query.py:
    python webapp/app.py --rerank --model qwen3-embedding8b:latest --gen-model llama3.2

Everything else (Ollama running locally, ChromaDB + BM25 index already
built via run_ingestion.py / run_bm25_index.py) must already be in place,
exactly as required by run_query.py / run_answer.py today.
"""

import argparse
import sys
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

# Make the sibling `rag_pipeline` package importable, same trick run_query.py uses.
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.config import (  # noqa: E402
    EmbedGeneratorConfig,
    ChromaIngestionConfig,
    BM25IndexConfig,
    HybridRetrieverConfig,
    RerankerConfig,
    LlmClientConfig,
    ContextBuilderConfig,
)
from rag_pipeline.settings import load_settings  # noqa: E402
from rag_pipeline.embedding.embed_generator import EmbedGenerator  # noqa: E402
from rag_pipeline.storage.chroma_ingestion import ChromaIngestor  # noqa: E402
from rag_pipeline.storage.bm25_index import BM25Index  # noqa: E402
from rag_pipeline.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from rag_pipeline.retrieval.reranker import CrossEncoderReranker  # noqa: E402
from rag_pipeline.generation.llm_client import LlmClient  # noqa: E402
from rag_pipeline.generation.rag_answerer import RagAnswerer  # noqa: E402

from chat_store import ChatStore  # noqa: E402

_S = load_settings()


def parse_args():
    p = argparse.ArgumentParser(description="Chat web UI for the RAG pipeline.")
    p.add_argument("--chroma-dir", default=_S.paths.chroma_dir)
    p.add_argument("--collection", default=_S.paths.collection_name)
    p.add_argument("--bm25-index", default=_S.paths.bm25_index_path)
    p.add_argument("--model", default=_S.embed_generator.model_name,
                    help="Embedding model tag. Default: settings.yaml models.embedding.")
    p.add_argument("--gen-model", default=_S.llm_client.model,
                    help="Ollama generation model tag. Default: settings.yaml models.generation.")
    p.add_argument("--ollama-host", default=_S.llm_client.host)
    p.add_argument("--n-results", type=int, default=_S.n_results)
    p.add_argument("--rerank", action="store_true", default=_S.rerank_runtime.enabled)
    p.add_argument("--rerank-model", default=_S.reranker.model_name)
    p.add_argument("--candidate-pool", type=int, default=_S.rerank_runtime.candidate_pool)
    p.add_argument("--history-file", default=_S.paths.chat_history_file)
    p.add_argument("--host", default=_S.webapp.host)
    p.add_argument("--port", type=int, default=_S.webapp.port)
    p.add_argument("--debug", action="store_true", default=_S.webapp.debug)
    return p.parse_args()


def build_answerer(args) -> RagAnswerer:
    embedder = EmbedGenerator(
        EmbedGeneratorConfig(model_name=args.model, host=args.ollama_host, num_ctx=_S.embed_generator.num_ctx),
        runtime=_S.runtime,
    )
    chroma = ChromaIngestor(ChromaIngestionConfig(
        persist_directory=args.chroma_dir, collection_name=args.collection
    ))
    bm25 = BM25Index(BM25IndexConfig(persist_path=args.bm25_index))
    bm25.load()

    final_n = args.candidate_pool if args.rerank else args.n_results
    retriever = HybridRetriever(
        chroma, bm25, embedder, HybridRetrieverConfig(final_n_results=final_n)
    )

    reranker = None
    if args.rerank:
        reranker = CrossEncoderReranker(
            RerankerConfig(
                model_name=args.rerank_model,
                final_n_results=args.n_results,
                ollama_num_ctx=_S.reranker.ollama_num_ctx,
            ),
            runtime=_S.runtime,
        )

    llm_client = LlmClient(
        LlmClientConfig(model=args.gen_model, host=args.ollama_host, num_ctx=_S.llm_client.num_ctx),
        runtime=_S.runtime,
    )

    return RagAnswerer(
        retriever=retriever,
        llm_client=llm_client,
        reranker=reranker,
        context_config=ContextBuilderConfig(
            include_type_label=_S.context_builder.include_type_label,
            include_source=_S.context_builder.include_source,
            include_breadcrumb=_S.context_builder.include_breadcrumb,
            max_chunks=args.n_results,
            max_chars_per_chunk=_S.context_builder.max_chars_per_chunk,
        ),
        runtime=_S.runtime,
    )


def create_app(args) -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="")
    store = ChatStore(args.history_file)

    # Built lazily on first question, not at import time, so the server can
    # still start (and serve the UI / chat list) even if Ollama/Chroma
    # aren't reachable yet — the error surfaces in the chat instead of
    # crashing the whole process.
    state = {"answerer": None, "error": None}

    def get_answerer():
        if state["answerer"] is None and state["error"] is None:
            try:
                state["answerer"] = build_answerer(args)
            except Exception as exc:  # noqa: BLE001
                state["error"] = str(exc)
                traceback.print_exc()
        return state["answerer"]

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/chats")
    def list_chats():
        return jsonify(store.list_chats())

    @app.post("/api/chats")
    def new_chat():
        chat = store.create_chat()
        return jsonify(chat)

    @app.get("/api/chats/<chat_id>")
    def get_chat(chat_id):
        chat = store.get_chat(chat_id)
        if chat is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(chat)

    @app.delete("/api/chats/<chat_id>")
    def delete_chat(chat_id):
        ok = store.delete_chat(chat_id)
        return jsonify({"deleted": ok}), (200 if ok else 404)

    @app.patch("/api/chats/<chat_id>")
    def rename_chat(chat_id):
        body = request.get_json(force=True, silent=True) or {}
        title = (body.get("title") or "").strip()
        if not title:
            return jsonify({"error": "title required"}), 400
        ok = store.rename_chat(chat_id, title)
        return jsonify({"renamed": ok}), (200 if ok else 404)

    @app.post("/api/chats/<chat_id>/messages")
    def post_message(chat_id):
        chat = store.get_chat(chat_id)
        if chat is None:
            return jsonify({"error": "not found"}), 404

        body = request.get_json(force=True, silent=True) or {}
        question = (body.get("message") or "").strip()
        if not question:
            return jsonify({"error": "message required"}), 400

        store.add_message(chat_id, "user", question)

        answerer = get_answerer()
        if answerer is None:
            error_text = (
                "The RAG pipeline isn't reachable right now "
                f"({state['error']}). Make sure Ollama is running and the "
                "Chroma/BM25 indexes have been built, then restart this server."
            )
            store.add_message(chat_id, "assistant", error_text)
            return jsonify({"role": "assistant", "content": error_text, "sources": []}), 502

        try:
            result = answerer.answer(question)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            error_text = f"Something went wrong generating the answer: {exc}"
            store.add_message(chat_id, "assistant", error_text)
            return jsonify({"role": "assistant", "content": error_text, "sources": []}), 500

        message = store.add_message(chat_id, "assistant", result.answer, result.sources)
        return jsonify(message)

    return app


def main():
    args = parse_args()
    app = create_app(args)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
