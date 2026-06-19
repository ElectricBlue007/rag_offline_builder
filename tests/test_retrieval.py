from adapters.graph_input_adapter import GraphInputAdapter
from core.embedding_engine import EmbeddingEngine
from core.graph_connector import GraphConnector
from core.retrieval_engine import RetrievalEngine
from rag_doc_compiler.pipeline import DocumentCompiler
from storage.faiss_store import FaissStore


def test_vector_retrieval_returns_relevant_chunk():
    result = DocumentCompiler().compile_text("# API\nConfigure the API model pipeline.\n")
    chunks = [chunk.to_dict() for chunk in result["final_chunks"].chunks]
    graph = GraphConnector(GraphInputAdapter().build_edges(chunks))
    engine = RetrievalEngine(EmbeddingEngine(), FaissStore(), graph)
    engine.index_chunks(chunks)
    retrieved = engine.retrieve("API model pipeline", top_k=2)
    assert retrieved[0]["payload"]["chunk_id"]
    assert "API" in retrieved[0]["payload"]["embedding_text"]


def test_metadata_filtering_by_content_type():
    result = DocumentCompiler().compile_text("# API\nConfigure the API model pipeline.\n")
    chunks = [chunk.to_dict() for chunk in result["final_chunks"].chunks]
    graph = GraphConnector(GraphInputAdapter().build_edges(chunks))
    engine = RetrievalEngine(EmbeddingEngine(), FaissStore(), graph)
    engine.index_chunks(chunks)
    retrieved = engine.retrieve("API model", filters={"content_type": "technical"})
    assert retrieved
    assert all(item["payload"]["content_type"] == "technical" or item["graph_distance"] > 0 for item in retrieved)
