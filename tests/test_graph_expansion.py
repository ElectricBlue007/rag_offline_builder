from adapters.graph_input_adapter import GraphInputAdapter
from core.graph_connector import GraphConnector
from rag_doc_compiler.pipeline import DocumentCompiler


def test_graph_expands_from_semantic_chunk_to_parent_section():
    result = DocumentCompiler().compile_text("# A\nUseful content.\n")
    chunks = [chunk.to_dict() for chunk in result["final_chunks"].chunks]
    edges = GraphInputAdapter().build_edges(chunks)
    semantic_id = [chunk["chunk_id"] for chunk in chunks if chunk["chunk_kind"] == "semantic"][0]
    distances = GraphConnector(edges).expand([semantic_id], depth=1)
    assert any(chunk_id != semantic_id for chunk_id in distances)
