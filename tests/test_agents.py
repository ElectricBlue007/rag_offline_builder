from api.rag_service import RAGService
from config.settings import ModelSettings
from rag_doc_compiler.pipeline import DocumentCompiler


def test_agent_answer_contains_required_citation(tmp_path):
    input_path = tmp_path / "doc.md"
    output_dir = tmp_path / "out"
    input_path.write_text("# Retrieval\nRAG retrieval uses evidence chunks.\n", encoding="utf-8")
    result = DocumentCompiler().run_file(input_path, output_dir)
    service = RAGService(settings=ModelSettings(embedding_provider="local_hash"), use_llm=False)
    service.load_chunks(str(result["chunks_path"]))
    answer = service.ask("What uses evidence chunks?")["answer"]
    assert "[Source:" in answer
    assert "Retrieval" in answer
