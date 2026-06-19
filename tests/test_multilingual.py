from api.rag_service import RAGService
from config.settings import ModelSettings
from rag_doc_compiler.pipeline import DocumentCompiler


def test_spanish_query_retrieves_spanish_evidence(tmp_path):
    input_path = tmp_path / "doc.md"
    output_dir = tmp_path / "out"
    input_path.write_text("# Sistema\nSeleccione informe del sistema.\n", encoding="utf-8")
    result = DocumentCompiler().run_file(input_path, output_dir)
    service = RAGService(settings=ModelSettings(embedding_provider="local_hash"), use_llm=False)
    service.load_chunks(str(result["chunks_path"]))
    response = service.ask("informe del sistema")
    assert response["sources"]
    assert "Source:" in response["answer"]
    assert response["plan"]["language"] == "es"
