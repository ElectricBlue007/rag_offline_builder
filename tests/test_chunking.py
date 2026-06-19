import json

from rag_doc_compiler.pipeline import DocumentCompiler


def test_final_chunking_runs_after_clean_markdown():
    result = DocumentCompiler().compile_text("#Title\nRAG is useful.\n")
    final_chunks = result["final_chunks"].chunks
    assert result["markdown"].startswith("# Title")
    assert final_chunks[0].chunk_kind == "section"
    assert final_chunks[1].chunk_kind == "semantic"
    assert final_chunks[1].heading_path == ["Title"]
    assert final_chunks[1].original_text == "RAG is useful."


def test_final_chunks_have_no_semantic_line_overlap():
    result = DocumentCompiler().compile_text("# A\nOne.\n\nTwo.\n")
    semantic_chunks = [chunk for chunk in result["final_chunks"].chunks if chunk.chunk_kind == "semantic"]
    seen_lines = set()
    for chunk in semantic_chunks:
        chunk_lines = set(range(chunk.line_range[0], chunk.line_range[1] + 1))
        assert not seen_lines.intersection(chunk_lines)
        seen_lines.update(chunk_lines)


def test_chunks_json_is_exported(tmp_path):
    input_path = tmp_path / "input.md"
    output_dir = tmp_path / "out"
    input_path.write_text("# A\nConfigure the API model pipeline.\n", encoding="utf-8")
    result = DocumentCompiler().run_file(input_path, output_dir)
    chunks = json.loads(result["chunks_path"].read_text(encoding="utf-8"))
    assert chunks["strategy"]["name"] == "hierarchy_aware_no_overlap_v1"
    assert chunks["chunks"][0]["original_text"] == "# A"
    assert chunks["chunks"][1]["content_type"] == "technical"


def test_multilingual_language_metadata_for_spanish_text():
    result = DocumentCompiler().compile_text("# Sistema\nSeleccione informe del sistema.\n")
    semantic = [chunk for chunk in result["final_chunks"].chunks if chunk.chunk_kind == "semantic"][0]
    assert semantic.language["detected_language"] == "es"
    assert "Heading path: Sistema" in semantic.embedding_text
