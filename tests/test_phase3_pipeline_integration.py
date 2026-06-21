import json

from rag_doc_compiler.pipeline import DocumentCompiler


def test_compiler_exports_semantic_chunk_plan_in_index():
    result = DocumentCompiler().compile_text(
        "# Installation\n"
        "Install the package.\n\n"
        "## Pip\n"
        "pip install package\n"
    )

    plan = result["index"]["semantic_chunk_plan"]
    review = result["index"]["chunk_review"]

    assert result["chunk_plan"].regions
    assert plan["passes"][0]["name"] == "structural_regions"
    assert plan["regions"][0]["heading_path"] == ["Installation"]
    assert review["summary"]["chunk_count"] == len(result["chunks"])
    assert review["chunk_reviews"][0]["chunk_id"] == result["chunks"][0].chunk_id


def test_run_file_writes_chunk_review_report(tmp_path):
    input_path = tmp_path / "input.md"
    output_dir = tmp_path / "compiled"
    input_path.write_text(
        "# Installation\n"
        "Install Chonkie with pip for semantic retrieval chunks.\n",
        encoding="utf-8",
    )

    result = DocumentCompiler().run_file(input_path, output_dir)

    review_path = result["chunk_review_path"]
    assert review_path.name == "chunk_review_report.json"
    assert review_path.exists()

    index = json.loads(result["index_path"].read_text(encoding="utf-8"))
    report = json.loads(review_path.read_text(encoding="utf-8"))
    assert report == index["chunk_review"]
    assert report["summary"]["chunk_count"] == len(result["chunks"])
