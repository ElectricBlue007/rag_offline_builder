import json

from agents.chunk_repair_agent import ChunkRepairAgent
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


def test_run_file_writes_chunk_repair_report_when_enabled(tmp_path):
    input_path = tmp_path / "input.md"
    output_dir = tmp_path / "compiled"
    input_path.write_text("# Installation\nInstall Chonkie.\n", encoding="utf-8")

    repair_agent = ChunkRepairAgent(
        llm_client=lambda prompt: (
            "passage: Installation\n\n"
            "SemanticChunker is a Chonkie component that uses embeddings to create semantic retrieval chunks. "
            "Install Chonkie with pip before indexing markdown documents for retrieval. This chunk includes "
            "the Installation heading, the Chonkie entity, the action to install, and searchable concepts for "
            "embedding readiness. SemanticChunker creates chunk candidates and keeps setup context connected "
            "to the document section."
        ),
        model_name="fake-model",
    )

    result = DocumentCompiler(repair_chunks=True, chunk_repair_agent=repair_agent).run_file(input_path, output_dir)

    assert result["chunk_repair_path"].name == "chunk_repair_report.json"
    assert result["chunk_repair_path"].exists()

    index = json.loads(result["index_path"].read_text(encoding="utf-8"))
    repair_report = json.loads(result["chunk_repair_path"].read_text(encoding="utf-8"))
    review_report = json.loads(result["chunk_review_path"].read_text(encoding="utf-8"))

    assert repair_report == index["chunk_repair_report"]
    assert review_report == index["chunk_review"]
    assert repair_report["enabled"] is True
    assert repair_report["attempted_count"] >= 1
