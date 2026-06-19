import json

from api.batch_pipeline import BatchPipeline
from config.settings import ModelSettings


def test_batch_pipeline_skip_ocr_compiles_markdowns_and_queries(tmp_path):
    raw_dir = tmp_path / "workspace" / "output_markdowns" / "book_markdown"
    raw_dir.mkdir(parents=True)
    (raw_dir / "book.md").write_text("# Manual\nRAG retrieval uses evidence.\n", encoding="utf-8")
    settings = ModelSettings(embedding_provider="local_hash", llm_provider="none")
    result = BatchPipeline(settings=settings).run(
        input_dir=tmp_path / "input",
        workspace_dir=tmp_path / "workspace",
        query="RAG evidence",
        skip_ocr=True,
        use_llm=False,
    )
    corpus_path = tmp_path / "workspace" / "corpus" / "chunks.json"
    assert result["corpus_chunks"] == str(corpus_path)
    assert corpus_path.exists()
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    assert corpus["observability"]["document_count"] == 1
    assert result["response"]["sources"]
