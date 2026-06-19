from core.embedding_engine import EmbeddingEngine


def test_embedding_is_deterministic_and_dimensioned():
    engine = EmbeddingEngine("generic")
    left = engine.embed_text("Heading path: A\nKeywords: rag\nOriginal text: RAG pipeline")
    right = engine.embed_text("Heading path: A\nKeywords: rag\nOriginal text: RAG pipeline")
    assert left == right
    assert len(left) == 64


def test_empty_embedding_fails_fast():
    engine = EmbeddingEngine("generic")
    try:
        engine.embed_text(" ")
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("Expected empty embedding failure.")
