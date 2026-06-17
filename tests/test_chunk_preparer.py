from core.chunk_preparer import ChunkPreparer
from core.markdown_normalizer import MarkdownNormalizer
from core.structure_analyzer import StructureAnalyzer


def chunks(markdown):
    normalized = MarkdownNormalizer().normalize(markdown)
    analysis = StructureAnalyzer().analyze(normalized)
    return ChunkPreparer().prepare(analysis, normalized.flags_by_line)


def test_creates_required_chunk_schema_and_adjacency():
    prepared = chunks("# API\nAPI is a technical interface.\n\n## Steps\n- Run setup\n- Configure model\n")
    assert prepared[0].chunk_id == "sec_1::chunk_1::api"
    assert prepared[0].adjacency["previous_chunk_id"] is None
    assert prepared[0].adjacency["next_chunk_id"] == prepared[1].chunk_id
    assert prepared[1].intent == "list"
    assert prepared[1].content_type == "procedural"
    assert 0.0 <= prepared[0].importance_score <= 1.0


def test_records_normalization_flags_in_debug_metadata():
    prepared = chunks("#Title\nBody.\n")
    assert prepared[0].debug_metadata["normalization_flags"] == {1: ["atx_heading_space_inserted"]}


def test_empty_document_has_no_chunks():
    normalized = MarkdownNormalizer().normalize("\n")
    analysis = StructureAnalyzer().analyze(normalized)
    assert ChunkPreparer().prepare(analysis) == []
