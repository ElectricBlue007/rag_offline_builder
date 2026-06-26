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


def test_merges_sparse_adjacent_paragraphs_within_same_section():
    prepared = chunks(
        "# Credits\n"
        "Alice Example\n\n"
        "Bob Example\n\n"
        "Carol Example\n\n"
        "## Main Content\n"
        "This section contains enough explanation to stand apart from the short credit lines.\n"
    )

    credit_chunks = [chunk for chunk in prepared if chunk.heading_path == ["Credits"]]
    assert len(credit_chunks) == 1
    assert "Alice Example" in credit_chunks[0].text
    assert "Bob Example" in credit_chunks[0].text
    assert "Carol Example" in credit_chunks[0].text
    assert credit_chunks[0].debug_metadata["extraction_method"] == "chunk_preparer::sparse_merge"


def test_merges_sparse_sibling_subsections_with_labels():
    prepared = chunks(
        "# Credits\n"
        "## Reviewers\n"
        "Alice Example\n\n"
        "## Editors\n"
        "Bob Example\n"
    )

    assert len(prepared) == 1
    assert "Reviewers:" in prepared[0].text
    assert "Editors:" in prepared[0].text
    assert "Alice Example" in prepared[0].text
    assert "Bob Example" in prepared[0].text


def test_filters_ocr_boilerplate_and_merges_bridge_paragraphs():
    prepared = chunks(
        "# Prompt Engineering\n"
        "Author: Lee Boonstra\n\n"
        "![Google logo](logo.webp)The Google logo is decorative.\n\n"
        "## Acknowledgements\n"
        "Reviewer Name\n\n"
        "## Introduction\n"
        "Table 1. An example of zero-shot prompting. General prompting / zero shot.\n\n"
        "## One-shot & few-shot\n"
        "When creating prompts for AI models, examples help the model understand the requested output structure.\n"
    )

    assert len(prepared) == 1
    assert "Author:" not in prepared[0].text
    assert "Google logo" not in prepared[0].text
    assert "Reviewer Name" not in prepared[0].text
    assert "Introduction:" in prepared[0].text
    assert "One-shot & few-shot:" in prepared[0].text
