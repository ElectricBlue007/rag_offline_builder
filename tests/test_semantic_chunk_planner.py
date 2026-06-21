from core.semantic_chunk_planner import SemanticChunkPlanner


def plan(markdown):
    return SemanticChunkPlanner().plan(markdown)


def test_builds_semantic_regions_before_chunking():
    result = plan(
        "# Installation\n"
        "Install the package.\n\n"
        "# Usage\n"
        "Use the package.\n"
    )

    assert [region.heading_path for region in result.regions] == [["Installation"], ["Usage"]]
    assert result.passes[0]["name"] == "structural_regions"


def test_h1_regions_never_merge():
    result = plan(
        "# Installation\n"
        "Python 3.10+\n\n"
        "# API\n"
        "POST /users creates a user.\n"
    )

    assert all(region.merge_policy == "never_merge" for region in result.regions)
    assert all(not region.merge_candidates for region in result.regions)


def test_tiny_h2_sibling_sections_are_merge_candidates():
    result = plan(
        "# Installation\n"
        "Install Chonkie.\n\n"
        "## Pip\n"
        "pip install chonkie\n\n"
        "## UV\n"
        "uv add chonkie\n"
    )

    pip = next(region for region in result.regions if region.heading_path[-1] == "Pip")
    uv = next(region for region in result.regions if region.heading_path[-1] == "UV")

    assert pip.merge_policy == "merge_if_tiny"
    assert uv.region_id in pip.merge_candidates
    assert pip.region_id in uv.merge_candidates


def test_heading_path_and_parent_context_are_injected():
    result = plan(
        "# Chonkie\n"
        "Chonkie is a chunking toolkit.\n\n"
        "## Installation\n"
        "Install Chonkie.\n\n"
        "### Requirements\n"
        "Python 3.10+\n"
    )

    requirements = next(region for region in result.regions if region.heading_path[-1] == "Requirements")

    assert requirements.context_to_inject == ["Chonkie", "Installation", "Requirements"]
    assert "parent_context_required" in requirements.validation_flags


def test_preserves_definition_requirement_procedure_api_config_example_blocks():
    result = plan(
        "# Reference\n"
        "SemanticChunker is a semantic splitting component.\n\n"
        "Requirements:\n"
        "- Python 3.10+\n\n"
        "Step 1: install the package.\n"
        "Step 2: run the command.\n\n"
        "POST /users creates a user.\n\n"
        "config.yaml sets the default model.\n\n"
        "Example: use SemanticChunker with AutoEmbeddings.\n"
    )

    reasons = {
        reason
        for region in result.regions
        for block in region.blocks
        for reason in block.preserve_reasons
    }

    assert "definition_block" in reasons
    assert "requirement_block" in reasons
    assert "procedure_block" in reasons
    assert "api_block" in reasons
    assert "config_block" in reasons
    assert "example_block" in reasons


def test_code_table_list_quote_and_faq_are_anchors():
    result = plan(
        "# Anchors\n"
        "Use this code:\n\n"
        "```python\n"
        "print('hello')\n"
        "```\n\n"
        "| Name | Value |\n"
        "| --- | --- |\n"
        "| model | e5 |\n\n"
        "- Keep list context\n"
        "- Preserve list items\n\n"
        "> Important quote.\n\n"
        "Q: How do I install it?\n"
        "A: Run pip install chonkie.\n"
    )

    region = result.regions[0]
    reasons = {reason for block in region.blocks for reason in block.preserve_reasons}

    assert "code_anchor" in reasons
    assert "table_anchor" in reasons
    assert "list_anchor" in reasons
    assert "quote_anchor" in reasons
    assert "faq_block" in reasons
    assert len(region.anchor_blocks) >= 5


def test_cross_references_entities_aliases_and_scores_are_recorded():
    result = plan(
        "# Semantic Chunking\n"
        "SemanticChunker (semantic) uses AutoEmbeddings. See Installation section.\n"
    )

    region = result.regions[0]
    block = region.blocks[0]

    assert block.aliases["SemanticChunker"] == "semantic"
    assert "SemanticChunker" in block.entities
    assert "AutoEmbeddings" in block.entities
    assert any("Installation" in reference for reference in block.cross_references)
    assert 0.0 <= region.confidence_score <= 1.0
    assert 0.0 <= region.retrieval_score <= 1.0


def test_similarity_valley_creates_split_candidate_without_anchor_breaking():
    result = plan(
        "# Mixed Topic\n"
        "Chonkie chunks markdown content for retrieval.\n\n"
        "SemanticChunker chunks documents by meaning.\n\n"
        "Pizza dough needs flour and water.\n\n"
        "Mozzarella cheese melts on pizza.\n"
    )

    region = result.regions[0]

    assert region.split_candidates
