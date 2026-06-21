from core.semantic_chunk_planner import SemanticChunkPlanner


def plan(markdown):
    return SemanticChunkPlanner().plan(markdown)


def region(result, title):
    return next(item for item in result.regions if item.heading_path[-1] == title)


def reasons(result):
    return {
        reason
        for item in result.regions
        for block in item.blocks
        for reason in block.preserve_reasons
    }


def test_rule_201_build_semantic_regions_first():
    result = plan("# Installation\nInstall it.\n\n# Usage\nUse it.\n")

    assert result.passes[0]["name"] == "structural_regions"
    assert [item.heading_path for item in result.regions] == [["Installation"], ["Usage"]]


def test_rule_202_heading_is_soft_boundary_for_h2():
    result = plan("# Guide\nIntro text.\n\n## Pip\nInstall with pip.\n")

    pip = region(result, "Pip")

    assert pip.boundary_strength == "soft"


def test_rule_203_h1_never_merge():
    result = plan("# Installation\nTiny.\n\n# API\nTiny.\n")

    assert all(item.merge_policy == "never_merge" for item in result.regions)
    assert all(not item.merge_candidates for item in result.regions)


def test_rule_204_h2_usually_separate_but_tiny_h2_can_merge():
    result = plan("# Install\nIntro.\n\n## Pip\npip install chonkie\n\n## UV\nuv add chonkie\n")

    pip = region(result, "Pip")
    uv = region(result, "UV")

    assert pip.merge_policy == "merge_if_tiny"
    assert uv.region_id in pip.merge_candidates


def test_rule_205_h3_h4_are_merge_candidates():
    result = plan("# Docs\nIntro.\n\n## Install\nIntro.\n\n### Notes\nTiny note.\n")

    notes = region(result, "Notes")

    assert notes.boundary_strength == "merge_candidate"
    assert notes.merge_policy == "merge_candidate"


def test_rule_206_tiny_section_expansion_is_flagged():
    result = plan("# Install\nIntro.\n\n## Requirements\nPython 3.10+\n")

    requirements = region(result, "Requirements")

    assert "tiny_section_expansion_needed" in requirements.validation_flags


def test_rule_207_parent_expansion_is_planned():
    result = plan("# Chonkie\nIntro.\n\n## Install\nIntro.\n\n### Requirements\nPython 3.10+\n")

    requirements = region(result, "Requirements")

    assert "parent_context_required" in requirements.validation_flags
    assert requirements.context_to_inject == ["Chonkie", "Install", "Requirements"]


def test_rule_208_heading_path_injection_is_present():
    result = plan("# Chonkie\nIntro.\n\n## Installation\nIntro.\n\n### Pip\npip install chonkie\n")

    pip = region(result, "Pip")

    assert pip.context_to_inject == ["Chonkie", "Installation", "Pip"]


def test_rule_209_preserve_definition_blocks():
    result = plan("# Concepts\nSemanticChunker is a semantic chunking component.\n")

    assert "definition_block" in reasons(result)


def test_rule_210_preserve_requirement_blocks():
    result = plan("# Install\nRequirements:\n- Python 3.10+\n- pip\n")

    assert "requirement_block" in reasons(result)


def test_rule_211_preserve_procedure_blocks():
    result = plan("# Install\nStep 1: install package.\nStep 2: run command.\n")

    assert "procedure_block" in reasons(result)


def test_rule_212_preserve_api_blocks():
    result = plan("# API\nPOST /users creates a user.\n")

    assert "api_block" in reasons(result)


def test_rule_213_preserve_config_blocks():
    result = plan("# Config\nconfig.yaml sets the default model.\n")

    assert "config_block" in reasons(result)


def test_rule_214_preserve_examples():
    result = plan("# Examples\nExample: use SemanticChunker with AutoEmbeddings.\n")

    assert "example_block" in reasons(result)


def test_rule_215_code_anchor_rule():
    result = plan("# Code\nExplanation before.\n\n```python\nprint('x')\n```\n\nExplanation after.\n")

    item = result.regions[0]

    assert "code_anchor" in reasons(result)
    assert item.anchor_blocks


def test_rule_216_table_anchor_rule():
    result = plan("# Table\nExplanation.\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n")

    assert "table_anchor" in reasons(result)
    assert result.regions[0].anchor_blocks


def test_rule_217_list_anchor_rule():
    result = plan("# List\nContext.\n\n- one\n- two\n")

    assert "list_anchor" in reasons(result)
    assert result.regions[0].anchor_blocks


def test_rule_218_quote_anchor_rule():
    result = plan("# Quote\n> Important quoted context.\n")

    assert "quote_anchor" in reasons(result)
    assert result.regions[0].anchor_blocks


def test_rule_219_faq_anchor_rule():
    result = plan("# FAQ\nQ: How install?\nA: Run pip install chonkie.\n")

    assert "faq_block" in reasons(result)
    assert result.regions[0].anchor_blocks


def test_rule_220_cross_reference_preservation():
    result = plan("# Usage\nSee Installation section for setup.\n")

    block = result.regions[0].blocks[0]

    assert any("Installation" in reference for reference in block.cross_references)


def test_rule_221_semantic_cohesion_first_uses_valleys():
    result = plan(
        "# Mixed\n"
        "Chonkie chunks markdown for retrieval.\n\n"
        "SemanticChunker chunks meaningfully.\n\n"
        "Pizza dough uses flour.\n\n"
        "Mozzarella melts on pizza.\n"
    )

    assert result.regions[0].split_candidates


def test_rule_222_adaptive_chunk_size_keeps_ideal_token_metadata():
    result = plan("# Long\n" + "word " * 260 + "\n")

    item = result.regions[0]

    assert SemanticChunkPlanner.IDEAL_MIN_TOKENS <= item.token_estimate <= SemanticChunkPlanner.IDEAL_MAX_TOKENS
    assert item.confidence_score > 0.7


def test_rule_223_similarity_valley_detection():
    result = plan(
        "# Valley\n"
        "retrieval chunk markdown context\n\n"
        "retrieval chunk semantic context\n\n"
        "banana orange fruit salad\n"
    )

    assert result.passes[2]["name"] == "semantic_boundaries"
    assert result.passes[2]["split_candidates"] >= 1


def test_rule_224_local_minimum_detection():
    result = plan(
        "# Local Minimum\n"
        "alpha beta gamma\n\n"
        "alpha beta delta\n\n"
        "pizza cheese tomato\n\n"
        "pizza cheese basil\n"
    )

    assert result.regions[0].split_candidates


def test_rule_225_topic_drift_detection():
    result = plan(
        "# Drift\n"
        "Embedding models encode documents.\n\n"
        "Vector retrieval searches documents.\n\n"
        "Pizza recipes use yeast.\n"
    )

    assert result.regions[0].split_candidates


def test_rule_226_chunk_self_sufficiency_score():
    result = plan("# Enough\n" + "This section explains itself with enough details. " * 20 + "\n")

    assert result.regions[0].self_sufficient is True


def test_rule_227_no_orphan_chunks_flag():
    result = plan("# Notes\nTiny.\n")

    assert "orphan_tiny_region" in result.regions[0].validation_flags


def test_rule_228_no_contextless_chunks():
    result = plan("# Context\nText with heading context.\n")

    assert result.regions[0].context_to_inject == ["Context"]
    assert "contextless_region" not in result.regions[0].validation_flags


def test_rule_229_entity_preservation():
    result = plan("# Entities\nChonkie uses SemanticChunker and AutoEmbeddings.\n")

    block = result.regions[0].blocks[0]

    assert "Chonkie" in block.entities
    assert "SemanticChunker" in block.entities
    assert "AutoEmbeddings" in block.entities


def test_rule_230_alias_preservation():
    result = plan("# Alias\nSemanticChunker (semantic) splits by meaning.\n")

    assert result.regions[0].blocks[0].aliases["SemanticChunker"] == "semantic"


def test_rule_231_semantic_density_scoring():
    result = plan("# Density\nUnique semantic retrieval terms appear here.\n")

    assert result.regions[0].density_score > 0
    assert result.regions[0].blocks[0].density_score > 0


def test_rule_232_merge_low_density_blocks():
    result = plan("# Parent\nIntro.\n\n## Note A\nx\n\n## Note B\ny\n")

    note_a = region(result, "Note A")
    note_b = region(result, "Note B")

    assert "low_density_region" in note_a.validation_flags
    assert note_b.region_id in note_a.merge_candidates


def test_rule_233_chunk_confidence_score():
    result = plan("# Confidence\n" + "This is a useful self contained section. " * 12 + "\n")

    assert 0.0 <= result.regions[0].confidence_score <= 1.0


def test_rule_234_retrieval_utility_score():
    result = plan("# Retrieval\nSemanticChunker is used to install and configure retrieval examples.\n")

    assert 0.0 <= result.regions[0].retrieval_score <= 1.0
    assert result.regions[0].retrieval_score > 0.5


def test_rule_235_reject_bad_chunks_for_rechunking():
    result = plan("# Bad\nx\n")

    assert "low_confidence_rechunk_required" in result.regions[0].validation_flags


def test_rule_236_multi_pass_chunking_plan():
    result = plan("# Multi Pass\nContent for planning.\n")

    assert [item["name"] for item in result.passes] == [
        "structural_regions",
        "anchor_detection",
        "semantic_boundaries",
        "optimization",
        "validation",
    ]
