from agents.chunk_review_agent import ChunkReviewAgent


def chunk(text, **overrides):
    data = {
        "chunk_id": "chunk_1",
        "text": text,
        "token_estimate": max(1, len(text.split())),
        "heading_path": ["Chonkie", "Installation"],
        "parent_section_id": "sec_1",
        "debug_metadata": {"source_offset": {"start_line": 1, "end_line": 1}},
        "evaluation_metadata": {"chunk_coherence_score": 0.9, "density_score": 0.8},
        "semantic_tags": ["h1:chonkie", "h2:installation"],
    }
    data.update(overrides)
    return data


def review_one(data, embedding_model="bge"):
    return ChunkReviewAgent().review_chunks([data], embedding_model=embedding_model)[0]


def rule_ids(result):
    return {issue.rule_id for issue in result.issues}


def test_layer_1_structural_rules_301_to_310():
    agent = ChunkReviewAgent()

    assert 301 in rule_ids(review_one(chunk("", token_estimate=0)))
    assert 302 in rule_ids(review_one(chunk("-----")))
    assert 303 in rule_ids(review_one(chunk("## Installation")))
    assert 304 in rule_ids(review_one(chunk("[broken link](url")))
    assert 305 in rule_ids(review_one(chunk("```python\nprint('x')\n")))
    assert 306 in rule_ids(review_one(chunk("| A | B |\n| --- | --- |\n| 1 | 2 | 3 |")))
    assert 307 in rule_ids(review_one(chunk("-missing-space")))
    assert 308 in rule_ids(review_one(chunk("Install Chonkie with pip.", heading_path=[])))
    assert 309 in rule_ids(review_one(chunk("Install Chonkie with pip.", parent_section_id=None)))
    assert 310 in rule_ids(review_one(chunk("Install Chonkie with pip.", debug_metadata={})))

    report = agent.review_index({"flat_chunk_view": [chunk("Install Chonkie with pip.")]}, embedding_model="bge")
    assert report.summary["chunk_count"] == 1


def test_layer_2_token_rules_311_to_315():
    too_small = review_one(chunk("Install Chonkie with pip.", token_estimate=30))
    dangerous = review_one(chunk("Tiny.", token_estimate=10))
    too_large = review_one(chunk("word " * 900, token_estimate=900))
    very_large = review_one(chunk("word " * 1300, token_estimate=1300))
    low_density = review_one(chunk("word " + ("-" * 500), token_estimate=80))

    assert 311 in rule_ids(too_small)
    assert 312 in rule_ids(dangerous)
    assert 313 in rule_ids(too_large)
    assert 314 in rule_ids(very_large)
    assert 315 in rule_ids(low_density)


def test_layer_3_semantic_rules_316_to_325():
    low_semantic = review_one(chunk("x y z " * 30, token_estimate=90))
    useful = review_one(
        chunk(
            "SemanticChunker is a component that explains how to install and configure chunking. "
            "It enables retrieval because it preserves semantic boundaries.",
            token_estimate=80,
        )
    )
    not_answerable = review_one(chunk("blue red green yellow purple", token_estimate=80))
    dependent = review_one(chunk("This uses that and it follows the previous same below.", token_estimate=80))
    multi_topic = review_one(
        chunk(
            "Install the python package with pip. POST /users returns an API response. "
            "Embedding retrieval chunks semantic vectors. Pizza recipe uses cheese.",
            token_estimate=100,
        )
    )
    weak_cohesion = review_one(
        chunk(
            "Install the package. Pizza cheese melts. Vector search retrieves chunks.",
            token_estimate=80,
            evaluation_metadata={"chunk_coherence_score": 0.2, "density_score": 0.7},
        )
    )

    assert 316 in rule_ids(low_semantic)
    assert {317, 318, 319, 320}.issubset(rule_ids(low_semantic))
    assert 321 in rule_ids(not_answerable)
    assert 322 in rule_ids(review_one(chunk("Tiny.", token_estimate=10)))
    assert 323 in rule_ids(dependent)
    assert 324 in rule_ids(multi_topic)
    assert 325 in rule_ids(weak_cohesion)
    assert useful.semantic_score > low_semantic.semantic_score


def test_layer_4_retrieval_rules_326_to_332():
    empty_info = review_one(chunk("blue red green yellow purple", token_estimate=80))
    no_entities = review_one(chunk("install and configure the semantic retrieval chunk", token_estimate=80))
    useful = review_one(
        chunk(
            "SemanticChunker is a tool that uses embeddings and creates semantic retrieval chunks.",
            token_estimate=80,
        )
    )

    assert 326 in rule_ids(empty_info)
    assert 327 in rule_ids(empty_info)
    assert 328 in rule_ids(empty_info)
    assert 329 in rule_ids(no_entities)
    assert 330 in rule_ids(empty_info)
    assert 331 in rule_ids(empty_info)
    assert 332 in rule_ids(empty_info)
    assert useful.retrieval_score > empty_info.retrieval_score


def test_layer_5_noise_rules_333_to_339():
    navigation = review_one(chunk("Previous Next Home Back", token_estimate=80))
    badge = review_one(chunk("![build](https://img.shields.io/badge/build-pass.svg)", token_estimate=80))
    footer = review_one(chunk("Copyright 2026. All rights reserved.", token_estimate=80))
    boilerplate = review_one(chunk("The Google logo is an abstract geometric shape.", token_estimate=80))
    license_chunk = review_one(chunk("MIT License permission is hereby granted.", token_estimate=80))
    repeated_reviews = ChunkReviewAgent().review_chunks(
        [
            chunk("SemanticChunker is useful for retrieval.", chunk_id="a", token_estimate=80),
            chunk("SemanticChunker is useful for retrieval.", chunk_id="b", token_estimate=80),
        ],
        embedding_model="bge",
    )
    near_duplicate_reviews = ChunkReviewAgent().review_chunks(
        [
            chunk("SemanticChunker is useful for semantic retrieval chunks.", chunk_id="a", token_estimate=80),
            chunk("SemanticChunker is useful for semantic retrieval chunk.", chunk_id="b", token_estimate=80),
        ],
        embedding_model="bge",
    )

    assert 333 in rule_ids(navigation)
    assert 334 in rule_ids(badge)
    assert 335 in rule_ids(footer)
    assert 336 in rule_ids(boilerplate)
    assert 337 in rule_ids(license_chunk)
    assert 338 in rule_ids(repeated_reviews[0])
    assert 339 in rule_ids(near_duplicate_reviews[0])


def test_layer_6_context_rules_340_to_343():
    missing_heading = review_one(chunk("Install Chonkie.", heading_path=[]))
    missing_section = review_one(chunk("Install Chonkie.", parent_section_id=None))
    cross_ref = review_one(
        chunk(
            "See Installation section for setup.",
            debug_metadata={"source_offset": {"start_line": 1, "end_line": 1}},
            token_estimate=80,
        )
    )

    assert 340 in rule_ids(missing_heading)
    assert 342 in rule_ids(missing_heading)
    assert 341 in rule_ids(missing_section)
    assert 343 in rule_ids(cross_ref)


def test_layer_7_embedding_rules_344_to_350():
    missing_prefix = review_one(chunk("Install Chonkie with pip.", token_estimate=80), embedding_model="e5")
    unknown_language = review_one(chunk("12345 !!!", token_estimate=80), embedding_model="bge")
    decomposed = review_one(chunk("Cafe\u0301 installs Chonkie.", token_estimate=80), embedding_model="bge")
    mojibake = review_one(chunk("AI â€“ prompt engineering with Chonkie.", token_estimate=80), embedding_model="bge")
    mixed = review_one(chunk("Install Chonkie برای retrieval.", token_estimate=80), embedding_model="bge")
    costly = review_one(chunk("word " * 900, token_estimate=900), embedding_model="bge")
    low_ready = review_one(chunk("x", token_estimate=5), embedding_model="e5")

    assert 344 in rule_ids(missing_prefix)
    assert 345 in rule_ids(unknown_language)
    assert 346 in rule_ids(decomposed)
    assert 347 in rule_ids(mojibake)
    assert 348 in rule_ids(mixed)
    assert 349 in rule_ids(costly)
    assert 350 in rule_ids(low_ready)


def test_decision_thresholds_and_golden_scores():
    approved = review_one(
        chunk(
            "passage: SemanticChunker is a component that uses embeddings to create semantic retrieval chunks. "
            "It explains how to install, configure, and use Chonkie for document retrieval with stable context.",
            token_estimate=120,
        ),
        embedding_model="e5",
    )
    rechunk = review_one(chunk("Install Chonkie.", token_estimate=30), embedding_model="bge")
    reject = review_one(chunk("", token_estimate=0), embedding_model="bge")

    assert approved.self_sufficiency_score >= 75
    assert approved.retrieval_score >= 75
    assert approved.embedding_stability_score >= 75
    assert approved.final_score >= 75
    assert rechunk.decision in {"REPAIR", "RECHUNK", "REJECT"}
    assert reject.decision == "REJECT"
