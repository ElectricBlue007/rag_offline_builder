import json
import subprocess

from agents.chunk_repair_agent import ChunkRepairAgent
from agents.chunk_review_agent import ChunkReviewAgent


def chunk(text, **overrides):
    data = {
        "chunk_id": "chunk_1",
        "text": text,
        "token_estimate": max(1, len(text.split())),
        "heading_path": ["Chonkie", "Installation"],
        "parent_section_id": "sec_1",
        "debug_metadata": {"source_offset": {"start_line": 1, "end_line": 3}},
        "evaluation_metadata": {"chunk_coherence_score": 0.9, "density_score": 0.8},
        "semantic_tags": ["h1:chonkie", "h2:installation"],
    }
    data.update(overrides)
    return data


def repair_text():
    return (
        "passage: Chonkie > Installation\n\n"
        "SemanticChunker is a Chonkie component that uses embeddings to create semantic retrieval chunks. "
        "To install Chonkie for markdown chunking, run pip install chonkie and configure the chunking "
        "pipeline before indexing. This installation context is standalone, searchable, and ready for "
        "embedding because it includes the document path, entity names, action, and retrieval concepts. "
        "The SemanticChunker creates chunk candidates for retrieval workflows and keeps setup guidance "
        "connected to the package installation step."
    )


def approved_repair_text():
    return (
        "passage: Chonkie > Installation\n\n"
        "SemanticChunker is a component that uses embeddings and creates semantic retrieval chunks for Chonkie. "
        "It explains how to install, configure, and use Chonkie in a markdown document retrieval pipeline. "
        "The chunk preserves the Installation heading, the Chonkie entity, and the relationship that "
        "SemanticChunker uses embeddings to prepare searchable chunks. Use pip install chonkie before indexing "
        "documents, then configure the chunking pipeline so retrieval queries can find installation guidance. "
        "This standalone passage includes actionable setup information, a definition, entities, searchable "
        "keywords, and stable context for E5 embedding models."
    )


def test_chunk_repair_agent_replaces_reviewed_chunk_and_reruns_review():
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return approved_repair_text()

    bad_chunk = chunk("Install Chonkie.", token_estimate=8)
    index = {"flat_chunk_view": [bad_chunk]}
    index["chunk_review"] = ChunkReviewAgent().review_index(index, embedding_model="e5").to_dict()

    repaired = ChunkRepairAgent(llm_client=fake_llm, model_name="fake-model").repair_index(index, embedding_model="e5")

    repaired_chunk = repaired["flat_chunk_view"][0]
    report = repaired["chunk_repair_report"]
    before_score = index["chunk_review"]["chunk_reviews"][0]["final_score"]
    after_score = repaired["chunk_review"]["chunk_reviews"][0]["final_score"]

    assert prompts
    assert "Context JSON" in prompts[0]
    assert "Too Small Chunk" in prompts[0] or "Dangerously Small" in prompts[0]
    assert repaired_chunk["text"].startswith("passage:")
    assert "SemanticChunker is a component" in repaired_chunk["text"]
    assert repaired_chunk["token_estimate"] >= 50
    assert repaired_chunk["debug_metadata"]["chunk_repair"]["model"] == "fake-model"
    assert report["attempted_count"] == 1
    assert report["repaired_count"] == 1
    assert report["repairs"][0]["after_score"] == after_score
    assert after_score >= before_score


def test_chunk_repair_agent_skips_rejected_chunks_by_default():
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return repair_text()

    rejected_chunk = chunk("Previous Next Home Back", token_estimate=80)
    index = {"flat_chunk_view": [rejected_chunk]}
    index["chunk_review"] = ChunkReviewAgent().review_index(index, embedding_model="e5").to_dict()

    repaired = ChunkRepairAgent(llm_client=fake_llm).repair_index(index, embedding_model="e5")

    assert calls == []
    assert repaired["flat_chunk_view"][0]["text"] == "Previous Next Home Back"
    assert repaired["chunk_repair_report"]["attempted_count"] == 0
    assert repaired["chunk_repair_report"]["repaired_count"] == 0


def test_chunk_repair_agent_strips_markdown_fence_from_model_output():
    agent = ChunkRepairAgent(llm_client=lambda prompt: f"```markdown\n{approved_repair_text()}\n```")
    index = {"flat_chunk_view": [chunk("Install Chonkie.", token_estimate=8)]}

    repaired = agent.repair_index(index, embedding_model="e5")

    assert repaired["flat_chunk_view"][0]["text"].startswith("passage:")
    assert "```" not in repaired["flat_chunk_view"][0]["text"]


def test_chunk_repair_report_is_json_serializable():
    agent = ChunkRepairAgent(llm_client=lambda prompt: approved_repair_text())
    index = {"flat_chunk_view": [chunk("Install Chonkie.", token_estimate=8)]}

    repaired = agent.repair_index(index, embedding_model="e5")

    json.dumps(repaired["chunk_repair_report"])


def test_ollama_call_forces_utf8_decoding(monkeypatch):
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="متن repaired", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    output = ChunkRepairAgent(model_name="fake-model")._call_model("prompt")

    assert output == "متن repaired"
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"


def test_chunk_repair_agent_retries_invalid_llm_output():
    replies = iter(["Here is the fixed chunk: bad", approved_repair_text()])
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return next(replies)

    index = {"flat_chunk_view": [chunk("Install Chonkie.", token_estimate=8)]}

    repaired = ChunkRepairAgent(llm_client=fake_llm, max_attempts=2).repair_index(index, embedding_model="e5")

    assert len(prompts) == 2
    assert "VALIDATOR FEEDBACK FROM PREVIOUS ATTEMPT" in prompts[1]
    assert repaired["flat_chunk_view"][0]["text"].startswith("passage:")
    assert repaired["chunk_repair_report"]["repaired_count"] == 1


def test_chunk_repair_rolls_back_non_approved_repair():
    index = {"flat_chunk_view": [chunk("Install Chonkie.", token_estimate=8)]}
    original_text = index["flat_chunk_view"][0]["text"]

    repaired = ChunkRepairAgent(
        llm_client=lambda prompt: "passage: Chonkie > Installation\n\nInstall Chonkie with pip."
    ).repair_index(index, embedding_model="e5")

    report = repaired["chunk_repair_report"]
    assert repaired["flat_chunk_view"][0]["text"] == original_text
    assert report["repaired_count"] == 0
    assert report["rolled_back_count"] == 1
    assert report["repairs"][0]["status"] == "ROLLED_BACK"
