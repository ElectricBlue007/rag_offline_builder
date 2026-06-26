from agents.rule_repair_prompt_agent import RuleRepairPromptAgent


def test_rule_prompt_agent_covers_markdown_validation_and_chunk_rules():
    covered = RuleRepairPromptAgent().covered_rule_ids()

    assert set(range(1, 71)).issubset(covered)
    assert set(range(101, 151)).issubset(covered)
    assert set(range(201, 237)).issubset(covered)
    assert set(range(301, 351)).issubset(covered)


def test_markdown_prompt_is_rule_specific():
    prompt = RuleRepairPromptAgent().markdown_prompt(
        issue={
            "rule_id": 107,
            "rule_name": "Broken Code Block",
            "description": "Code fence is unclosed.",
        },
        context="# API\n```python\nprint('x')\n",
        target_text="```python\nprint('x')\n",
        line_num=2,
        scope="context",
    )

    assert "RULE-SPECIFIC AGENT" in prompt
    assert "Broken Code Block" in prompt
    assert "Repair structural Markdown" in prompt
    assert "balanced code fences" in prompt


def test_chunk_prompt_includes_rule_repair_plan_for_each_issue():
    prompt = RuleRepairPromptAgent().chunk_prompt(
        chunk={
            "chunk_id": "chunk_1",
            "heading_path": ["Prompt Engineering", "Introduction"],
            "parent_section_id": "sec_1",
            "semantic_tags": ["paragraph"],
            "text": "Install Chonkie.",
        },
        review={
            "decision": "REPAIR",
            "final_score": 75,
            "issues": [
                {
                    "rule_id": 311,
                    "rule_name": "Too Small Chunk",
                    "severity": "high",
                    "action": "REPAIR",
                    "message": "Chunk is below minimum semantic mass.",
                },
                {
                    "rule_id": 344,
                    "rule_name": "Prefix Validation",
                    "severity": "medium",
                    "action": "REPAIR",
                    "message": "E5 passage prefix is missing.",
                },
            ],
        },
        embedding_model="e5",
    )

    assert "rule_repair_plan" in prompt
    assert "Too Small Chunk" in prompt
    assert "Prefix Validation" in prompt
    assert "passage:" in prompt
    assert "HARD OUTPUT CONTRACT" in prompt
    assert "maximum_words" in prompt
    assert "target_final_decision" in prompt
    assert "APPROVE" in prompt
    assert "No preface" in prompt
    assert "balanced <table> and </table>" in prompt
    assert "If the chunk cannot be repaired to APPROVE" in prompt


def test_chunk_prompt_feedback_targets_previous_validator_error():
    prompt = RuleRepairPromptAgent().chunk_prompt(
        chunk={
            "chunk_id": "chunk_1",
            "heading_path": ["Doc"],
            "parent_section_id": "sec_1",
            "semantic_tags": ["table"],
            "text": "<table><tr><td>A</td></tr></table>",
        },
        review={
            "decision": "REPAIR",
            "final_score": 75,
            "issues": [
                {
                    "rule_id": 306,
                    "rule_name": "Broken Table",
                    "severity": "high",
                    "action": "REPAIR",
                    "message": "Table is broken.",
                }
            ],
        },
        embedding_model="e5",
        feedback="Repaired chunk has an unbalanced HTML table.",
    )

    assert "VALIDATOR FEEDBACK FROM PREVIOUS ATTEMPT" in prompt
    assert "Repaired chunk has an unbalanced HTML table" in prompt
    assert "Do not discuss this feedback" in prompt
