import json

from agents.self_healing_agent import SelfHealingAgent


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_collect_fixable_items_from_nested_qualification_output(tmp_path):
    markdown = tmp_path / "doc.md"
    issues = tmp_path / "result.json"
    markdown.write_text("# Title\n\nBad line\n", encoding="utf-8")
    write_json(
        issues,
        {
            "phase_2_qualification": {
                "issues": [
                    {
                        "rule_id": 101,
                        "rule_name": "Bad Line",
                        "description": "Fix bad line",
                        "location": {"line": 3},
                    }
                ],
                "warnings": [],
            }
        },
    )

    items = SelfHealingAgent().collect_fixable_items(str(markdown), str(issues))

    assert len(items) == 1
    assert items[0]["line"] == 3
    assert "Bad line" in items[0]["context"]


def test_collect_fixable_items_can_include_warnings_by_mode(tmp_path):
    markdown = tmp_path / "doc.md"
    issues = tmp_path / "result.json"
    markdown.write_text("# Title\n\nTiny\n", encoding="utf-8")
    write_json(
        issues,
        {
            "phase_2_qualification": {
                "issues": [],
                "warnings": [
                    {
                        "rule_id": 141,
                        "rule_name": "Below Minimum Semantic Mass",
                        "description": "Tiny block",
                        "context": {"content": "Tiny"},
                    }
                ],
            }
        },
    )

    default_items = SelfHealingAgent().collect_fixable_items(str(markdown), str(issues))
    warning_items = SelfHealingAgent().collect_fixable_items(str(markdown), str(issues), mode="semantic-enhance")

    assert default_items == []
    assert len(warning_items) == 1
    assert warning_items[0]["line"] == 3


def test_dry_run_lists_lines_without_writing_output(tmp_path):
    markdown = tmp_path / "doc.md"
    issues = tmp_path / "result.json"
    output = tmp_path / "fixed.md"
    markdown.write_text("# Title\n\nBad line\n", encoding="utf-8")
    write_json(
        issues,
        {
            "issues": [
                {
                    "rule_id": 101,
                    "description": "Fix bad line",
                    "location": {"line": 3},
                }
            ]
        },
    )

    result = SelfHealingAgent(llm_client=lambda prompt: "Fixed line").heal(
        str(markdown),
        str(issues),
        str(output),
        dry_run=True,
    )

    assert result is None
    assert not output.exists()


def test_heal_uses_injected_llm_and_replaces_only_target_line(tmp_path):
    markdown = tmp_path / "doc.md"
    issues = tmp_path / "result.json"
    output = tmp_path / "fixed.md"
    markdown.write_text("# Title\n\nBad line\n\nKeep me\n", encoding="utf-8")
    write_json(
        issues,
        {
            "issues": [
                {
                    "rule_id": 101,
                    "description": "Fix bad line",
                    "location": {"line": 3},
                }
            ]
        },
    )

    result = SelfHealingAgent(llm_client=lambda prompt: "Fixed line").heal(
        str(markdown),
        str(issues),
        str(output),
        scope="line",
    )

    assert result == str(output)
    assert output.read_text(encoding="utf-8") == "# Title\n\nFixed line\n\nKeep me\n"
