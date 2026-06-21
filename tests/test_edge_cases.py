import json

import pytest

from cli.main import main
from core.markdown_normalizer import MarkdownNormalizer
from core.structure_analyzer import StructureAnalyzer
from rag_doc_compiler.pipeline import DocumentCompiler


def test_nested_lists_remain_single_list_block():
    normalized = MarkdownNormalizer().normalize("# A\n- one\n  - nested\n- two\n")
    analysis = StructureAnalyzer().analyze(normalized)
    assert len(analysis.ordered_blocks) == 1
    assert analysis.ordered_blocks[0].node_type == "list"


def test_empty_section_is_allowed_but_not_chunked():
    result = DocumentCompiler().compile_text("# A\n## Empty\n")
    assert result["chunks"] == []
    assert result["index"]["observability"]["section_count"] == 2


def test_cli_run_writes_outputs(tmp_path, capsys):
    input_path = tmp_path / "input.md"
    output_dir = tmp_path / "out"
    input_path.write_text("# A\nRAG is useful.\n", encoding="utf-8")
    assert main(["run", str(input_path), str(output_dir)]) == 0
    assert (output_dir / "clean.md").exists()
    index = json.loads((output_dir / "index.json").read_text(encoding="utf-8"))
    assert index["flat_chunk_view"][0]["intent"] == "definition"
    # Verify CLI outputs the index path in stdout
    assert "index_path" in capsys.readouterr().out


def test_cli_validate_fail_fast_returns_error(tmp_path, capsys):
    input_path = tmp_path / "bad.md"
    input_path.write_text("## Bad\n", encoding="utf-8")
    assert main(["validate", str(input_path)]) == 2
    assert "StructureError" in capsys.readouterr().err
