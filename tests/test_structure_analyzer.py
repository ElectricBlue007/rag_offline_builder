import pytest

from core.markdown_normalizer import MarkdownNormalizer
from core.structure_analyzer import StructureAnalyzer
from rag_doc_compiler.errors import StructureError


def analyze(markdown):
    return StructureAnalyzer().analyze(MarkdownNormalizer().normalize(markdown))


def test_builds_hierarchy_and_flat_map():
    analysis = analyze("# A\nPara.\n\n## B\n- one\n- two\n")
    assert "sec_1" in analysis.flat_map
    assert "sec_2" in analysis.flat_map
    assert [block.node_type for block in analysis.ordered_blocks] == ["paragraph", "list"]
    assert analysis.flat_map["sec_2"].heading_path == ["A", "B"]


def test_detects_tables_and_code_blocks():
    analysis = analyze("# A\n| x | y |\n| --- | --- |\n| 1 | 2 |\n\n```\na\n```\n")
    assert [block.node_type for block in analysis.ordered_blocks] == ["table", "code_block"]


def test_detects_html_table_blocks():
    analysis = analyze("# A\nIntro.<table><tr><td>A</td></tr></table>\n")
    assert [block.node_type for block in analysis.ordered_blocks] == ["paragraph", "table"]


def test_fails_on_heading_level_jump():
    with pytest.raises(StructureError):
        analyze("## Bad jump\nText\n")
