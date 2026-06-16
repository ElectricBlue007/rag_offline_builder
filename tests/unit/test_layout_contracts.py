from __future__ import annotations

import unittest

from core.contracts.layout import LayoutNode, ReadingOrderResult


class TestLayoutContracts(unittest.TestCase):
    def test_layout_node_validation(self) -> None:
        node = LayoutNode(node_id="n1", node_type="heading", text="Title", order_index=0)
        node.validate()
        self.assertEqual(node.debug()["node_type"], "heading")

    def test_reading_order_result_validation(self) -> None:
        result = ReadingOrderResult(
            source_path="/tmp/sample.docx",
            nodes=[LayoutNode(node_id="n1", node_type="paragraph", text="Body", order_index=0)],
        )
        result.validate()


if __name__ == "__main__":
    unittest.main()

