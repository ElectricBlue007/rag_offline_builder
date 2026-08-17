"""
Tests for enrichers/page_mapper.py -- page attribution.

The bug these lock down: page numbers used to be derived by comparing the
pipeline's chunk_index against the OCR sidecar's per-page chunk counts. Those
count different things (49 OCR chunks vs 11 pipeline chunks for the same
6-page document), so every chunk was squeezed onto the first pages and most
of the document became impossible to cite.
"""

import unittest

from rag_pipeline.enrichers.page_mapper import (
    MarkerInfo,
    PageIndex,
    assign_pages_by_length,
    build_page_index,
    pages_for_span,
    resolve_marker_pages,
)

# Real shape of Testing_metadata.json: 6 pages, token counts per page, and
# 49 OCR chunks that have nothing to do with the pipeline's own chunking.
TESTING_META = {
    "file_name": "Testing.pdf",
    "num_pages": 6,
    "total_chunks": 49,
    "pages": [
        {"page_num": 0, "token_count": 281, "num_chunks": 4},
        {"page_num": 1, "token_count": 819, "num_chunks": 20},
        {"page_num": 2, "token_count": 493, "num_chunks": 3},
        {"page_num": 3, "token_count": 377, "num_chunks": 6},
        {"page_num": 4, "token_count": 908, "num_chunks": 10},
        {"page_num": 5, "token_count": 677, "num_chunks": 6},
    ],
}


class TestBuildPageIndex(unittest.TestCase):

    def test_prefers_token_counts_when_available(self):
        index = build_page_index(TESTING_META)
        self.assertEqual(index.weight_source, "tokens")

    def test_zero_indexed_page_numbers_are_shifted_to_human_numbering(self):
        index = build_page_index(TESTING_META)
        self.assertEqual(index.page_numbers, [1, 2, 3, 4, 5, 6])

    def test_one_indexed_page_numbers_are_left_alone(self):
        meta = {"pages": [{"page_num": 1, "token_count": 10},
                          {"page_num": 2, "token_count": 10}]}
        self.assertEqual(build_page_index(meta).page_numbers, [1, 2])

    def test_falls_back_to_element_counts_when_no_token_counts(self):
        meta = {"pages": [{"page_num": 0, "num_chunks": 3},
                          {"page_num": 1, "num_chunks": 9}]}
        index = build_page_index(meta)
        self.assertEqual(index.weight_source, "counts")
        # Page 1 holds a quarter of the elements, so it ends at 25%.
        self.assertAlmostEqual(index.cumulative[0], 0.25, places=3)

    def test_falls_back_to_uniform_pages_when_only_a_total_is_known(self):
        index = build_page_index({"num_pages": 4})
        self.assertEqual(index.weight_source, "uniform")
        self.assertEqual(index.page_numbers, [1, 2, 3, 4])

    def test_accepts_alternative_total_key_names(self):
        self.assertEqual(build_page_index({"page_count": 3}).total_pages, 3)
        self.assertEqual(build_page_index({"total_pages": 5}).total_pages, 5)

    def test_accepts_page_stats_key_used_by_chandra(self):
        meta = {"page_stats": [{"page": 0, "token_count": 5},
                               {"page": 1, "token_count": 5}]}
        self.assertEqual(build_page_index(meta).page_numbers, [1, 2])

    def test_returns_none_when_the_sidecar_says_nothing_about_pages(self):
        self.assertIsNone(build_page_index({}))
        self.assertIsNone(build_page_index(None))
        self.assertIsNone(build_page_index({"file_name": "x.pdf"}))

    def test_cumulative_ends_at_exactly_one(self):
        self.assertEqual(build_page_index(TESTING_META).cumulative[-1], 1.0)

    def test_string_numbers_in_the_sidecar_are_tolerated(self):
        meta = {"pages": [{"page_num": "0", "token_count": "100"},
                          {"page_num": "1", "token_count": "100"}]}
        self.assertEqual(build_page_index(meta).page_numbers, [1, 2])

    def test_an_empty_page_still_gets_width_so_it_stays_citable(self):
        meta = {"pages": [{"page_num": 0, "token_count": 500},
                          {"page_num": 1, "token_count": 0},
                          {"page_num": 2, "token_count": 500}]}
        index = build_page_index(meta)
        start, end = index.page_span(1)
        self.assertGreater(end, start)

    def test_missing_page_numbers_fall_back_to_list_position(self):
        meta = {"pages": [{"token_count": 10}, {"token_count": 10}]}
        self.assertEqual(build_page_index(meta).page_numbers, [1, 2])


class TestPagesForSpan(unittest.TestCase):

    def setUp(self):
        # Three equal pages: 0-1/3, 1/3-2/3, 2/3-1.
        self.index = PageIndex(
            page_numbers=[1, 2, 3],
            cumulative=[1 / 3, 2 / 3, 1.0],
            weight_source="uniform",
        )

    def test_span_inside_one_page_returns_that_page(self):
        self.assertEqual(pages_for_span(self.index, 0.4, 0.5), [2])

    def test_span_crossing_a_boundary_returns_both_pages(self):
        self.assertEqual(pages_for_span(self.index, 0.3, 0.4), [1, 2])

    def test_span_covering_everything_returns_every_page(self):
        self.assertEqual(pages_for_span(self.index, 0.0, 1.0), [1, 2, 3])

    def test_start_of_document_is_first_page(self):
        self.assertEqual(pages_for_span(self.index, 0.0, 0.1), [1])

    def test_end_of_document_is_last_page(self):
        self.assertEqual(pages_for_span(self.index, 0.95, 1.0), [3])

    def test_zero_width_span_is_treated_as_a_point(self):
        self.assertEqual(pages_for_span(self.index, 0.5, 0.5), [2])

    def test_reversed_span_is_normalised(self):
        self.assertEqual(pages_for_span(self.index, 0.5, 0.3), [1, 2])

    def test_out_of_range_values_are_clamped(self):
        self.assertEqual(pages_for_span(self.index, -5.0, 0.1), [1])
        self.assertEqual(pages_for_span(self.index, 0.9, 99.0), [3])

    def test_empty_index_returns_no_pages(self):
        self.assertEqual(pages_for_span(PageIndex(), 0.1, 0.2), [])


class TestAssignPagesByLength(unittest.TestCase):

    def test_every_page_of_the_document_is_reachable(self):
        """
        THE regression test. Eleven pipeline chunks over a 6-page document
        must be able to cite all six pages; the old code could only reach
        pages 1-2 because it counted the OCR tool's 49 chunks instead.
        """
        index = build_page_index(TESTING_META)
        assigned = assign_pages_by_length([500] * 11, index)
        reached = {page for pages in assigned for page in pages}
        self.assertEqual(reached, {1, 2, 3, 4, 5, 6})

    def test_pages_advance_monotonically_through_the_document(self):
        index = build_page_index(TESTING_META)
        assigned = assign_pages_by_length([300] * 20, index)
        firsts = [pages[0] for pages in assigned]
        self.assertEqual(firsts, sorted(firsts))

    def test_first_chunk_is_on_the_first_page(self):
        index = build_page_index(TESTING_META)
        assigned = assign_pages_by_length([100] * 30, index)
        self.assertEqual(assigned[0][0], 1)

    def test_last_chunk_is_on_the_last_page(self):
        index = build_page_index(TESTING_META)
        assigned = assign_pages_by_length([100] * 30, index)
        self.assertEqual(assigned[-1][-1], 6)

    def test_long_chunks_carry_more_weight_than_short_ones(self):
        """
        A huge first chunk should push everything after it further into the
        document than an equal-sized split would.
        """
        index = build_page_index(TESTING_META)
        equal = assign_pages_by_length([100, 100, 100], index)
        skewed = assign_pages_by_length([5000, 100, 100], index)
        self.assertGreater(skewed[1][0], equal[1][0])

    def test_a_single_chunk_document_spans_all_pages(self):
        index = build_page_index(TESTING_META)
        self.assertEqual(assign_pages_by_length([1000], index), [[1, 2, 3, 4, 5, 6]])

    def test_zero_length_chunks_still_get_pages_in_order(self):
        index = build_page_index(TESTING_META)
        assigned = assign_pages_by_length([0, 0, 0], index)
        self.assertTrue(all(pages for pages in assigned))
        self.assertEqual(assigned[0][0], 1)

    def test_no_chunks_returns_no_assignments(self):
        self.assertEqual(assign_pages_by_length([], build_page_index(TESTING_META)), [])

    def test_no_index_returns_empty_lists_rather_than_guessing(self):
        self.assertEqual(assign_pages_by_length([10, 20], None), [[], []])

    def test_result_length_always_matches_input_length(self):
        index = build_page_index(TESTING_META)
        for count in (1, 2, 7, 40):
            self.assertEqual(len(assign_pages_by_length([50] * count, index)), count)


class TestResolveMarkerPages(unittest.TestCase):

    def test_chunk_with_no_marker_inherits_the_open_page(self):
        resolved = resolve_marker_pages([
            MarkerInfo(pages=[3], starts_with_marker=True),
            MarkerInfo(pages=[]),
        ])
        self.assertEqual(resolved, [[3], [3]])

    def test_text_before_a_marker_belongs_to_the_previous_page(self):
        resolved = resolve_marker_pages([
            MarkerInfo(pages=[1], starts_with_marker=True),
            MarkerInfo(pages=[2], starts_with_marker=False),
        ])
        self.assertEqual(resolved, [[1], [1, 2]])

    def test_marker_at_the_very_start_does_not_borrow_the_previous_page(self):
        resolved = resolve_marker_pages([
            MarkerInfo(pages=[1], starts_with_marker=True),
            MarkerInfo(pages=[2], starts_with_marker=True),
        ])
        self.assertEqual(resolved, [[1], [2]])

    def test_trailing_marker_is_deferred_to_the_next_chunk(self):
        """
        A marker with no text after it announces the NEXT page, so it must
        not be attributed to the chunk that happens to contain it.
        """
        resolved = resolve_marker_pages([
            MarkerInfo(pages=[4, 5], starts_with_marker=True, ends_with_marker=True),
            MarkerInfo(pages=[]),
        ])
        self.assertEqual(resolved, [[4], [5]])

    def test_leading_chunks_with_no_marker_yet_report_no_page(self):
        resolved = resolve_marker_pages([
            MarkerInfo(pages=[]),
            MarkerInfo(pages=[2], starts_with_marker=True),
        ])
        self.assertEqual(resolved, [[], [2]])

    def test_several_markers_in_one_chunk_are_all_reported(self):
        resolved = resolve_marker_pages([
            MarkerInfo(pages=[7, 8, 9], starts_with_marker=True),
        ])
        self.assertEqual(resolved, [[7, 8, 9]])

    def test_pages_are_sorted_and_deduplicated(self):
        resolved = resolve_marker_pages([
            MarkerInfo(pages=[5, 5, 4], starts_with_marker=True),
        ])
        self.assertEqual(resolved, [[4, 5]])

    def test_result_length_matches_input_length(self):
        markers = [MarkerInfo(pages=[]) for _ in range(6)]
        self.assertEqual(len(resolve_marker_pages(markers)), 6)

    def test_empty_input_returns_empty_output(self):
        self.assertEqual(resolve_marker_pages([]), [])

    def test_a_long_run_of_unmarked_chunks_keeps_the_same_page(self):
        markers = [MarkerInfo(pages=[6], starts_with_marker=True)]
        markers += [MarkerInfo(pages=[]) for _ in range(5)]
        self.assertEqual(resolve_marker_pages(markers), [[6]] * 6)


class TestHonestyOfPageReporting(unittest.TestCase):
    """An estimate must be distinguishable from ground truth."""

    def test_estimated_and_exact_paths_are_reported_differently(self):
        from rag_pipeline.document_reference import page_is_estimated

        self.assertTrue(page_is_estimated({"page_source": "estimated"}))
        self.assertFalse(page_is_estimated({"page_source": "markers"}))
        self.assertFalse(page_is_estimated({}))

    def test_estimated_pages_render_with_a_tilde(self):
        from rag_pipeline.document_reference import format_reference

        metadata = {
            "source_pdf": "Testing.pdf",
            "page_numbers": [4],
            "page_source": "estimated",
            "header_texts": ["Prompt Engineering"],
        }
        self.assertIn("~page 4", format_reference(metadata))

    def test_exact_pages_render_without_a_tilde(self):
        from rag_pipeline.document_reference import format_reference

        metadata = {
            "source_pdf": "Testing.pdf",
            "page_numbers": [4],
            "page_source": "markers",
            "header_texts": ["Prompt Engineering"],
        }
        rendered = format_reference(metadata)
        self.assertIn("page 4", rendered)
        self.assertNotIn("~", rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
