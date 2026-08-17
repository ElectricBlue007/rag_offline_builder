"""
tests/test_document_map.py

Covers the two things that make GENERAL questions work:

  1. diversify_by_section() -- the five prompt slots get spent on different
     sections instead of five neighbouring paragraphs of one section.
  2. the document map -- the heading tree is injected so the model can see
     that few-shot is ONE OF SEVERAL strategies, a fact that is written in
     no chunk.

Plus the auto-detection that decides which questions get a map, and the
failure paths (no headings, unreadable corpus, map turned off), because a
feature that quietly breaks answering would be far worse than no feature.
"""

import unittest

from rag_pipeline.config import ContextBuilderConfig
from rag_pipeline.generation.rag_answerer import RagAnswerer
from rag_pipeline.retrieval.document_map import (
    build_document_map_block,
    build_outline,
    diversify_by_section,
    format_document_map,
    is_broad_question,
    section_key,
    select_chapters,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def meta(section="Few-shot", chapter="Prompt Engineering", page=3, source="Testing.md"):
    """Metadata in the LEGACY breadcrumb-only shape, on purpose: it proves
    everything here still works on a chroma_db built before 1.5.0."""
    crumb = "## " + chapter
    if section:
        crumb += " > ### " + section
    return {
        "source_file": source,
        "page_numbers": [page],
        "breadcrumb": crumb,
        "chunk_type": "text",
    }


class Chunk:
    def __init__(self, chunk_id, section, page=3, chapter="Prompt Engineering"):
        self.chunk_id = chunk_id
        self.text = "Body about " + section + "."
        self.metadata = meta(section, chapter, page)


class FakeChroma:
    def __init__(self, metadatas, explode=False):
        self._metadatas = metadatas
        self.explode = explode
        self.calls = 0

    def list_metadatas(self, limit=5000):
        self.calls += 1
        if self.explode:
            raise RuntimeError("collection unreadable")
        return self._metadatas


class FakeRetriever:
    def __init__(self, chunks, chroma=None):
        self._chunks = chunks
        self.chroma = chroma

    def retrieve(self, query):
        return list(self._chunks)


class FakeLlm:
    def __init__(self):
        self.user_prompt = ""

    def generate(self, system_prompt, user_prompt):
        self.user_prompt = user_prompt
        return "An answer [1]."


# ---------------------------------------------------------------------------
# 1. Which questions are "broad"?
# ---------------------------------------------------------------------------

class TestIsBroadQuestion(unittest.TestCase):
    def test_what_is_question_is_broad(self):
        self.assertTrue(is_broad_question("what is prompt engineering?").is_broad)

    def test_explain_and_overview_are_broad(self):
        self.assertTrue(is_broad_question("explain prompt engineering").is_broad)
        self.assertTrue(is_broad_question("overview of prompting").is_broad)
        self.assertTrue(is_broad_question("tell me about chunking").is_broad)

    def test_persian_overview_question_is_broad(self):
        self.assertTrue(is_broad_question("مهندسی پرامپت چیست؟").is_broad)

    def test_specific_question_is_not_broad(self):
        self.assertFalse(is_broad_question("how do i enable the reranker").is_broad)

    def test_specific_signal_beats_broad_phrasing(self):
        # Opens with "what is", but it is asking about one concrete thing.
        verdict = is_broad_question("what is the error in step 3")
        self.assertFalse(verdict.is_broad)
        self.assertIn("step 3", verdict.matched)

    def test_long_question_is_treated_as_specific(self):
        verdict = is_broad_question(
            "what is the recommended temperature setting when generating "
            "summaries of long legal documents for review"
        )
        self.assertFalse(verdict.is_broad)
        self.assertIn("long", verdict.reason_en)

    def test_empty_question_is_not_broad(self):
        self.assertFalse(is_broad_question("").is_broad)
        self.assertFalse(is_broad_question(None).is_broad)

    def test_verdict_always_explains_itself_in_both_languages(self):
        for question in ["what is prompt engineering?", "how do i run it", ""]:
            verdict = is_broad_question(question)
            self.assertTrue(verdict.reason_en.strip())
            self.assertTrue(verdict.reason_fa.strip())

    def test_somewhat_isolated_does_not_trigger_what_is(self):
        # Word-boundary anchoring: 'somewhat isolated' contains 'what is'.
        self.assertFalse(is_broad_question("somewhat isolated cases").is_broad)


# ---------------------------------------------------------------------------
# 2. Section identity
# ---------------------------------------------------------------------------

class TestSectionKey(unittest.TestCase):
    def test_key_is_chapter_plus_section(self):
        self.assertEqual(
            section_key(meta("Few-shot")), "Prompt Engineering > Few-shot"
        )

    def test_hashes_are_stripped(self):
        self.assertNotIn("#", section_key(meta("Few-shot")))

    def test_subsections_of_one_section_share_a_key(self):
        deep = meta("Few-shot")
        deep["breadcrumb"] += " > #### Worked example"
        self.assertEqual(section_key(deep), section_key(meta("Few-shot")))

    def test_chapter_only_chunk_has_its_own_key(self):
        self.assertEqual(section_key(meta(section="")), "Prompt Engineering")

    def test_falls_back_to_source_file_when_no_headings(self):
        self.assertEqual(section_key({"source_file": "Notes.md"}), "Notes.md")

    def test_empty_metadata_is_safe(self):
        self.assertEqual(section_key({}), "(unknown)")
        self.assertEqual(section_key(None), "(unknown)")


# ---------------------------------------------------------------------------
# 3. Diversity cap
# ---------------------------------------------------------------------------

class TestDiversifyBySection(unittest.TestCase):
    def setUp(self):
        # What a broad query really returns: four neighbouring few-shot
        # paragraphs crowding out the rest of the chapter.
        self.chunks = [
            Chunk("c1", "Few-shot"),
            Chunk("c2", "Few-shot"),
            Chunk("c3", "Few-shot"),
            Chunk("c4", "Zero-shot"),
            Chunk("c5", "Few-shot"),
            Chunk("c6", "Chain-of-thought"),
        ]

    def sections(self, chunks):
        return [section_key(c.metadata).split(" > ")[-1] for c in chunks]

    def test_plain_truncation_would_show_only_two_sections(self):
        # The behaviour we are fixing, asserted so the fix has a baseline.
        self.assertEqual(len(set(self.sections(self.chunks[:5]))), 2)

    def test_cap_increases_the_number_of_sections_shown(self):
        picked = diversify_by_section(self.chunks, 5, 2)
        self.assertEqual(len(set(self.sections(picked))), 3)

    def test_never_returns_fewer_chunks_than_truncation(self):
        picked = diversify_by_section(self.chunks, 5, 2)
        self.assertEqual(len(picked), 5)

    def test_top_ranked_chunk_is_still_first(self):
        picked = diversify_by_section(self.chunks, 5, 2)
        self.assertEqual(picked[0].chunk_id, "c1")

    def test_diverse_picks_come_first_then_backfill_each_in_rank_order(self):
        # The contract is deliberate: the capped, diverse selection takes the
        # early citation slots (c1, c2, c4, c6 — in rank order), and the
        # over-cap chunks that were skipped are appended afterwards (c3, c5 —
        # also in rank order) only to fill leftover room. Nothing is sorted
        # or reshuffled; chunks are either taken in order or deferred.
        picked_ids = [c.chunk_id for c in diversify_by_section(self.chunks, 6, 2)]
        self.assertEqual(picked_ids[:4], ["c1", "c2", "c4", "c6"])
        self.assertEqual(picked_ids[4:], ["c3", "c5"])

    def test_capped_chunks_are_backfilled_not_dropped(self):
        picked_ids = [c.chunk_id for c in diversify_by_section(self.chunks, 6, 2)]
        self.assertEqual(len(picked_ids), 6)
        self.assertEqual(set(picked_ids), {"c1", "c2", "c3", "c4", "c5", "c6"})

    def test_zero_disables_the_cap(self):
        picked = diversify_by_section(self.chunks, 5, 0)
        self.assertEqual([c.chunk_id for c in picked], ["c1", "c2", "c3", "c4", "c5"])

    def test_none_disables_the_cap(self):
        picked = diversify_by_section(self.chunks, 5, None)
        self.assertEqual([c.chunk_id for c in picked], ["c1", "c2", "c3", "c4", "c5"])

    def test_single_section_document_is_unaffected(self):
        only = [Chunk("a", "Few-shot"), Chunk("b", "Few-shot"), Chunk("c", "Few-shot")]
        picked = diversify_by_section(only, 2, 1)
        self.assertEqual([c.chunk_id for c in picked], ["a", "b"])

    def test_limit_zero_returns_nothing(self):
        self.assertEqual(diversify_by_section(self.chunks, 0, 2), [])

    def test_empty_input_is_safe(self):
        self.assertEqual(diversify_by_section([], 5, 2), [])

    def test_works_on_dict_chunks_too(self):
        dicts = [{"metadata": meta("Few-shot")}, {"metadata": meta("Few-shot")},
                 {"metadata": meta("Zero-shot")}]
        picked = diversify_by_section(dicts, 2, 1)
        self.assertEqual(
            [section_key(c["metadata"]) for c in picked],
            ["Prompt Engineering > Few-shot", "Prompt Engineering > Zero-shot"],
        )


# ---------------------------------------------------------------------------
# 4. Outline building
# ---------------------------------------------------------------------------

class TestBuildOutline(unittest.TestCase):
    def test_groups_sections_under_their_chapter(self):
        outline = build_outline([meta("Few-shot"), meta("Zero-shot")])
        self.assertEqual(
            outline.chapters[("Testing.md", "Prompt Engineering")],
            ["Few-shot", "Zero-shot"],
        )

    def test_duplicate_sections_appear_once(self):
        outline = build_outline([meta("Few-shot")] * 5)
        self.assertEqual(
            outline.chapters[("Testing.md", "Prompt Engineering")], ["Few-shot"]
        )

    def test_document_order_is_preserved(self):
        outline = build_outline([meta("Zero-shot"), meta("Few-shot"), meta("Zero-shot")])
        self.assertEqual(
            outline.chapters[("Testing.md", "Prompt Engineering")],
            ["Zero-shot", "Few-shot"],
        )

    def test_separate_chapters_stay_separate(self):
        outline = build_outline(
            [meta("Few-shot"), meta("Docker", chapter="Deployment")]
        )
        self.assertEqual(len(outline.chapters), 2)

    def test_chunks_without_headings_are_skipped(self):
        outline = build_outline([{"source_file": "x.md"}, meta("Few-shot")])
        self.assertEqual(len(outline.chapters), 1)

    def test_non_dict_entries_are_ignored(self):
        outline = build_outline([None, "junk", 42, meta("Few-shot")])
        self.assertEqual(len(outline.chapters), 1)

    def test_empty_input_is_falsy(self):
        self.assertFalse(build_outline([]))
        self.assertFalse(build_outline(None))

    def test_reads_structured_fields_when_present(self):
        # The 1.5.0 shape, not a breadcrumb.
        structured = {
            "source_file": "Guide.md",
            "header_texts": ["Prompting", "Few-shot"],
            "header_levels": [2, 3],
        }
        outline = build_outline([structured])
        self.assertEqual(outline.chapters[("Guide.md", "Prompting")], ["Few-shot"])


# ---------------------------------------------------------------------------
# 5. Chapter selection and rendering
# ---------------------------------------------------------------------------

class TestSelectAndFormat(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            meta("Zero-shot"), meta("Few-shot"), meta("Chain-of-thought"),
            meta("Docker", chapter="Deployment"),
        ]
        self.outline = build_outline(self.corpus)

    def test_chapter_of_a_retrieved_chunk_is_selected(self):
        selected = select_chapters(self.outline, "anything", [meta("Few-shot")])
        self.assertEqual(selected[0][0][1], "Prompt Engineering")

    def test_chapter_matching_question_wording_is_selected(self):
        selected = select_chapters(self.outline, "tell me about deployment", [])
        self.assertEqual([key[1] for key, _ in selected], ["Deployment"])

    def test_retrieved_chapter_comes_before_title_match(self):
        selected = select_chapters(
            self.outline, "deployment overview", [meta("Few-shot")]
        )
        self.assertEqual([key[1] for key, _ in selected][0], "Prompt Engineering")

    def test_max_chapters_is_respected(self):
        selected = select_chapters(
            self.outline, "prompt engineering and deployment", [], max_chapters=1
        )
        self.assertEqual(len(selected), 1)

    def test_no_match_selects_nothing(self):
        self.assertEqual(select_chapters(self.outline, "zzz unrelated", []), [])

    def test_empty_outline_selects_nothing(self):
        self.assertEqual(select_chapters(build_outline([]), "anything", []), [])

    def test_rendered_map_lists_sibling_sections(self):
        block = build_document_map_block(
            "what is prompt engineering", self.corpus, [meta("Few-shot")]
        )
        for section in ["Zero-shot", "Few-shot", "Chain-of-thought"]:
            self.assertIn(section, block)

    def test_rendered_map_forbids_citing_itself(self):
        block = build_document_map_block(
            "what is prompt engineering", self.corpus, [meta("Few-shot")]
        )
        self.assertIn("do NOT cite", block)
        self.assertNotIn("[1]", block)

    def test_map_names_the_chapter(self):
        block = build_document_map_block(
            "what is prompt engineering", self.corpus, [meta("Few-shot")]
        )
        self.assertIn("Chapter: Prompt Engineering", block)

    def test_section_list_is_capped_and_reports_the_remainder(self):
        many = [meta("S" + str(i)) for i in range(20)]
        block = build_document_map_block(
            "what is prompt engineering", many, [], max_sections_per_chapter=5
        )
        self.assertIn("(+15 more)", block)

    def test_nothing_selected_renders_empty_string(self):
        self.assertEqual(format_document_map([]), "")


# ---------------------------------------------------------------------------
# 6. End to end through RagAnswerer
# ---------------------------------------------------------------------------

class TestAnswererBreadth(unittest.TestCase):
    def build(self, document_map="auto", max_per_section=2):
        chunks = [
            Chunk("c1", "Few-shot"), Chunk("c2", "Few-shot"),
            Chunk("c3", "Few-shot"), Chunk("c4", "Zero-shot"),
            Chunk("c5", "Few-shot"), Chunk("c6", "Chain-of-thought"),
        ]
        corpus = [meta("Zero-shot"), meta("Few-shot"), meta("Chain-of-thought"),
                  meta("Temperature")]
        self.chroma = FakeChroma(corpus)
        self.llm = FakeLlm()
        cfg = ContextBuilderConfig(
            max_chunks=5,
            max_chunks_per_section=max_per_section,
            document_map=document_map,
        )
        return RagAnswerer(
            FakeRetriever(chunks, self.chroma), self.llm, context_config=cfg
        )

    def test_broad_question_gets_the_map(self):
        result = self.build().answer("what is prompt engineering?")
        self.assertTrue(result.breadth["included"])
        self.assertIn("Document map", result.context_used)

    def test_map_reaches_the_actual_prompt(self):
        answerer = self.build()
        answerer.answer("what is prompt engineering?")
        self.assertIn("Document map", self.llm.user_prompt)

    def test_map_names_a_section_that_was_never_retrieved(self):
        # The whole point: "Temperature" is in no retrieved chunk, so only
        # the map can tell the model the chapter also covers it.
        result = self.build().answer("what is prompt engineering?")
        self.assertIn("Temperature", result.document_map)

    def test_specific_question_gets_no_map(self):
        result = self.build().answer("how do i enable the reranker")
        self.assertFalse(result.breadth["included"])
        self.assertNotIn("Document map", result.context_used)

    def test_off_mode_never_adds_a_map(self):
        result = self.build(document_map="off").answer("what is prompt engineering?")
        self.assertFalse(result.breadth["included"])
        self.assertIn("turned off", result.breadth["reason_en"])

    def test_always_mode_adds_a_map_to_a_specific_question(self):
        result = self.build(document_map="always").answer("how do i enable the reranker")
        self.assertTrue(result.breadth["included"])

    def test_breadth_always_explains_the_decision(self):
        for question in ["what is prompt engineering?", "how do i enable the reranker"]:
            result = self.build().answer(question)
            self.assertTrue(result.breadth["reason_en"].strip())
            self.assertTrue(result.breadth["reason_fa"].strip())

    def test_diversity_applies_to_the_answer_sources(self):
        result = self.build().answer("what is prompt engineering?")
        sections = {source["section"] for source in result.sources}
        self.assertEqual(len(sections), 3)

    def test_citation_numbers_still_match_the_context_order(self):
        result = self.build().answer("what is prompt engineering?")
        self.assertEqual(
            [source["citation_number"] for source in result.sources], [1, 2, 3, 4, 5]
        )
        for source in result.sources:
            marker = "[" + str(source["citation_number"]) + "]"
            self.assertIn(marker, result.context_used)

    def test_corpus_outline_is_read_once_and_reused(self):
        answerer = self.build()
        answerer.answer("what is prompt engineering?")
        answerer.answer("what is chunking?")
        self.assertEqual(self.chroma.calls, 1)

    def test_unreadable_corpus_does_not_break_answering(self):
        chunks = [Chunk("c1", "Few-shot"), Chunk("c2", "Zero-shot")]
        answerer = RagAnswerer(
            FakeRetriever(chunks, FakeChroma([], explode=True)),
            FakeLlm(),
            context_config=ContextBuilderConfig(max_chunks=5),
        )
        result = answerer.answer("what is prompt engineering?")
        self.assertEqual(result.answer, "An answer [1].")
        self.assertEqual(len(result.sources), 2)

    def test_retriever_without_chroma_still_answers(self):
        answerer = RagAnswerer(
            FakeRetriever([Chunk("c1", "Few-shot")]),
            FakeLlm(),
            context_config=ContextBuilderConfig(max_chunks=5),
        )
        result = answerer.answer("what is prompt engineering?")
        self.assertEqual(len(result.sources), 1)

    def test_documents_without_headings_report_why_there_is_no_map(self):
        flat = [{"source_file": "Notes.md", "page_numbers": [1]}]

        class FlatChunk:
            chunk_id = "f1"
            text = "Some text."
            metadata = flat[0]

        answerer = RagAnswerer(
            FakeRetriever([FlatChunk()], FakeChroma(flat)),
            FakeLlm(),
            context_config=ContextBuilderConfig(max_chunks=5),
        )
        result = answerer.answer("what is prompt engineering?")
        self.assertFalse(result.breadth["included"])
        self.assertIn("headings", result.breadth["reason_en"])

    def test_no_results_answer_has_empty_breadth(self):
        answerer = RagAnswerer(
            FakeRetriever([]), FakeLlm(), context_config=ContextBuilderConfig()
        )
        result = answerer.answer("what is prompt engineering?")
        self.assertEqual(result.document_map, "")
        self.assertEqual(result.breadth, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
