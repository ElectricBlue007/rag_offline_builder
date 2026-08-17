# Document references — page, chapter, section

**Version 1.5.0**

---

## The problem this solves

Before this version, an answer looked like this:

```
One-shot prompting gives the model a single worked example [1].

Sources:
  [1] Testing.md — Prompt Engineering > One-shot & few-shot
```

A reader could see that *something* was used, but not **where it is in the
original document**. To verify the claim you had to open the PDF and search
for the text by hand.

Now the same answer looks like this:

```
One-shot prompting gives the model a single worked example
[1] (Testing.pdf, page 3 — Chapter: Prompt Engineering > Section: One-shot & few-shot).

Sources:
  [1] Testing.pdf, page 3 - Chapter: Prompt Engineering > Section: One-shot & few-shot
```

You can now take any citation, open the PDF at that page, and check it.

<div dir="rtl">

## مشکلی که این نسخه حل می‌کند

پیش از این، پاسخ فقط شمارهٔ منبع را نشان می‌داد و خواننده نمی‌دانست آن
مطلب در **کدام صفحه، کدام فصل و کدام بخش** از سند اصلی است. حالا هر ارجاع
محل دقیق خود را در سند همراه دارد و می‌توان آن را راستی‌آزمایی کرد.

</div>

---

## Where the information comes from

Both halves of a reference already existed in the pipeline; they were simply
thrown away before reaching the model.

| Part of the reference | Where it comes from |
|---|---|
| **Page number** | Inline OCR page markers in the Markdown (`<span id="page-11-0">`, `<!-- page: 5 -->`, `{4}------`, `--- Page 5 ---`, `[Page 5]`), extracted by `metadata_enricher.py` and **stripped** from the text so they never pollute embeddings |
| **Chapter / Section / Subsection** | The Markdown heading chain (`#`, `##`, `###`) tracked by `structure_splitter.py` while it walks the document |
| **File name** | The source PDF name from the OCR sidecar `*_metadata.json`, falling back to the Markdown file name |

### Chapter is decided by POSITION, not by `#` depth

Many OCR'd documents start at `##`, or skip from `#` straight to `###`. So
the **outermost heading a chunk sits under is its chapter**, whatever its
hash depth, the next one down is its section, and the one after that is its
subsection. A document that starts at `##` still has chapters.

Levels deeper than three reuse the label `Subsection` rather than inventing
names nobody uses.

---

## What each chunk now stores

`metadata_enricher.py` adds these fields to every chunk's metadata:

| Field | Example |
|---|---|
| `chapter` | `Prompt Engineering` |
| `section` | `One-shot & few-shot` |
| `subsection` | `Worked example` |
| `heading_path` | `Prompt Engineering > One-shot & few-shot` |
| `heading_depth` | `2` |
| `page_display` | `3`, or `4-6` for a chunk spanning pages, or `4, 9` when there is a gap |
| `reference` | `Testing.pdf, page 3 - Chapter: Prompt Engineering > Section: One-shot & few-shot` |

All values are **flat scalars**, because ChromaDB rejects list-valued
metadata. This is enforced by a test.

---

## What the model receives

Each numbered entry in the context block now carries its location on the
header line:

```
[1] Text — Testing.pdf, page 3 — Chapter: Prompt Engineering > Section: One-shot & few-shot
One-shot prompting gives the model a single worked example before the real
question. Few-shot gives several...
```

The words `Chapter:` and `Section:` are deliberate. A model can only report
"chapter X, section Y" if those words appear next to the titles — given a
bare `A > B` it does not know what `A` and `B` are.

The system prompt then instructs the model to repeat that location after
every citation, and to **copy it exactly** rather than guessing a page
number, and to omit any part the header line does not provide.

---

## It works on your existing database

A corpus ingested before this version has no `chapter` or `section` fields —
only the old `breadcrumb` string. Rather than force a re-ingest,
`document_reference.py` **parses the breadcrumb** when the explicit fields
are missing:

```
"# Prompt Engineering > ## Few-shot"  ->  Chapter: Prompt Engineering, Section: Few-shot
```

So references appear immediately on your current `chroma_db`. Re-ingesting
is optional and only makes the fields directly filterable in queries.

Comma-separated strings are also handled, because ChromaDB flattens lists
(`page_numbers=[1,2]` comes back as `"1,2"`).

<div dir="rtl">

### نیازی به ساخت دوبارهٔ پایگاه داده نیست

اگر پایگاه دادهٔ فعلی شما پیش از این نسخه ساخته شده باشد، فیلدهای فصل و بخش
را ندارد؛ اما این ماژول همان رشتهٔ `breadcrumb` قدیمی را تجزیه می‌کند و
ارجاع‌ها بلافاصله کار می‌کنند. ساخت دوبارهٔ ایندکس اختیاری است.

</div>

---

## Where you see it

| Surface | What it shows |
|---|---|
| Web app source chips | `[1] Testing.pdf, page 3 - Chapter: … > Section: …` |
| Query Inspector chunk cards | The same location on **every** candidate — including dropped ones, so you can see whether the right part of the document was found but discarded |
| `run_answer.py` | Full location on each `Sources:` line |
| `run_query.py` | Full location on each retrieval hit |

---

## Settings

`settings.yaml` → `generation.context_builder`:

```yaml
label_heading_levels: true    # "Chapter: X > Section: Y" vs bare "X > Y"
cite_document_location: true  # ask the model to repeat the location per citation
```

Set `cite_document_location: false` if you prefer terse answers that cite
bare numbers. The `Sources:` list and the inspector keep showing full
locations either way.

---

## Tests

`rag_pipeline/tests/test_document_reference.py` — **46 tests** covering the
helpers and all three consumers. The two that matter most:

- `test_legacy_chunk_still_reports_chapter_and_section` — an old database
  keeps working.
- `test_all_values_are_chroma_safe_scalars` — nothing that would make
  ingestion fail.

Run everything:

```bash
cd rag_pipeline && python3 tests/run_tests.py
```
