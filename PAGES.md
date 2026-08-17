# Page numbers in references / شماره صفحه در ارجاع‌ها

**Version 1.7.0** — fixes the reported bug: chapters and sections were
always right, but the page numbers were wrong.

**نسخه ۱.۷.۰** — رفع باگ گزارش‌شده: فصل و بخش همیشه درست بود، اما شماره
صفحه اشتباه بود.

---

## 1. Why headings were right and pages were wrong

They come from two completely different places. That asymmetry is the whole
explanation.

| | Where it comes from | Reliability |
|---|---|---|
| Chapter / Section | The Markdown heading tree (`#`, `##`, `###`) | **Exact.** The headings are literally in the text. |
| Page number | The original PDF page layout | **Not in the Markdown at all.** It must be recovered. |

When your OCR converts a PDF to Markdown, headings survive as text. Page
boundaries do not — a PDF page break is a layout fact, and Markdown has no
concept of pages. So the page must be reconstructed from something else.

### چرا عنوان‌ها درست بود و صفحه‌ها غلط

این دو از دو جای کاملاً متفاوت می‌آیند. عنوان فصل و بخش مستقیماً از درخت
عنوان‌های Markdown خوانده می‌شود، پس همیشه دقیق است. اما شماره صفحه در
Markdown وجود ندارد؛ مرز صفحه یک واقعیت مربوط به صفحه‌آرایی PDF است و
Markdown مفهوم «صفحه» ندارد. پس باید بازسازی شود.

---

## 2. What was actually broken

Two separate defects. The first one is what you saw.

### Defect A — two different things were being compared

The old code decided a chunk's page like this:

1. Take the pipeline's `chunk_index` (0, 1, 2, ...).
2. Walk the OCR sidecar's `pages[]`, adding up each page's `num_chunks`.
3. Stop at the first page whose running total passes `chunk_index`.

The flaw: **those two numbers count different things.**

- The sidecar's `num_chunks` counts the **OCR tool's** chunks — `Testing.pdf`
  reports **49** of them, spread `[4, 20, 3, 6, 10, 6]` across 6 pages.
- The pipeline builds its **own** semantic chunks from the heading
  structure — **11** for that same file.

So chunk index 10 (the pipeline's last chunk, at the very end of the
document) was compared against a running total that reached 24 by page 2.
Everything landed on the first pages:

```
BEFORE (measured on your real Testing.pdf, 11 chunks, 6 pages)
  chunks  0- 3  ->  page 1
  chunks  4-10  ->  page 2
  pages 3, 4, 5, 6  ->  NEVER cited. Unreachable.
```

Every answer drawn from the second half of that document cited page 1 or 2.
Headings stayed correct throughout, because they never went through this
code at all — exactly the symptom you described.

### Defect B — page markers were read as labels instead of boundaries

When an OCR tool *does* emit inline page markers, a marker means *"a new
page starts here"*. The old code treated it as *"this chunk is on page N"*,
which gets three cases wrong:

- A chunk containing **no marker** got **no page**, even though it obviously
  sits on whatever page was open when it started.
- Text **before** the first marker in a chunk belongs to the **previous**
  page, but was attributed to the newly opened one. This is the classic
  off-by-one at every page break.
- A marker at the **very end** of a chunk announces the **next** chunk's
  page, not the current one's.

### آنچه خراب بود

نقص اول: کد قدیمی `chunk_index` خط لوله را با تعداد chunk های ابزار OCR
مقایسه می‌کرد؛ این دو چیز متفاوت را می‌شمارند (۴۹ در برابر ۱۱). نتیجه این
بود که همه chunk ها روی صفحه‌های اول جمع می‌شدند و صفحه‌های ۳ تا ۶ هرگز
قابل ارجاع نبودند. نقص دوم: نشانگر صفحه یک «مرز» است نه «برچسب»، و به همین
دلیل در هر مرز صفحه خطای یک‌واحدی رخ می‌داد.

---

## 3. How it works now

Stop counting chunks. **Measure position.**

A chunk that starts 70% of the way through the text belongs on whichever
page covers the 70% mark. Pages are weighted by how much text each one
really holds (`token_count` from the sidecar), because pages are not equally
full. This survives the pipeline chunking the text differently from the OCR
tool — which was the entire problem.

```
AFTER (same real file, same 11 chunks)
  chunk  0  ->  pages 1-2      chunk  6  ->  pages 3-4
  chunk  1  ->  page 2         chunk  7  ->  pages 4-5
  chunk  2  ->  page 2         chunk  8  ->  page 5
  chunk  3  ->  page 2         chunk  9  ->  pages 5-6
  chunk  4  ->  pages 2-3      chunk 10  ->  page 6
  chunk  5  ->  page 3

  pages reachable: 1, 2, 3, 4, 5, 6   (was: only 1, 2)
```

The best available method always wins:

| Method | `page_source` | When | Precision |
|---|---|---|---|
| Inline page markers | `markers` | The OCR output contains page boundaries | **Exact** |
| Proportional position | `estimated` | No markers, but the sidecar knows page sizes | **Approximate** |
| Nothing to go on | `none` | Neither is available | No page shown |

A chunk that genuinely straddles a page break now reports **both** pages
(`pages 4-5`). That is the honest answer; silently picking one side is a
coin flip presented as a fact.

---

## 4. An estimate never pretends to be exact

This is the most important design rule in the fix.

**A page number the reader cannot verify must not look identical to one
they can.** A confidently wrong page is worse than an obviously fuzzy one,
because the reader has no way to know they need to check it — and one bad
citation quietly discredits every other citation on the screen.

So estimated pages render with a `~`:

```
Exact      Testing.pdf, page 4 - Chapter: Prompt Engineering > Section: Few-shot
Estimated  Testing.pdf, ~page 4 - Chapter: Prompt Engineering > Section: Few-shot
```

The Query Inspector shows the same marker on each chunk (`~p.4`), and every
source carries a `page_source` field so you can always tell which method
produced the number.

### تخمین هرگز خود را دقیق جا نمی‌زند

صفحه‌ای که کاربر نمی‌تواند بررسی کند نباید شبیه صفحه دقیق دیده شود. پس
صفحه‌های تخمینی با علامت `~` نمایش داده می‌شوند (مثلاً `~page 4`) و فیلد
`page_source` همیشه می‌گوید عدد از کدام روش آمده است.

---

## 5. How to get EXACT pages for your documents

Right now your documents produce `estimated` pages. I verified this
directly: your Markdown files contain **zero** page markers.

```bash
# Returns 0 for every one of your files:
grep -c 'span id="page-' Testing.md
```

That is not a bug in the pipeline — the information was never written into
the Markdown, so nothing downstream can recover it exactly. Estimates are
the best any code can do from the data as it stands.

To upgrade to exact pages, give the pipeline one of these:

**Option 1 — re-run OCR so it emits page markers (best).**
The pipeline already understands five conventions out of the box, so any of
these in your Markdown is enough:

```html
<span id="page-3-0"></span>     <!-- Chandra style, 0-indexed -> page 4 -->
<!-- page: 4 -->
{4}------------------------
--- page 4 ---
[page 4]
```

Markers are stripped from the text before embedding, so they never leak
into the model's context or pollute your vectors.

**Option 2 — keep a sidecar with per-page `token_count`.**
You already have this (`Testing_metadata.json`), which is why estimates are
reasonably tight rather than uniform guesses. More accurate `token_count`
values produce more accurate estimates.

### چگونه صفحه دقیق بگیرید

فایل‌های Markdown شما هیچ نشانگر صفحه‌ای ندارند (با `grep` بررسی شد و صفر
بود)، پس صفحه دقیق از دل آن‌ها قابل استخراج نیست. دو راه: OCR را دوباره
اجرا کنید تا نشانگر صفحه بگذارد (پنج قالب پشتیبانی می‌شود)، یا فایل جانبی
با `token_count` هر صفحه نگه دارید تا تخمین دقیق‌تر شود.

---

## 6. Where the code lives

| File | Role |
|---|---|
| `rag_pipeline/enrichers/page_mapper.py` | **New.** All page arithmetic. Pure stdlib, no I/O. |
| `rag_pipeline/enrichers/metadata_enricher.py` | Two-pass `enrich()`; picks the best method per document. |
| `rag_pipeline/document_reference.py` | Renders `~page N` for estimates. |
| `rag_pipeline/tests/test_page_mapper.py` | 44 tests, including the regression test. |

Why a separate module: page mapping is arithmetic over positions and is
worth testing in isolation, without needing a document, a sidecar file, or
a running pipeline to exercise it.

The old `_estimate_page_numbers` function is **kept but no longer called**,
so its original tests still document the old behaviour for comparison.

### The regression test that locks the bug shut

```python
def test_every_page_is_reachable_not_only_the_first_ones(self):
    chunks = [make_chunk(content="word " * 40, chunk_index=i) for i in range(11)]
    result = self._enrich_with_sidecar(chunks, TESTING_META, stem="Testing")
    cited = {page for chunk in result for page in chunk.metadata["page_numbers"]}
    self.assertEqual(cited, {1, 2, 3, 4, 5, 6})
```

If anyone ever reintroduces chunk-counting, this test fails immediately.

---

## 7. Verify it yourself

```bash
# 44 page-mapping tests
python3 -m unittest rag_pipeline.tests.test_page_mapper -v

# Full suite: 1043 tests, 0 failures
# (9 errors are the optional 'ragas' package not being installed)
cd rag_pipeline && python3 tests/run_tests.py

# See the real before/after on your own fixture
python3 -c "
import json
from rag_pipeline.enrichers.page_mapper import build_page_index, assign_pages_by_length
meta = json.load(open('rag_pipeline/tests/fixtures/Testing_metadata.json'))
index = build_page_index(meta)
print('weights from:', index.weight_source, '| pages:', index.page_numbers)
for i, pages in enumerate(assign_pages_by_length([500] * 11, index)):
    print(' chunk', i, '->', pages)
"
```

After upgrading, **re-run the ingestion pipeline**. Page numbers are stored
in chunk metadata at ingestion time, so already-indexed chunks keep their
old (wrong) pages until they are re-ingested.

### پس از ارتقا حتماً خط لوله را دوباره اجرا کنید؛ شماره صفحه هنگام ایندکس
ذخیره می‌شود، پس chunk های قبلی تا بازایندکس، صفحه اشتباه قدیمی را نگه
می‌دارند.
