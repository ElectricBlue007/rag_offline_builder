# General questions — breadth & the document map
# پرسش‌های کلی — گستردگی و نقشه سند

Version 1.6.0

---

## The question this answers

> "If I ask a very general question like *what is prompt engineering?*, will
> the model be told that few-shot is one strategy among several?"

Before 1.6.0 the honest answer was **partly**. Now it is **yes, and you can
see exactly how**.

---

## Why a general question used to give a narrow answer

Two separate causes, which need two separate fixes.

### Cause 1 — the five prompt slots were spent on one section

Dense search ranks by similarity. For a broad question, the five most
similar paragraphs are usually five *neighbouring* paragraphs of the same
section:

```
[1] Chapter: Prompt Engineering > Section: Few-shot
[2] Chapter: Prompt Engineering > Section: Few-shot
[3] Chapter: Prompt Engineering > Section: Few-shot
[4] Chapter: Prompt Engineering > Section: Zero-shot
[5] Chapter: Prompt Engineering > Section: Few-shot
```

The model then describes few-shot as if it were the whole topic. That is not
a hallucination — it never saw the rest of the chapter.

### Cause 2 — "one of several" is not written in any chunk

"Few-shot is ONE OF SEVERAL prompting strategies" appears in no paragraph of
your document. It is a property of the **heading tree**. Retrieval returns
passages, so retrieval can never return it — no matter how good the
embedding model is.

---

## Fix 1 — per-section cap (`max_chunks_per_section`)

At most N of the prompt slots may come from the same section. Same example,
with the cap at 2:

```
[1] Chapter: Prompt Engineering > Section: Few-shot
[2] Chapter: Prompt Engineering > Section: Few-shot
[3] Chapter: Prompt Engineering > Section: Zero-shot
[4] Chapter: Prompt Engineering > Section: Chain-of-thought
[5] Chapter: Prompt Engineering > Section: Few-shot     <- backfilled
```

Three guarantees, all covered by tests:

1. **Rank order is preserved.** The best chunk is still `[1]`. Nothing is
   re-sorted; chunks are either taken in order or deferred.
2. **It never sends fewer chunks.** Once every section has hit the cap, the
   deferred chunks backfill the remaining slots. A single-section document
   behaves exactly as it did before.
3. **It is off with one setting.** `max_chunks_per_section: 0`.

---

## Fix 2 — the document map

When a question looks like an overview question, the heading tree is placed
above the numbered sources:

```
Document map -- how the source document is organised. Orientation only:
use it to explain how topics relate and what a chapter covers, but do NOT
cite it and do not state facts from it beyond the structure shown. Cite
only the numbered sources below.
- Testing.pdf > Chapter: Prompt Engineering
  Sections: Zero-shot; One-shot & few-shot; Chain-of-thought; Temperature

[1] Text — Testing.pdf, page 3 — Chapter: Prompt Engineering > Section: One-shot & few-shot
...
```

Note **Temperature**: it appears in the map even though no chunk from it was
retrieved. That is the point — the model can now say "the chapter also
covers temperature" instead of implying few-shot is the whole story.

The map is built from the **whole corpus**, read once per process and
cached, not from the retrieved chunks. A map built from retrieved chunks
only would claim the chapter contains nothing but few-shot: confidently
wrong, which is worse than no map.

The map carries **no `[n]` numbers** and explicitly forbids citing itself,
so it cannot produce a citation you are unable to verify against a page.

---

## When does the map appear? (`document_map: auto`)

`auto` adds the map only for overview questions. The detection is
deliberately conservative and always explains itself.

| Question | Map? | Why |
|---|---|---|
| `what is prompt engineering?` | yes | overview phrasing, short question |
| `explain chunking` | yes | overview phrasing |
| `مهندسی پرامپت چیست؟` | yes | Persian overview phrasing |
| `how do i enable the reranker` | no | "how do i" = specific |
| `what is the error in step 3` | no | "step 3" beats the "what is" opener |
| a 15-word question | no | long questions carry their own qualifiers |

Why a heuristic and not an LLM call: asking a model would add a round trip
to every question and make retrieval non-deterministic and hard to debug.
Being wrong here is cheap — the map is a few lines of orientation text, not
a replacement for real passages.

Other modes: `always` (every question) and `off` (never).

---

## Seeing it in the web app

Open **Why this answer?** on any reply, then the **Prompt** tab. A section
called *General-question handling* shows:

- whether the map was added, **and the reason in plain English and Persian**
- which different sections reached the model
- the exact map text that was sent

If a broad question produced no map, the reason says so — usually
*"no chapter/section headings matched"*, which points at the real cause:
headings.

---

## The one thing that can break this

Both fixes read your document's headings. They work when headings are
**descriptive**:

- Good: `## Prompt Engineering`, `### One-shot & few-shot`
- Bad: `## Chapter 3`, `### 3.2`

With `## Chapter 3`, the map still lists the sections, but a question about
"prompt engineering" cannot match the chapter by title, and the embedding
bridge described in `REFERENCES.md` weakens too. If your OCR output has
numeric-only headings, fixing them in the Markdown is the single highest
value change you can make to retrieval quality.

---

## Settings

```yaml
generation:
  context_builder:
    max_chunks_per_section: 2    # 0 or null disables the cap
    document_map: auto           # auto | always | off
    document_map_max_chapters: 3
    document_map_max_sections: 12
```

These are forwarded at every place the config is built (web app, `run_answer.py`,
`run_evaluation.py`), so the web app, the CLI and RAGAS evaluation all send
the *same* prompt.

---

<div dir="rtl">

## به فارسی

### مسئله

وقتی می‌پرسید «مهندسی پرامپت چیست؟» جستجوی معنایی پنج پاراگراف *شبیه‌ترین* را
برمی‌گرداند و این پنج پاراگراف معمولاً پاراگراف‌های کنار هم از یک بخش هستند.
نتیجه: مدل «few-shot» را طوری توضیح می‌دهد که انگار کل موضوع همین است.

همچنین این حقیقت که «few-shot تنها یکی از چند راهبرد است» در هیچ پاراگرافی
نوشته نشده؛ این اطلاعات در **درخت عناوین** سند وجود دارد، پس بازیابی قطعات
هرگز نمی‌تواند آن را برگرداند.

### دو راه‌حل

۱. **سقف هر بخش** (`max_chunks_per_section`): حداکثر ۲ قطعه از یک بخش وارد
پرامپت می‌شود تا جای بقیه بخش‌ها گرفته نشود. ترتیب رتبه حفظ می‌شود و هرگز
تعداد قطعات ارسالی کمتر از قبل نمی‌شود.

۲. **نقشه سند** (`document_map`): درخت عناوین بالای منابع شماره‌دار قرار
می‌گیرد، مثلاً «فصل: مهندسی پرامپت — بخش‌ها: Zero-shot؛ Few-shot؛ ...».
حتی بخش‌هایی که هیچ قطعه‌ای از آن‌ها بازیابی نشده در نقشه دیده می‌شوند.
نقشه شماره استناد ندارد و مدل صریحاً از استناد به آن منع شده است.

### تشخیص خودکار

حالت `auto` فقط برای پرسش‌های کلی («چیست»، «توضیح بده»، «what is»، «explain»)
نقشه را اضافه می‌کند. پرسش‌های جزئی («چطور فعال کنم»، «مرحله ۳») نقشه
نمی‌گیرند. دلیل تصمیم همیشه در پنل «چرا این پاسخ؟» به دو زبان نمایش داده
می‌شود.

### مهم‌ترین نکته

هر دو قابلیت به عناوین سند شما وابسته‌اند. اگر عنوان‌ها توصیفی باشند
(«مهندسی پرامپت») همه‌چیز کار می‌کند؛ اگر فقط شماره باشند («فصل ۳») تطابق
عنوان با پرسش از دست می‌رود. اصلاح عناوین در فایل Markdown بیشترین تأثیر را
بر کیفیت بازیابی دارد.

</div>
