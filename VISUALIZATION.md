# Query Inspector — “Why this answer?”

<div dir="rtl">

# بازرس پرس‌وجو — «چرا این پاسخ؟»

</div>

When you ask a question in the web app, you now get a panel under every answer
that shows **exactly which chunks were found, which of them actually reached
the model, and the literal prompt the model received**.

<div dir="rtl">

حالا زیر هر پاسخ در برنامهٔ وب، پنلی می‌بینید که نشان می‌دهد **دقیقاً چه قطعاتی پیدا شد، کدام‌یک واقعاً به مدل رسید، و متن دقیقی که به مدل داده شد چه بود**.

</div>

---

## 1. How to use it · روش استفاده

```bash
cd your_project
python webapp/app.py
# open http://localhost:5000
```

1. The **Inspect · بازرسی** switch in the top-right is **on by default**.
2. Ask a question.
3. Under the answer, click **🔍 Why this answer? · چرا این پاسخ؟**

No model running yet? Click **🔍 See example inspector · نمونه** in the sidebar.
It loads a realistic example so you can see the whole visualization without
Ollama, Chroma or BM25.

<div dir="rtl">

اگر مدل هنوز اجرا نشده است، در نوار کناری روی «نمونه» کلیک کنید تا کل نمای بصری را بدون Ollama ببینید.

</div>

---

## 2. The four tabs · چهار زبانه

### Flow · مسیر

* **What to look at** — plain-language findings for *this* question
  (🔴 serious, 🟡 worth knowing, 🟢 healthy). Example: *“2 retrieved chunks were
  thrown away before the prompt because max_chunks = 3.”*
* **Funnel** — how many chunks survived each stage:

  ```
  Dense (meaning)      ████████████████████ 30
  Sparse (keywords)    ████████████████████ 30
  After merge (RRF)    ██████               10
  After rerank         █████                 8
  Sent to the model    ███                   5   ← only these were visible to the LLM
  Cited in answer      ██                    2
  ```
* **Step by step** — every stage with its real duration and a one-sentence
  explanation of *why that stage exists*.

### Chunks · قطعات

One card per candidate, in final order. Each card tells you why it ranked where
it did and whether it mattered:

| Badge | Meaning |
|---|---|
| `✓ sent to model` | This chunk was inside the prompt |
| `✗ dropped` | Retrieved, then cut before the prompt — the model never saw it |
| `cited [2]` | The answer actually referenced it |
| `not cited` | It was sent but the model ignored it (noise) |
| `found by both searches` | Meaning and keyword search agreed — strongest signal |
| `meaning only` / `keywords only` | Only one retriever found it |
| `dense #3` / `bm25 #1` | Its position in each individual ranked list |
| `RRF 0.0325` | Fusion score (built from *rank positions*, not raw scores) |
| `rerank 0.94` | Cross-encoder relevance score |
| `↑ moved 4 by rerank` | The reranker promoted or demoted it |
| `1500 / 4200 chars` | Characters actually sent vs. the chunk's full length |
| `✂ cut off` | The chunk was truncated — the answer may be missing what got cut |

Click any card to read the chunk text, with **your question's words
highlighted** so it is obvious why it was retrieved.

### Exact prompt · پرامپت

The literal text sent to the model — not a reconstruction:

* **System prompt** — the instructions and citation rules
* **Context block** — your chunks after formatting, numbered `[1] [2] [3]`
* **Full user message** — context + your question, exactly as sent
* **Context window usage** — estimated tokens vs. `num_ctx`

Each box has a **copy** button, so you can paste the prompt into another tool
to compare.

> **The key rule this tab makes obvious:** the model can only use what is in
> this text. If an answer is wrong, check here first — usually the right chunk
> never made it into the prompt.

<div dir="rtl">

**قانون کلیدی:** مدل فقط همین متن را می‌بیند. اگر پاسخ اشتباه بود، اول اینجا را بررسی کنید؛ معمولاً قطعهٔ درست اصلاً وارد پرامپت نشده است.

</div>

### Setup · تنظیمات

The exact configuration behind this answer: models, temperature, `num_ctx`,
dense/sparse candidate counts, RRF `k`, language filter, `max_chunks`,
`max_chars_per_chunk`, and total time.

It also shows three versions of your query, which surprises most people:

| | |
|---|---|
| **Question as you typed it** | `What is one-shot prompting?` |
| **Text actually embedded** | `Instruct: ...\nQuery: What is one-shot prompting?` |
| **Text used for keyword search** | `What is one-shot prompting?` (raw, no prefix) |

---

## 3. Reading the findings · تفسیر هشدارها

| Finding | What it means | Usual fix |
|---|---|---|
| `dropped` | Retrieval found more than the prompt allows | Raise `generation.context_builder.max_chunks` |
| `truncated` | A chunk was cut mid-way | Raise `max_chars_per_chunk` or use smaller chunks |
| `uncited` | The model ignored some chunks it was given | Send fewer, better chunks (enable the reranker) |
| `invalid_citation` | The answer cites `[9]` when only `[1]-[5]` existed | The model invented it — treat the answer with suspicion |
| `single_source` | Every chunk came from one file | Check whether other documents were indexed |
| `language_filter_empty` | The language filter hid everything | Set `retrieval.hybrid.language_filter: false` and retry |
| `dense_empty` / `sparse_empty` | An index is missing or empty | Re-run `run_ingestion.py` / `run_bm25_index.py` |
| `context_overflow_risk` | Prompt is >90% of the context window | Lower `max_chunks` or raise `num_ctx` |
| `keyword_ignored` | Only the meaning search contributed | Normal for conceptual questions; suspicious for exact codes |

---

## 4. How it works (for developers)

`rag_pipeline/observability/query_trace.py`

The tracer **does not re-implement the answering logic**. Two code paths would
drift apart and the picture would start lying. Instead it:

1. temporarily wraps the methods of the live objects —
   `embedder.embed_texts`, `chroma.query`, `bm25.search`,
   `retriever.retrieve`, `reranker.rerank`, `llm_client.generate`;
2. calls the ordinary `RagAnswerer.answer(question)` **once**;
3. restores every method in a `finally` block, even if the call raised.

Consequences, by design:

* The answer is **byte-for-byte identical** whether tracing is on or off
  (asserted by a test).
* The displayed prompt is the **real** string handed to the LLM.
* If building the trace itself fails, the trace is dropped and **the answer
  still returns**. Observability must never cost you an answer.

### API

| Endpoint | Purpose |
|---|---|
| `POST /api/chats/<id>/messages` with `{"message": "...", "trace": true}` | Answer **and** trace (also `?trace=1`) |
| `GET /api/chats/<id>/messages/<index>/trace` | Re-open the inspector for an older message |
| `GET /api/trace/demo` | Example trace — no model required |
| `GET /api/health` | Whether the pipeline was built, and the last error |

Traces are stored on the message in `chat_history.json`, so the inspector keeps
working after a reload. Messages written by older versions simply have no
trace, and the UI treats that as “inspector not available”.

### Use it outside the web app

```python
from rag_pipeline.observability.query_trace import trace_answer

result, trace = trace_answer(answerer, "What is a CMI?")
print(result.answer)                       # unchanged
print(trace["prompt"]["user"])             # the exact prompt
for c in trace["candidates"]:
    print(c["used_in_prompt"], c["chunk_id"], c["rrf_score"])
```

### Tests

`rag_pipeline/tests/test_query_trace.py` — 26 tests, no Ollama required. They
use the **real** `HybridRetriever`, **real** RRF fusion and **real**
`RagAnswerer`, faking only the edges (embedding model, Chroma, BM25, LLM), and
cover: answer-is-unchanged, methods-restored-on-success, methods-restored-
after-an-exception, dense/sparse recording, prompt fidelity, truncation,
citations, invalid citations, rerank movement, empty results, language filter,
and demo/real shape parity.

```bash
cd your_project
python3 -m unittest rag_pipeline.tests.test_query_trace
python3 rag_pipeline/tests/run_tests.py     # full suite
```
