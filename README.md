---
title: RAG Document Analyzer
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---

# RAG Document Analyzer

Upload a PDF (built and tested on SEC 10-K filings) and ask questions about it. Retrieval combines keyword and semantic search, a reranker picks the best chunks, and an LLM answers from them.

**Live demo:** https://huggingface.co/spaces/FASTBuilds/rag-document-analyzer
(Free CPU tier, so processing a new PDF takes about a minute and the Space may need a moment to wake up.)

## How it works

```
PDF -> chunking -> MiniLM embeddings -> ChromaDB
                                           |
query -> BM25 + vector search (EnsembleRetriever, RRF) -> Cohere rerank -> Groq (Llama 3.3 70B) -> answer
```

1. **Ingest.** PyPDFLoader reads the PDF and the text is split into 1000-character chunks with 200 overlap.
2. **Index.** Each chunk is embedded locally with `all-MiniLM-L6-v2` and stored in ChromaDB. A BM25 keyword index is built over the same chunks.
3. **Hybrid retrieval.** For each query, BM25 and vector search each return their top 20 chunks, and `EnsembleRetriever` merges them with reciprocal rank fusion (50/50 weights). BM25 catches exact figures and terms, vector search catches meaning.
4. **Rerank.** Cohere Rerank scores the merged pool against the query and keeps the best 4.
5. **Answer.** Those 4 chunks go to Llama 3.3 70B on Groq, which answers from the context.

## Tech stack

Python, LangChain, ChromaDB, sentence-transformers (MiniLM), rank-bm25, Cohere Rerank, Groq, Streamlit, FastAPI.

## Why RAG instead of just using a long-context LLM?

Modern LLMs can fit a 100-page document directly in their context window, so it's fair to ask why bother retrieving at all. The catch is that LLMs are stateless: the model doesn't remember the document between calls, so a long-context setup re-sends the entire document with every question. A 100-page filing is roughly 60-80K tokens, so ten questions means around 800K tokens sent in total, versus about 20K with retrieval, where each call only carries the handful of chunks that matter.

Cost is the softer argument, though, because prompt caching narrows that gap for a single document. The stronger one is scale: you can't fit a library of hundreds of filings into one context window, while retrieval works the same whether you have one document or a hundred thousand. There's also a precision angle, since details buried in the middle of a very long context tend to get missed, whereas good retrieval surfaces the exact chunk containing the answer. For one document and a few questions, long context is simpler and perfectly fine. RAG is the architecture for many documents, many queries, and a cost budget.

## Evaluation

The repo includes a small eval harness (`eval_set.py`, `run_eval.py`). It runs 25 questions against the Tesla FY2022 10-K: 16 exact-figure lookups (revenue, R&D, net income, EPS, etc.) and 9 open-ended questions (risks, strategy, revenue recognition). Expected answers were checked against the filing's financial statements, and an LLM judge grades each answer for semantic correctness, so formatting differences like "$12,556 million" vs "$12.5 billion" don't cause false fails.

**Result: 25/25.**

Take that number with the right caveats: it's a small set, a single document, mostly factual lookups, and the judge is the same model family that generates the answers. It shows the pipeline works well on this document, not that it's perfect in general.

## Run it locally

```bash
git clone https://github.com/I-am-Carbon/RAG-Document-Analyzer
cd RAG-Document-Analyzer
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_key
COHERE_API_KEY=your_key
```

**Streamlit UI:**
```bash
streamlit run app.py
```

**FastAPI server:**
```bash
uvicorn api:app --reload
```
Then open http://127.0.0.1:8000/docs for the interactive docs.

**Run the eval:**
```bash
python run_eval.py
```
(Set `PDF_PATH` in `run_eval.py` to your copy of the Tesla 10-K. The Cohere free tier is rate limited, so the script sleeps between questions and takes a few minutes.)

## API

| Method | Endpoint | What it does |
|--------|----------|--------------|
| GET | `/health` | Liveness check |
| POST | `/upload` | Upload a PDF, returns a `session_id` |
| POST | `/ask` | Takes `session_id` + `question`, returns the answer |

Each upload gets its own session with its own ChromaDB collection and retrievers, so concurrent users don't overwrite each other's documents.

## Known limitations

- **Concurrency is correct, not scaled.** Sessions are isolated, but embedding is CPU-bound and sessions live in memory, so the free tier handles only a couple of simultaneous users well. Next steps would be a background job queue for embedding and session expiry.
- **Session cleanup.** Per-session ChromaDB folders and temp files aren't cleaned up automatically.
- **Streamlit calls the core logic directly** rather than going through the API. A production setup would have every frontend call the API over HTTP.
- **Small eval set** (see above).

## Project structure

```
main.py         core RAG logic (ingest, hybrid retrieval, rerank, answer)
app.py          Streamlit UI
api.py          FastAPI layer
eval_set.py     golden Q&A set
run_eval.py     eval runner with LLM judge
```
