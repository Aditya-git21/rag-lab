# RAG Lab — Retrieval & Adaptive Inference Prototype

A hands-on exploration of the core systems behind production AI inference:
vector search, hybrid retrieval, and adaptive strategy selection.

Built on: Apple M4, Python 3.11, FAISS, Ollama (llama3.2:1b), all-MiniLM-L6-v2

---

## What's in here

| File | What it does |
|------|-------------|
| `faiss_demo.py` | Builds Flat and HNSW vector indexes, compares M parameter effect on build time and recall |
| `rag_pipeline.py` | Full RAG pipeline — chunks a document, builds FAISS + BM25 indexes, retrieves context, generates answers via local LLM |
| `bandit_latency.py` | Benchmarks retrieval latency (P50/P95/P99), runs epsilon-greedy bandit to adaptively select fastest strategy |

---

## Benchmark Results (measured on M4, 20 runs each)

| Strategy | P50 | P95 | P99 | Quality |
|----------|-----|-----|-----|---------|
| Vector (FAISS HNSW) | 4.6ms | 7.9ms | 31.4ms | Correct — understands meaning |
| BM25 (keyword) | 0.021ms | 0.03ms | 0.068ms | Fails on semantic queries |

**BM25 is 220x faster at retrieval but retrieves wrong chunks for semantic questions.**
Vector search is slower but meaning-aware.

---

## Key finding — why hybrid search exists

Pure BM25: fast but fails on queries like "what is a vector database?"
(returns latency-related chunks because it matches surface keywords)

Pure vector: accurate but 31ms P99 tail latency at small scale —
gets worse as corpus grows to millions of documents.

Hybrid search (alpha-weighted combination of both scores) is the
practical middle ground used in production RAG systems.

---

## Adaptive inference — bandit result

Epsilon-greedy bandit (epsilon=0.2) over 50 simulated queries:

- After 10 queries: already preferring BM25 (8/10 selections)
- After 30 queries: fully converged to BM25 (10/10)
- Avg reward — vector: 75.6 | bm25: 982.2

The bandit correctly learned BM25 is faster for retrieval-only latency.
In a real system the reward signal would include answer quality,
not just speed — which would shift the balance back toward vector.

---

## How to run

```bash
# Setup
python3.11 -m venv venv311
source venv311/bin/activate
pip install faiss-cpu sentence-transformers rank-bm25 numpy langchain langchain-community langchain-ollama einops

# Run Ollama (separate terminal)
ollama serve
ollama pull llama3.2:1b

# Run each phase
python3 faiss_demo.py
python3 rag_pipeline.py
python3 bandit_latency.py
```

---

## What I'd build next (production scale)

- Add a similarity score threshold to filter low-confidence retrievals
- Replace small doc with a real PDF loader (LangChain's PyPDFLoader)
- Add re-ranking using a cross-encoder model
- Scale FAISS to 100k+ vectors and observe where HNSW starts beating Flat
- Make bandit reward signal include answer quality, not just latency
