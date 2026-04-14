import faiss
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import requests
import json

# --------------------------------------------------
# OUR "DOCUMENT" — imagine this is a PDF or webpage
# you loaded. We'll chunk it into paragraphs.
# --------------------------------------------------
document = """
Retrieval Augmented Generation (RAG) is a technique that combines
information retrieval with language model generation. Instead of relying
solely on the model's training data, RAG fetches relevant documents at
query time and includes them in the prompt.

FAISS is Facebook AI Similarity Search, a library for efficient similarity
search over dense vectors. It supports multiple index types including Flat,
IVF, and HNSW. HNSW stands for Hierarchical Navigable Small World graphs.

Vector databases store embeddings and allow fast nearest neighbor search.
Popular vector databases include Pinecone, Weaviate, Milvus, and Qdrant.
FAISS is a library not a full database but is widely used for prototyping.

BM25 is a classical keyword-based ranking algorithm used in search engines.
It scores documents based on term frequency and inverse document frequency.
Hybrid search combines BM25 keyword scores with vector similarity scores.

Large Language Models like GPT-4, Claude, and Llama are transformer-based
models trained on massive text corpora. They generate text by predicting
the next token given a context window of previous tokens.

Latency in AI systems is measured using percentiles. P50 is the median
response time. P95 means 95% of requests are faster than this value.
P99 is the tail latency — the slowest 1% of requests. Optimizing P95
and P99 is critical for production AI systems.
"""

# --------------------------------------------------
# STEP 1 — CHUNK the document into sentences
# --------------------------------------------------
chunks = [s.strip() for s in document.split('\n') if s.strip()]
print(f"Document split into {len(chunks)} chunks\n")

# --------------------------------------------------
# STEP 2 — EMBED chunks and build FAISS index
# --------------------------------------------------
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embedder.encode(chunks, normalize_embeddings=True)
chunk_embeddings = np.array(chunk_embeddings).astype("float32")

dim = chunk_embeddings.shape[1]
index = faiss.IndexHNSWFlat(dim, 16)
index.hnsw.efSearch = 50
index.add(chunk_embeddings)
print(f"FAISS index built with {index.ntotal} chunks")

# --------------------------------------------------
# STEP 3 — BUILD BM25 index (keyword search)
# --------------------------------------------------
tokenized_chunks = [chunk.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)
print(f"BM25 index built\n")

# --------------------------------------------------
# STEP 4 — RETRIEVAL FUNCTIONS
# --------------------------------------------------

def retrieve_vector(query, k=3):
    """Pure vector search using FAISS"""
    qvec = embedder.encode([query], normalize_embeddings=True)
    qvec = np.array(qvec).astype("float32")
    scores, indices = index.search(qvec, k)
    return [chunks[i] for i in indices[0]]

def retrieve_bm25(query, k=3):
    """Pure keyword search using BM25"""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_k = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_k]

def retrieve_hybrid(query, k=3, alpha=0.5):
    """
    Hybrid search — combine vector + BM25 scores
    alpha=1.0 means pure vector, alpha=0.0 means pure BM25
    alpha=0.5 means equal weight (default)
    """
    # Vector scores
    qvec = embedder.encode([query], normalize_embeddings=True)
    qvec = np.array(qvec).astype("float32")
    vec_scores, vec_indices = index.search(qvec, len(chunks))
    
    # Normalize vector scores to 0-1
    vec_score_map = {}
    for score, idx in zip(vec_scores[0], vec_indices[0]):
        vec_score_map[idx] = float(score)
    
    # BM25 scores
    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1
    
    # Combine scores
    combined = {}
    for i in range(len(chunks)):
        v_score = vec_score_map.get(i, 0)
        b_score = bm25_scores[i] / bm25_max  # normalize
        combined[i] = alpha * v_score + (1 - alpha) * b_score
    
    top_k = sorted(combined, key=combined.get, reverse=True)[:k]
    return [chunks[i] for i in top_k]

# --------------------------------------------------
# STEP 5 — CALL LOCAL OLLAMA LLM
# --------------------------------------------------

def ask_llm(context_chunks, question):
    """Send retrieved context + question to local Ollama"""
    context = "\n".join(f"- {c}" for c in context_chunks)
    prompt = f"""You are a helpful assistant. Use the context below to answer the question. Be direct and concise.

Context:
{context}

Question: {question}

Answer based on the context above:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:1b", "prompt": prompt, "stream": False},
            timeout=30
        )
        return response.json()["response"].strip()
    except Exception as e:
        return f"LLM error: {e}"

# --------------------------------------------------
# STEP 6 — FULL RAG PIPELINE (retrieve + generate)
# --------------------------------------------------

def rag(question, retriever="hybrid"):
    t0 = time.perf_counter()
    
    if retriever == "vector":
        context = retrieve_vector(question)
    elif retriever == "bm25":
        context = retrieve_bm25(question)
    else:
        context = retrieve_hybrid(question)
    
    retrieval_time = (time.perf_counter() - t0) * 1000
    
    t1 = time.perf_counter()
    answer = ask_llm(context, question)
    llm_time = (time.perf_counter() - t1) * 1000
    
    return {
        "answer": answer,
        "context": context,
        "retrieval_ms": round(retrieval_time, 2),
        "llm_ms": round(llm_time, 2)
    }

# --------------------------------------------------
# STEP 7 — RUN SOME QUESTIONS
# --------------------------------------------------

questions = [
    "What is RAG?",
    "What is P95 latency?",
    "What is BM25?",
]

print("="*55)
print("RAG PIPELINE RESULTS")
print("="*55)

for q in questions:
    print(f"\nQ: {q}")
    for mode in ["vector", "bm25", "hybrid"]:
        result = rag(q, retriever=mode)
        print(f"\n  [{mode.upper()}] ({result['retrieval_ms']}ms retrieval + {result['llm_ms']}ms LLM)")
        print(f"  Retrieved: {result['context'][0][:80]}...")
        print(f"  Answer: {result['answer'][:150]}")

print("\n\nDone! Full RAG pipeline with vector, BM25, and hybrid retrieval.")
