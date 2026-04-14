import time
import numpy as np
import random
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import requests

# --------------------------------------------------
# SETUP — same doc, same indexes as before
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

Latency in AI systems is measured using percentiles. P50 is the median
response time. P95 means 95% of requests are faster than this value.
P99 is the tail latency — the slowest 1% of requests. Optimizing P95
and P99 is critical for production AI systems.
"""

chunks = [s.strip() for s in document.split('\n') if s.strip()]

print("Setting up indexes...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.array(
    embedder.encode(chunks, normalize_embeddings=True)
).astype("float32")

dim = embeddings.shape[1]
faiss_index = faiss.IndexHNSWFlat(dim, 16)
faiss_index.hnsw.efSearch = 50
faiss_index.add(embeddings)

tokenized = [c.lower().split() for c in chunks]
bm25 = BM25Okapi(tokenized)

# --------------------------------------------------
# RETRIEVAL STRATEGIES
# --------------------------------------------------

def retrieve_vector(query, k=3):
    qvec = np.array(
        embedder.encode([query], normalize_embeddings=True)
    ).astype("float32")
    _, indices = faiss_index.search(qvec, k)
    return [chunks[i] for i in indices[0]]

def retrieve_bm25(query, k=3):
    scores = bm25.get_scores(query.lower().split())
    top_k = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_k]

STRATEGIES = {
    "vector": retrieve_vector,
    "bm25":   retrieve_bm25,
}

# --------------------------------------------------
# LATENCY MEASUREMENT HELPER
# --------------------------------------------------

def measure_latency(fn, query, n=20):
    """Run fn(query) n times, return latency percentiles in ms"""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(query)
        times.append((time.perf_counter() - t0) * 1000)
    times = sorted(times)
    return {
        "p50": round(np.percentile(times, 50), 3),
        "p95": round(np.percentile(times, 95), 3),
        "p99": round(np.percentile(times, 99), 3),
        "mean": round(np.mean(times), 3),
    }

# --------------------------------------------------
# LATENCY BENCHMARK
# --------------------------------------------------

print("\n" + "="*55)
print("LATENCY BENCHMARK (20 runs each)")
print("="*55)

test_query = "What is a vector database?"

for name, fn in STRATEGIES.items():
    stats = measure_latency(fn, test_query, n=20)
    print(f"\n  {name.upper()}")
    print(f"    P50  : {stats['p50']}ms  ← median")
    print(f"    P95  : {stats['p95']}ms  ← 95% of requests faster than this")
    print(f"    P99  : {stats['p99']}ms  ← tail latency (worst 1%)")
    print(f"    Mean : {stats['mean']}ms")

# --------------------------------------------------
# EPSILON-GREEDY BANDIT
# --------------------------------------------------
# The bandit chooses between strategies to minimize latency.
# Epsilon = chance of exploring (trying a random strategy)
# 1-Epsilon = chance of exploiting (using the best known strategy)
# Over time it learns which strategy is faster and uses it more.
# --------------------------------------------------

class EpsilonGreedyBandit:
    def __init__(self, strategies, epsilon=0.2):
        self.strategies = strategies          # dict of name -> function
        self.epsilon = epsilon                # 20% explore, 80% exploit
        self.counts  = {k: 0   for k in strategies}   # how many times tried
        self.rewards = {k: 0.0 for k in strategies}   # cumulative reward

    def choose(self):
        if random.random() < self.epsilon:
            # EXPLORE — try a random strategy
            return random.choice(list(self.strategies.keys()))
        else:
            # EXPLOIT — pick the best average so far
            avg = {
                k: self.rewards[k] / self.counts[k]
                if self.counts[k] > 0 else float('inf')
                for k in self.strategies
            }
            return max(avg, key=avg.get)

    def update(self, strategy, latency_ms):
        self.counts[strategy]  += 1
        # Reward = inverse of latency (faster = higher reward)
        reward = 1000.0 / (latency_ms + 1)
        self.rewards[strategy] += reward

    def stats(self):
        print("\n  Bandit learned stats:")
        for k in self.strategies:
            n = self.counts[k]
            avg_r = self.rewards[k] / n if n > 0 else 0
            print(f"    {k:10s} tried {n:3d}x | avg reward: {avg_r:.2f}")

# --------------------------------------------------
# RUN THE BANDIT — simulate 50 queries
# --------------------------------------------------

print("\n" + "="*55)
print("EPSILON-GREEDY BANDIT SIMULATION (50 queries)")
print("="*55)

bandit = EpsilonGreedyBandit(STRATEGIES, epsilon=0.2)
queries = [
    "What is RAG?",
    "How does FAISS work?",
    "What is BM25?",
    "Explain vector databases",
    "What is P95 latency?",
] * 10  # repeat 5 questions × 10 = 50 total queries

random.shuffle(queries)

choices = []
for i, query in enumerate(queries):
    strategy_name = bandit.choose()
    fn = STRATEGIES[strategy_name]

    t0 = time.perf_counter()
    fn(query)
    latency = (time.perf_counter() - t0) * 1000

    bandit.update(strategy_name, latency)
    choices.append(strategy_name)

    if (i + 1) % 10 == 0:
        counts = {k: choices[-10:].count(k) for k in STRATEGIES}
        print(f"  After {i+1:2d} queries: {counts}")

bandit.stats()

# Final summary
print("\n" + "="*55)
print("SUMMARY")
print("="*55)
best = max(
    bandit.strategies,
    key=lambda k: bandit.rewards[k] / bandit.counts[k] if bandit.counts[k] > 0 else 0
)
print(f"  Bandit settled on: {best.upper()} as the fastest strategy")
print(f"  Total queries served: {sum(bandit.counts.values())}")
print("\nPhase 4 complete!")
