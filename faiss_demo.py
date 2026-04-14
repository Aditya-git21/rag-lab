import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time

# --------------------------------------------------
# 1. OUR "DATASET" — 15 sentences about random topics
# --------------------------------------------------
sentences = [
    "The Eiffel Tower is located in Paris, France.",
    "Python is a popular programming language for data science.",
    "The mitochondria is the powerhouse of the cell.",
    "Machine learning models learn patterns from data.",
    "The Amazon rainforest produces 20% of the world's oxygen.",
    "Neural networks are inspired by the human brain.",
    "The speed of light is approximately 3 x 10^8 meters per second.",
    "RAG stands for Retrieval Augmented Generation.",
    "FAISS is a library for efficient similarity search.",
    "The Great Wall of China stretches over 13,000 miles.",
    "Embeddings convert text into numerical vectors.",
    "Vector databases store and search high-dimensional vectors.",
    "India is the most populous country in the world.",
    "Transformers are the backbone of modern LLMs.",
    "Apple M4 chip uses unified memory architecture.",
]

# --------------------------------------------------
# 2. LOAD EMBEDDING MODEL (runs locally, no API key)
# --------------------------------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding sentences...")
embeddings = model.encode(sentences, normalize_embeddings=True)
embeddings = np.array(embeddings).astype("float32")

print(f"Embedding shape: {embeddings.shape}")
# Shape will be (15, 768) — 15 sentences, each a 768-dim vector

# --------------------------------------------------
# 3. BUILD A FLAT INDEX (brute force, exact search)
#    This is the baseline — checks every single vector
# --------------------------------------------------
dim = embeddings.shape[1]
flat_index = faiss.IndexFlatIP(dim)  # IP = Inner Product (cosine sim when normalized)
flat_index.add(embeddings)
print(f"\nFlat index: {flat_index.ntotal} vectors stored")

# --------------------------------------------------
# 4. BUILD AN HNSW INDEX (approximate, fast)
#    M = number of connections per node in the graph
#    Higher M = better recall, more memory, slower build
# --------------------------------------------------
M = 16                          # try changing this to 4 or 32 and see what happens
hnsw_index = faiss.IndexHNSWFlat(dim, M)
hnsw_index.hnsw.efConstruction = 200   # higher = better quality graph, slower build
hnsw_index.hnsw.efSearch = 5          # higher = better recall at query time, slower
hnsw_index.add(embeddings)
print(f"HNSW index: {hnsw_index.ntotal} vectors stored (M={M})")

# --------------------------------------------------
# 5. QUERY BOTH INDEXES AND COMPARE
# --------------------------------------------------
queries = [
    "What is a vector database?",
    "Tell me about Paris landmarks",
    "How do neural networks work?",
    "what is a banana",
]

print("\n" + "="*55)
print("SEARCH RESULTS")
print("="*55)

for query in queries:
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec).astype("float32")

    # Search flat (exact)
    t0 = time.perf_counter()
    scores_flat, idx_flat = flat_index.search(query_vec, k=5)
    time_flat = (time.perf_counter() - t0) * 1000

    # Search HNSW (approximate)
    t0 = time.perf_counter()
    scores_hnsw, idx_hnsw = hnsw_index.search(query_vec, k=3)
    time_hnsw = (time.perf_counter() - t0) * 1000

    print(f"\nQuery: '{query}'")
    print(f"  Flat  ({time_flat:.2f}ms): {[sentences[i] for i in idx_flat[0]]}")
    print(f"  HNSW  ({time_hnsw:.2f}ms): {[sentences[i] for i in idx_hnsw[0]]}")
    
    # Check if results match
    match = set(idx_flat[0]) == set(idx_hnsw[0])
    print(f"  Results match: {match}")

# --------------------------------------------------
# 6. EXPERIMENT — what does M actually change?
# --------------------------------------------------
print("\n" + "="*55)
print("EXPERIMENT: M parameter effect on build time")
print("="*55)

for m_val in [4, 8, 16, 32]:
    idx = faiss.IndexHNSWFlat(dim, m_val)
    idx.hnsw.efConstruction = 200
    t0 = time.perf_counter()
    idx.add(embeddings)
    build_time = (time.perf_counter() - t0) * 1000
    print(f"  M={m_val:2d} → build time: {build_time:.1f}ms, memory ~{idx.ntotal * m_val * 2 * 4 / 1024:.1f}KB")

print("\nDone! You just built and queried a vector search engine.")
