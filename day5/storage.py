"""
RAG Pipeline — Storage Stage (pluggable)
============================================
Two vector store backends, both exposing the same minimal interface:

    upsert(ids, documents, metadatas, embeddings)
    query(query_embedding, n_results) -> list of (document, metadata, distance)

  "chroma"     — Chroma's persistent local store (SQLite-backed). Handles
                 indexing for you, scales to a real workload, is what you'd
                 actually deploy with. Default choice.

  "in_memory"  — Plain Python/numpy, brute-force cosine similarity over
                 everything in a dict. No indexing, no persistence, O(n)
                 per query. Exists purely to make the "storage is swappable"
                 point concrete and to show you understand what a vector
                 store is actually doing under the hood, rather than
                 treating Chroma as a black box.

Production note: pgvector, Pinecone, Weaviate, and Qdrant would slot in here
the same way — implement upsert()/query() against their client SDKs. Not
included in this scaffold since they need a running external service.
"""

import chromadb
import numpy as np


class ChromaStore:
    def __init__(self, embedding_function, persist_dir: str = "./chroma_store", collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            collection_name, embedding_function=embedding_function
        )

    def upsert(self, ids, documents, metadatas):
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, query_text: str, n_results: int = 3):
        results = self.collection.query(query_texts=[query_text], n_results=n_results)
        return list(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ))


class InMemoryStore:
    """Brute-force cosine-similarity store. No persistence between runs —
    everything lives in memory for the life of the process. Embeds
    documents itself using the same embedding_function passed in, so the
    interface matches ChromaStore even though there's no real indexing
    happening underneath."""

    def __init__(self, embedding_function):
        self._embed = embedding_function
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._vectors: np.ndarray | None = None

    def upsert(self, ids, documents, metadatas):
        new_vectors = np.array(self._embed(documents))
        for i, doc, meta in zip(ids, documents, metadatas):
            if i in self._ids:
                idx = self._ids.index(i)
                self._documents[idx] = doc
                self._metadatas[idx] = meta
            else:
                self._ids.append(i)
                self._documents.append(doc)
                self._metadatas.append(meta)
        # simplest correct approach: re-embed everything on upsert. Fine at
        # this scale (a personal RAG project); a real implementation would
        # append-and-grow the vector array incrementally instead.
        self._vectors = np.array(self._embed(self._documents))

    def query(self, query_text: str, n_results: int = 3):
        if self._vectors is None or len(self._documents) == 0:
            return []
        query_vec = np.array(self._embed([query_text])[0])
        # cosine distance = 1 - cosine similarity, to match Chroma's convention
        # (lower distance = more similar)
        norms = np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(query_vec)
        norms[norms == 0] = 1e-10  # avoid divide-by-zero on empty/zero vectors
        similarities = (self._vectors @ query_vec) / norms
        distances = 1 - similarities
        top_k_idx = np.argsort(distances)[:n_results]
        return [
            (self._documents[i], self._metadatas[i], float(distances[i]))
            for i in top_k_idx
        ]


def get_store(name: str, embedding_function, **kwargs):
    if name == "chroma":
        return ChromaStore(embedding_function, **kwargs)
    if name == "in_memory":
        return InMemoryStore(embedding_function)
    raise ValueError(f"Unknown store '{name}'. Options: chroma, in_memory")
