"""
RAG Ingestion + Retrieval Pipeline — Full Pluggable Orchestrator
====================================================================
Wires together every stage as a swappable strategy:

    EXTRACTION  -> extraction.py    (pdf, docx, txt, md, html)
    CHUNKING    -> chunking.py      (recursive, semantic)
    EMBEDDING   -> embedding.py     (hashing, tfidf, local_minilm, openai)
    STORAGE     -> storage.py       (chroma, in_memory)
    RETRIEVAL   -> retrieval.py     (similarity, mmr, hybrid)

Usage:
    pipeline = RAGPipeline(
        chunker="semantic",
        embedder_name="hashing",
        store_name="chroma",
        retrieval_strategy="hybrid",
    )
    pipeline.ingest_directory("sample_docs")
    results = pipeline.retrieve("your question")

Run this file directly for a demo that ingests sample_docs/ and runs the
same query across every embedder x store x retrieval-strategy combination,
so you can see how each choice actually changes the results.
"""

import os
from pathlib import Path

from extraction import extract, EXTRACTORS
from chunking import recursive_chunk, semantic_chunk
from embedding import get_embedder
from storage import get_store
from retrieval import retrieve as retrieve_with_strategy


def chunk_document(text: str, source: str, chunker: str = "recursive"):
    if chunker == "semantic":
        return semantic_chunk(text, source=source)
    if chunker == "recursive":
        return recursive_chunk(text, source=source)
    raise ValueError(f"Unknown chunker '{chunker}'. Options: recursive, semantic")


class RAGPipeline:
    def __init__(
        self,
        chunker: str = "recursive",
        embedder_name: str = "hashing",
        store_name: str = "chroma",
        retrieval_strategy: str = "similarity",
        persist_dir: str = "./chroma_store",
        collection_name: str = "documents",
    ):
        self.chunker = chunker
        self.embedder_name = embedder_name
        self.retrieval_strategy = retrieval_strategy

        # tfidf needs a corpus to fit on before it can embed anything — for
        # a fresh pipeline with nothing ingested yet, defer building it
        # until the first ingest call. For hashing/local_minilm/openai this
        # just builds immediately since they don't need a corpus.
        self._embedder = None if embedder_name == "tfidf" else get_embedder(embedder_name)
        self._store_name = store_name
        self._store_kwargs = {"persist_dir": persist_dir, "collection_name": collection_name} if store_name == "chroma" else {}
        self._store = None if embedder_name == "tfidf" else get_store(store_name, self._embedder, **self._store_kwargs)

    def _ensure_ready(self, full_corpus: list[str] | None = None):
        """
        Builds the embedder/store once. For corpus-fit embedders (tfidf),
        this MUST happen after seeing the full corpus, not incrementally
        per file — TF-IDF's output dimensionality depends on vocabulary
        size, so fitting again mid-ingestion changes the vector dimension
        and Chroma will reject the next upsert with a dimension mismatch.
        Hashing/local_minilm/openai embedders have fixed dimensionality
        and don't care, but they go through the same path for consistency.
        """
        if self._embedder is not None:
            return
        corpus_for_fit = full_corpus if self.embedder_name == "tfidf" else None
        self._embedder = get_embedder(self.embedder_name, corpus_for_fit=corpus_for_fit)
        self._store = get_store(self._store_name, self._embedder, **self._store_kwargs)

    def ingest_file(self, path: str):
        if self._embedder is None:
            raise RuntimeError(
                "Embedder not initialized. Call ingest_directory() (which handles "
                "the corpus-fit pass automatically), or call _ensure_ready(corpus) "
                "yourself before ingest_file() for a corpus-fit embedder like tfidf."
            )
        text = extract(path)
        chunks = chunk_document(text, source=os.path.basename(path), chunker=self.chunker)
        if not chunks:
            print(f"  [skip] {path}: no extractable text")
            return

        ids = [f"{c.source}::{self.chunker}::{c.chunk_index}" for c in chunks]
        docs = [c.text for c in chunks]
        metadatas = [
            {"source": c.source, "chunk_id": c.chunk_index, "chunker": c.method}
            for c in chunks
        ]
        self._store.upsert(ids=ids, documents=docs, metadatas=metadatas)
        print(f"  [ok] {path}: {len(chunks)} chunks ({self.chunker} chunking)")

    def ingest_directory(self, dir_path: str):
        files = [f for f in sorted(Path(dir_path).rglob("*")) if f.suffix.lower() in EXTRACTORS]

        # Pass 1: extract + chunk everything up front so corpus-fit embedders
        # (tfidf) see the full vocabulary before the embedder/store exist.
        file_chunks = {}
        all_texts: list[str] = []
        for f in files:
            text = extract(str(f))
            chunks = chunk_document(text, source=f.name, chunker=self.chunker)
            file_chunks[f] = chunks
            all_texts.extend(c.text for c in chunks)

        self._ensure_ready(full_corpus=all_texts)

        # Pass 2: actually upsert, now that the embedder has stable dimensionality.
        for f, chunks in file_chunks.items():
            if not chunks:
                print(f"  [skip] {f}: no extractable text")
                continue
            ids = [f"{c.source}::{self.chunker}::{c.chunk_index}" for c in chunks]
            docs = [c.text for c in chunks]
            metadatas = [
                {"source": c.source, "chunk_id": c.chunk_index, "chunker": c.method}
                for c in chunks
            ]
            self._store.upsert(ids=ids, documents=docs, metadatas=metadatas)
            print(f"  [ok] {f}: {len(chunks)} chunks ({self.chunker} chunking)")

    def retrieve(self, question: str, n_results: int = 3):
        return retrieve_with_strategy(
            self._store, question, strategy=self.retrieval_strategy, n_results=n_results
        )


# ---------------------------------------------------------------------------
# DEMO: same corpus, same query, every combination
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil

    QUESTION = "What AI governance certifications does the candidate hold?"

    combos = [
        {"chunker": "recursive", "embedder_name": "hashing", "store_name": "chroma", "retrieval_strategy": "similarity"},
        {"chunker": "semantic", "embedder_name": "hashing", "store_name": "chroma", "retrieval_strategy": "similarity"},
        {"chunker": "semantic", "embedder_name": "tfidf", "store_name": "chroma", "retrieval_strategy": "similarity"},
        {"chunker": "semantic", "embedder_name": "tfidf", "store_name": "in_memory", "retrieval_strategy": "hybrid"},
        {"chunker": "semantic", "embedder_name": "tfidf", "store_name": "chroma", "retrieval_strategy": "mmr"},
    ]

    for i, combo in enumerate(combos):
        persist_dir = f"./chroma_store_demo_{i}"
        shutil.rmtree(persist_dir, ignore_errors=True)
        kwargs = dict(combo)
        if kwargs["store_name"] == "chroma":
            kwargs["persist_dir"] = persist_dir

        print(f"\n{'=' * 90}")
        print(f"Combo {i+1}: {combo}")
        print("=" * 90)

        pipeline = RAGPipeline(**kwargs)
        pipeline.ingest_directory("sample_docs")

        print(f"\nQuery: {QUESTION}")
        for doc, meta, dist in pipeline.retrieve(QUESTION):
            print(f"  {dist:.3f} | {meta.get('source')} (chunk {meta.get('chunk_id')}) | {doc[:80]}...")

        shutil.rmtree(persist_dir, ignore_errors=True)
