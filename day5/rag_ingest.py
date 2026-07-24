"""
RAG Ingestion Pipeline — Stage 1: Extraction, Chunking, Embedding, Storage
----------------------------------------------------------------------------
Fully local, no API keys required. Handles PDF, DOCX, TXT, and MD ingestion.

Architecture:
  1. EXTRACTION   -> format-specific loaders turn raw files into plain text
  2. CHUNKING     -> recursive, structure-aware splitting (not naive fixed-size)
  3. EMBEDDING    -> local ONNX MiniLM model via Chroma's built-in embedder
  4. STORAGE      -> Chroma persistent local vector store
  5. RETRIEVAL    -> similarity search demo at the bottom

Extend this by swapping any stage independently:
  - Extraction: add loaders for pptx, html, or wire in Docling/Unstructured for
    higher-fidelity table/layout extraction
  - Chunking: swap in semantic chunking (embed sentences, split on similarity
    drop) instead of recursive character splitting
  - Embedding: swap Chroma's default embedder for OpenAI/Cohere/local
    sentence-transformers if you need higher retrieval quality
  - Storage: swap Chroma for Pinecone/Weaviate/pgvector for production scale
"""

import os
from pathlib import Path

import chromadb
from pypdf import PdfReader
import docx


# ---------------------------------------------------------------------------
# STAGE 1: EXTRACTION — format-specific loaders
# ---------------------------------------------------------------------------

def extract_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_docx(path: str) -> str:
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs if p.text.strip())


def extract_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


EXTRACTORS = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".txt": extract_txt,
    ".md": extract_txt,
}


def extract(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext not in EXTRACTORS:
        raise ValueError(f"No extractor registered for {ext}")
    return EXTRACTORS[ext](path)


# ---------------------------------------------------------------------------
# STAGE 2: CHUNKING — recursive + semantic, selectable per ingest
# ---------------------------------------------------------------------------
# Uses chunking.py (the two-strategy module, bugs fixed: period-eating on
# ". " splits, and corrupted char_start/char_end offsets from searching with
# overlap-included text). chunk_document() is a thin adapter so the rest of
# the pipeline doesn't care which strategy produced the chunks.

from chunking import recursive_chunk, semantic_chunk


def chunk_document(text: str, source: str, method: str = "recursive") -> list:
    if method == "semantic":
        return semantic_chunk(text, source=source)
    return recursive_chunk(text, source=source)


# ---------------------------------------------------------------------------
# STAGE 3 + 4: EMBEDDING + STORAGE
# ---------------------------------------------------------------------------
# Production note: Chroma's default embedder downloads a local ONNX MiniLM
# model (all-MiniLM-L6-v2) on first use — no API key needed, but it does
# require internet access to Hugging Face's CDN. Swap this HashingEmbedder
# for `chromadb.utils.embedding_functions.DefaultEmbeddingFunction()` (or an
# OpenAI/Cohere embedder) once you're running outside a network-restricted
# sandbox — that will give meaningfully better semantic retrieval than the
# hashing approach below, which only captures lexical/word-level overlap.

from sklearn.feature_extraction.text import HashingVectorizer
from chromadb import EmbeddingFunction


class HashingEmbedder(EmbeddingFunction):
    """Deterministic, offline stand-in for a real embedding model.
    Captures lexical overlap only (not semantic meaning) — good enough to
    prove the pipeline end-to-end, not good enough for production retrieval
    quality."""

    def __init__(self, n_features: int = 384):
        self._vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False)

    def __call__(self, input):
        vectors = self._vectorizer.transform(input)
        return vectors.toarray().tolist()

    def name(self) -> str:
        return "hashing-embedder-v1"


class IngestionPipeline:
    def __init__(self, persist_dir: str = "./chroma_store", collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            collection_name, embedding_function=HashingEmbedder()
        )

    def ingest_file(self, path: str, method: str = "recursive"):
        text = extract(path)
        chunks = chunk_document(text, source=os.path.basename(path), method=method)
        if not chunks:
            print(f"  [skip] {path}: no extractable text")
            return
        ids = [f"{c.source}::{method}::{c.chunk_index}" for c in chunks]
        docs = [c.text for c in chunks]
        metadatas = [
            {"source": c.source, "chunk_id": c.chunk_index, "method": c.method}
            for c in chunks
        ]
        self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
        print(f"  [ok] {path}: {len(chunks)} chunks ingested ({method})")

    def ingest_directory(self, dir_path: str, method: str = "recursive"):
        for f in Path(dir_path).rglob("*"):
            if f.suffix.lower() in EXTRACTORS:
                self.ingest_file(str(f), method=method)

    def query(self, question: str, n_results: int = 3):
        results = self.collection.query(query_texts=[question], n_results=n_results)
        return list(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ))


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipeline = IngestionPipeline(persist_dir="./chroma_store")

    print("Ingesting sample_docs/ ...")
    pipeline.ingest_directory("sample_docs")

    print("\nQuerying: 'What AI governance certifications does the candidate hold?'")
    for doc, meta, dist in pipeline.query("What AI governance certifications does the candidate hold?"):
        print(f"\n  source: {meta['source']} (chunk {meta['chunk_id']}, distance {dist:.3f})")
        print(f"  text: {doc[:200]}...")
