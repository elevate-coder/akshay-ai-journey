"""
RAG Ingestion Pipeline — Chunking Stage
========================================

Part of the personal RAG project (ingestion stage: extraction -> CHUNKING -> embedding -> vector storage).

This module implements TWO chunking strategies so they can be compared head-to-head
on the same source documents (e.g. CVs / case studies):

1. recursive_chunk()  -> structural chunking (fast, deterministic, no model calls)
2. semantic_chunk()   -> meaning-aware chunking (splits where topic/similarity shifts)

Design notes:
- Both return a list of `Chunk` objects with identical shape, so downstream code
  (embedding stage) doesn't care which strategy produced them.
- semantic_chunk() uses TF-IDF + cosine similarity between sentences rather than a
  neural embedding model. This keeps it dependency-light and works fully offline
  (no model download needed). In production you'd swap in real sentence embeddings
  (e.g. sentence-transformers, or your embedding-stage model) — the split LOGIC
  stays the same, only the similarity signal changes. That swap point is marked
  below with a comment.

Run this file directly to see both methods compared on a sample document:
    python chunking.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Shared data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single chunk of text plus metadata, ready for the embedding stage."""
    text: str
    chunk_index: int
    method: str                 # "recursive" or "semantic"
    source: Optional[str] = None
    char_start: int = 0
    char_end: int = 0
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(#{self.chunk_index}, {self.method}, {len(self.text)} chars) -> {preview!r}..."


# ---------------------------------------------------------------------------
# 1. Recursive chunking
# ---------------------------------------------------------------------------

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_on_separator(text: str, separator: str) -> List[str]:
    if separator == "":
        return list(text)  # last resort: split into individual characters
    return text.split(separator)


def _recursive_split(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    """Core recursive splitting logic. Returns pieces all <= chunk_size (best effort)."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # No separators left — hard split by character count.
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, remaining_separators = separators[0], separators[1:]
    pieces = _split_on_separator(text, separator)

    results: List[str] = []
    for piece in pieces:
        if not piece:
            continue
        if len(piece) <= chunk_size:
            results.append(piece)
        else:
            # Piece still too big — recurse with the next separator down the hierarchy.
            results.extend(_recursive_split(piece, chunk_size, remaining_separators))
    return results


def _merge_with_overlap(pieces: List[str], chunk_size: int, chunk_overlap: int, separator_join: str = " ") -> List[str]:
    """
    Greedily re-merge small pieces up to chunk_size, then apply overlap between
    consecutive final chunks so context isn't lost at boundaries.
    """
    merged: List[str] = []
    current = ""
    for piece in pieces:
        candidate = (current + separator_join + piece).strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = piece
    if current:
        merged.append(current)

    if chunk_overlap <= 0 or len(merged) <= 1:
        return merged

    overlapped: List[str] = [merged[0]]
    for prev, curr in zip(merged, merged[1:]):
        tail = prev[-chunk_overlap:]
        overlapped.append((tail + separator_join + curr).strip())
    return overlapped


def recursive_chunk(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None,
    source: Optional[str] = None,
) -> List[Chunk]:
    """
    Split `text` using a hierarchy of separators (paragraph -> line -> sentence -> word -> char),
    falling back to the next separator only when a piece is still too large.

    Args:
        text: raw extracted text (from PDF/DOCX/TXT/MD extraction stage)
        chunk_size: max characters per chunk (swap for a token counter later if needed)
        chunk_overlap: characters repeated between consecutive chunks for context continuity
        separators: ordered list of separators to try, most-preferred first
        source: filename or doc id, stored in chunk metadata

    Returns:
        List[Chunk]
    """
    separators = separators or DEFAULT_SEPARATORS
    raw_pieces = _recursive_split(text.strip(), chunk_size, separators)
    final_pieces = _merge_with_overlap(raw_pieces, chunk_size, chunk_overlap)

    chunks: List[Chunk] = []
    cursor = 0
    for i, piece in enumerate(final_pieces):
        start = text.find(piece[:30], cursor) if piece else cursor
        start = max(start, 0)
        end = start + len(piece)
        chunks.append(Chunk(
            text=piece,
            chunk_index=i,
            method="recursive",
            source=source,
            char_start=start,
            char_end=end,
        ))
        cursor = end
    return chunks


# ---------------------------------------------------------------------------
# 2. Semantic chunking
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter (no heavy NLP dependency)."""
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunk(
    text: str,
    similarity_threshold: float = 0.25,
    min_sentences_per_chunk: int = 2,
    max_chunk_size: int = 1200,
    source: Optional[str] = None,
) -> List[Chunk]:
    """
    Split `text` into chunks by detecting topic shifts between consecutive sentences.

    How it works:
        1. Split text into sentences.
        2. Vectorize sentences (TF-IDF here — swap for real sentence embeddings
           in production, e.g. `model.encode(sentences)` from sentence-transformers).
        3. Compute cosine similarity between each consecutive sentence pair.
        4. Where similarity drops below `similarity_threshold`, that's a topic
           boundary -> start a new chunk.
        5. Also hard-splits if a chunk would exceed `max_chunk_size` regardless
           of similarity, to keep chunks embedding-friendly.

    Args:
        text: raw extracted text
        similarity_threshold: below this cosine similarity, sentences are considered
            topically unrelated and a new chunk starts (lower = fewer, bigger chunks)
        min_sentences_per_chunk: avoid pathologically tiny chunks
        max_chunk_size: hard character cap per chunk regardless of similarity
        source: filename or doc id, stored in chunk metadata

    Returns:
        List[Chunk]
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [Chunk(text=text.strip(), chunk_index=0, method="semantic", source=source,
                       char_start=0, char_end=len(text))] if text.strip() else []

    # --- Similarity signal ---------------------------------------------------
    # SWAP POINT: replace this TF-IDF block with real sentence embeddings if
    # you want semantic meaning rather than lexical (word-overlap) similarity,
    # e.g.:
    #   from sentence_transformers import SentenceTransformer
    #   model = SentenceTransformer("all-MiniLM-L6-v2")
    #   vectors = model.encode(sentences)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(tfidf_matrix)
    # ---------------------------------------------------------------------------

    boundaries = [0]  # sentence indices where a new chunk starts
    current_len = len(sentences[0])
    sentences_in_current = 1

    for i in range(1, len(sentences)):
        sim = sim_matrix[i - 1, i]
        would_exceed_size = current_len + len(sentences[i]) > max_chunk_size
        topic_shift = sim < similarity_threshold and sentences_in_current >= min_sentences_per_chunk

        if topic_shift or would_exceed_size:
            boundaries.append(i)
            current_len = len(sentences[i])
            sentences_in_current = 1
        else:
            current_len += len(sentences[i])
            sentences_in_current += 1

    boundaries.append(len(sentences))

    chunks: List[Chunk] = []
    cursor = 0
    for idx, (start_i, end_i) in enumerate(zip(boundaries, boundaries[1:])):
        piece = " ".join(sentences[start_i:end_i]).strip()
        start = text.find(piece[:30], cursor)
        start = max(start, 0)
        end = start + len(piece)
        chunks.append(Chunk(
            text=piece,
            chunk_index=idx,
            method="semantic",
            source=source,
            char_start=start,
            char_end=end,
            extra={"sentence_count": end_i - start_i},
        ))
        cursor = end
    return chunks


# ---------------------------------------------------------------------------
# Demo / comparison harness
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """Akshay led the Managed Security Services engineering organization at NTT Communications, \
overseeing a 45-person team responsible for delivery across multiple markets. The group managed over \
SGD 20 million in statements of work, covering monitoring, incident response, and platform engineering. \
Under his leadership, the team improved SLA adherence and reduced escalation turnaround time significantly.

Later, Akshay moved into a commercial leadership role as Chief Customer Officer at Anchanto. In this role \
he was accountable for SGD 16 million in annual recurring revenue across 225 enterprise customers spanning \
ten countries. He restructured the customer success function, introduced tiered account management, and \
built renewal playbooks that improved retention.

Most recently, Akshay served as Director of Generative AI Services for APAC and as the MEA AI Governance \
Lead at NTT DATA. In this capacity he advised enterprise clients on responsible AI adoption, governance \
frameworks, and applied generative AI use cases. He also completed the AIGP certification from IAPP, \
reflecting a deeper specialization in AI governance and assurance."""


def run_demo():
    print("=" * 80)
    print("RECURSIVE CHUNKING")
    print("=" * 80)
    recursive_chunks = recursive_chunk(SAMPLE_TEXT, chunk_size=300, chunk_overlap=40, source="sample_cv.txt")
    for c in recursive_chunks:
        print(c)
        print("-" * 40)

    print("\n" + "=" * 80)
    print("SEMANTIC CHUNKING")
    print("=" * 80)
    semantic_chunks = semantic_chunk(SAMPLE_TEXT, similarity_threshold=0.15, source="sample_cv.txt")
    for c in semantic_chunks:
        print(c)
        print("-" * 40)

    print(f"\nrecursive_chunk -> {len(recursive_chunks)} chunks")
    print(f"semantic_chunk  -> {len(semantic_chunks)} chunks")


if __name__ == "__main__":
    run_demo()
