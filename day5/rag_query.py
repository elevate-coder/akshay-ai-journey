"""
RAG Query Pipeline — Stage 2: Generation
==========================================

Part of the personal RAG project. Sits on top of the Stage 1 ingestion
pipeline (rag_ingest.py) and completes the loop:

    question -> RETRIEVE top-k chunks -> build a grounded prompt with
    numbered sources -> LLM call -> answer that cites which chunk(s)
    it drew from

Design notes:
- Retrieval reuses IngestionPipeline.query() from rag_ingest.py unchanged —
  this stage only adds prompt construction + the LLM call on top.
- The prompt explicitly instructs the model to cite sources by number and
  to say when the retrieved context doesn't answer the question, rather
  than filling gaps from its own general knowledge. This is the difference
  between "a chatbot with some documents nearby" and an actual RAG system
  a hiring panel will probe on.
- Requires ANTHROPIC_API_KEY as an environment variable. This was NOT
  runnable inside the sandbox this was built in (no key available there) —
  retrieval and prompt-building were verified end-to-end; the actual
  generate() call needs to be tested in your own environment.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 rag_query.py "What AI governance certifications does the candidate hold?"
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List

from rag_pipeline import RAGPipeline

SYSTEM_PROMPT = """You are a retrieval-grounded assistant. You will be given a user question \
and a set of numbered source excerpts retrieved from a document store.

Rules:
1. Answer ONLY using information in the provided sources. Do not use outside knowledge.
2. Every claim in your answer must be followed by a citation to the source number it came \
from, like [1] or [1][3].
3. If the sources don't contain enough information to answer the question, say so explicitly \
rather than guessing or filling the gap from general knowledge.
4. Keep the answer concise — do not restate the sources, synthesize them.
"""


@dataclass
class RetrievedSource:
    index: int          # 1-based, used in citations
    text: str
    source: str          # filename
    chunk_id: int
    distance: float


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: List[RetrievedSource]

    def format_with_sources(self) -> str:
        lines = [self.answer, "", "Sources:"]
        for s in self.sources:
            lines.append(f"  [{s.index}] {s.source} (chunk {s.chunk_id}, distance {s.distance:.3f})")
        return "\n".join(lines)


class RAGQueryEngine:
    def __init__(self, pipeline: RAGPipeline, model: str = "claude-sonnet-4-6"):
        self.pipeline = pipeline
        self.model = model
        self._client = None  # lazy init — only needed for generate(), not for retrieve()/build_prompt()

    def retrieve(self, question: str, n_results: int = 3) -> List[RetrievedSource]:
        raw = self.pipeline.retrieve(question, n_results=n_results)
        return [
            RetrievedSource(
                index=i + 1,
                text=doc,
                source=meta["source"],
                chunk_id=meta["chunk_id"],
                distance=dist,
            )
            for i, (doc, meta, dist) in enumerate(raw)
        ]

    def build_prompt(self, question: str, sources: List[RetrievedSource]) -> str:
        source_block = "\n\n".join(
            f"[{s.index}] (from {s.source})\n{s.text}" for s in sources
        )
        return (
            f"Sources:\n{source_block}\n\n"
            f"Question: {question}\n\n"
            f"Answer the question using only the sources above, with citations."
        )

    def generate(self, question: str, n_results: int = 3) -> RAGAnswer:
        sources = self.retrieve(question, n_results=n_results)
        if not sources:
            return RAGAnswer(question=question, answer="No relevant sources found.", sources=[])

        prompt = self.build_prompt(question, sources)

        if self._client is None:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not set. Export it before calling generate(), "
                    "e.g. `export ANTHROPIC_API_KEY=sk-ant-...`. "
                    "retrieve() and build_prompt() work without it if you just want to "
                    "inspect what would be sent to the model."
                )
            self._client = anthropic.Anthropic(api_key=api_key)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        answer_text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return RAGAnswer(question=question, answer=answer_text, sources=sources)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What AI governance certifications does the candidate hold?"

    # This combo (semantic chunking + tfidf + hybrid retrieval) was the one
    # that correctly ranked the actually-relevant chunk first in the stage
    # comparison in rag_pipeline.py — swap any of these independently.
    pipeline = RAGPipeline(
        chunker="semantic",
        embedder_name="tfidf",
        store_name="in_memory",
        retrieval_strategy="hybrid",
    )
    pipeline.ingest_directory("sample_docs")
    engine = RAGQueryEngine(pipeline)

    print(f"Question: {question}\n")

    # Always safe to run — no API key needed for retrieval + prompt construction.
    sources = engine.retrieve(question)
    print("--- Retrieved sources ---")
    for s in sources:
        print(f"[{s.index}] {s.source} (chunk {s.chunk_id}, distance {s.distance:.3f})")
        print(f"    {s.text[:150]}...")
    print()

    prompt = engine.build_prompt(question, sources)
    print("--- Prompt that would be sent to the model ---")
    print(prompt[:800] + ("..." if len(prompt) > 800 else ""))
    print()

    # Requires ANTHROPIC_API_KEY — will raise a clear error if missing.
    try:
        result = engine.generate(question)
        print("--- Generated answer ---")
        print(result.format_with_sources())
    except RuntimeError as e:
        print(f"--- Skipped generation: {e} ---")
