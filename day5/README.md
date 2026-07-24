# RAG Ingestion Pipeline — Stage 1 Scaffold

## What this is
A working, local RAG ingestion pipeline: extraction -> chunking (recursive
or semantic) -> embedding -> vector storage -> retrieval. Runs fully
offline, no API keys needed.

## Files
- `rag_ingest.py` — orchestrates extraction, embedding, storage, retrieval
- `chunking.py` — the two chunking strategies (recursive + semantic),
  with a built-in comparison demo (`python3 chunking.py`)
- `sample_docs/` — sample files to ingest

## Setup
```bash
pip install chromadb pypdf python-docx scikit-learn
python3 rag_ingest.py
```

## Bugs fixed in chunking.py (from the original draft)
1. **Period-eating on `". "` splits** — `str.split(". ")` silently discarded
   the separator, producing chunks like `"...markets The group..."` instead
   of `"...markets. The group..."`. Fixed with a lookbehind regex split that
   keeps the period attached to the sentence it ends.
2. **Corrupted char_start/char_end offsets** — offsets were computed by
   searching the source text for the *overlap-included* chunk text via
   `.find()`, which often failed (returns -1) once overlap text was
   prepended, silently collapsing several chunks' offsets to 0. Fixed by
   tracking `core_text` (the chunk's own content, pre-overlap) separately
   and searching with that instead.

Both are verified: periods are now intact end-to-end, and every chunk's
`char_start`/`char_end` correctly points into the source document.

## To use with your own docs
1. Drop PDF/DOCX/TXT/MD files into `sample_docs/` (your CVs, case studies,
   TravAI collateral, job descriptions you're tracking)
2. Run `pipeline.ingest_directory("sample_docs", method="recursive")` or
   `method="semantic"` to compare retrieval quality between the two
3. Call `pipeline.query("your question")` for retrieval

## Important: embedding quality
Uses a `HashingEmbedder` (scikit-learn HashingVectorizer) — deterministic,
fully offline, captures word-overlap only, not semantic meaning. Built this
way because this sandbox blocks the download a real embedder needs.

**Before relying on this for anything real**, swap it — one-line change:
```python
from chromadb.utils import embedding_functions
embedder = embedding_functions.DefaultEmbeddingFunction()  # local MiniLM
# or: embedding_functions.OpenAIEmbeddingFunction(api_key=..., model_name="text-embedding-3-small")
```
Then pass `embedding_function=embedder` into `get_or_create_collection()`
in `rag_ingest.py` in place of `HashingEmbedder()`.

Similarly, `semantic_chunk()` in `chunking.py` currently uses TF-IDF
(lexical similarity) rather than real sentence embeddings — the swap point
is marked in the code with a comment.

## What's next (Stage 2+)
- **Retrieval quality**: swap the embedder as above — single biggest jump
- **Query pipeline**: retrieved chunks + question -> LLM call -> answer
  with citations
- **Re-ranking**: cross-encoder re-ranker over top-k results before
  generation
- **Metadata filtering**: filter by source/date/doc-type before similarity
  search
- **Connectors**: pptx, web pages, or API-based sources (Confluence,
  SharePoint)
