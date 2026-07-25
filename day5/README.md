# RAG Pipeline — Fully Pluggable Architecture

Every stage is a swappable strategy, selected by name. This is the
architecture, not just an ingestion demo — designed to answer "how would
you make this production-flexible" in an interview.

```
extraction.py   -> pdf | docx | txt | md | html
chunking.py     -> recursive | semantic
embedding.py    -> hashing | tfidf | local_minilm | openai
storage.py      -> chroma | in_memory
retrieval.py    -> similarity | mmr | hybrid
rag_pipeline.py -> orchestrates all of the above
rag_query.py    -> Stage 2: retrieval + LLM generation with citations
```

## Setup
```bash
pip install chromadb pypdf python-docx scikit-learn beautifulsoup4 rank_bm25 anthropic
python3 rag_pipeline.py   # runs 5 different stage combinations on the same corpus/query
python3 rag_query.py "your question"   # end-to-end retrieval + generation
```

## Usage
```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline(
    chunker="semantic",
    embedder_name="tfidf",
    store_name="in_memory",
    retrieval_strategy="hybrid",
)
pipeline.ingest_directory("sample_docs")
results = pipeline.retrieve("your question")
```

## What each option actually does differently

### Extraction (`extraction.py`)
Format-specific loaders, auto-detected from file extension. HTML extraction
strips `<script>`/`<style>` content specifically — otherwise JS/CSS text
pollutes chunks and embeddings.

### Chunking (`chunking.py`)
- `recursive`: fast, deterministic, splits on structural boundaries
  (paragraph -> line -> sentence -> word)
- `semantic`: splits where topic/similarity shifts between sentences (TF-IDF
  cosine similarity here — swap point for real sentence embeddings is
  marked in the code)

**Bugs fixed in this version**: period-eating on `". "` splits, and
corrupted `char_start`/`char_end` offsets caused by searching source text
with overlap-included text instead of the chunk's own pre-overlap content.

### Embedding (`embedding.py`)
| Option | Quality | Cost | Setup |
|---|---|---|---|
| `hashing` | Lexical only | Free | None — fully offline |
| `tfidf` | Lexical, idf-weighted | Free | Must fit on full corpus first (see note below) |
| `local_minilm` | Real semantic | Free | Needs internet once, to download ~90MB model |
| `openai` | Real semantic, best | Paid per call | Needs `OPENAI_API_KEY` |

**Important gotcha demonstrated in this build**: TF-IDF's output
*dimensionality* depends on vocabulary size. Fitting it incrementally
(per-file, as documents arrive) changes the vector dimension mid-ingestion
and Chroma will reject the next upsert with a dimension mismatch. Fixed by
doing a two-pass ingest — extract + chunk everything first to see the full
vocabulary, fit once, then write. Any corpus-fit embedder needs this same
two-pass treatment; hashing/local_minilm/openai don't care since their
dimensionality is fixed regardless of vocabulary.

### Storage (`storage.py`)
- `chroma`: persistent, SQLite-backed, indexed — what you'd actually deploy
- `in_memory`: brute-force numpy cosine similarity, no persistence. Exists
  to prove you understand what a vector store does under the hood rather
  than treating it as a black box. O(n) per query — fine for a personal
  project's corpus size, not for production scale.

pgvector/Pinecone/Weaviate/Qdrant would implement the same
`upsert()`/`query()` interface — not included here since they need a
running external service.

### Retrieval (`retrieval.py`)
- `similarity`: plain top-k by vector distance
- `mmr`: Maximal Marginal Relevance — over-fetches candidates, then
  greedily picks results that are relevant AND dissimilar to what's
  already been picked. Reduces redundancy when the corpus has near-
  duplicate content.
- `hybrid`: blends vector similarity with BM25 keyword scoring. Catches
  exact-term matches (acronyms, cert names) that an embedder might not
  represent well semantically.

**Concrete result from this build**: querying "What AI governance
certifications does the candidate hold?" — plain similarity search with
the `hashing` embedder ranked the chunk that actually says "AIGP
certification" *third*, behind two chunks that just mention certification
frameworks generically. Switching to `hybrid` retrieval + `tfidf` embedding
correctly ranked it *first*. That's a real, demonstrable before/after —
good interview material.

## Stage 2: Generation (`rag_query.py`)
Retrieval -> grounded, citation-instructed prompt -> LLM call -> answer with
source citations. System prompt explicitly instructs the model to answer
only from retrieved sources, cite every claim by source number, and say
so explicitly when sources don't cover the question — the standard
mitigation for hallucination in RAG, and the answer to "how do you prevent
hallucination here" in an interview.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 rag_query.py "your question"
```

Retrieval and prompt construction run without an API key — only the final
generation call needs one. This was verified end-to-end in a sandbox
without a key available; only the actual LLM call needs testing in your
own environment.

## Streamlit UI (`app.py`)

Interactive front end over the same pipeline — pick a strategy per stage
from dropdowns, upload documents, retrieve, and optionally generate a
cited answer. It's a thin layer over `RAGPipeline`/`RAGQueryEngine`, not a
separate implementation, so anything you verify at the command line
behaves identically here.

```bash
pip install streamlit
streamlit run app.py
```

- **Sidebar**: choose chunker / embedder / store / retrieval strategy.
  Changing any of these rebuilds the pipeline and clears prior results —
  vectors from one embedder aren't comparable to another, so old results
  wouldn't mean anything after a switch.
- **Upload**: PDF/DOCX/TXT/MD/HTML, multiple files at once.
- **Retrieve**: shows each retrieved chunk with its source and distance.
- **Generate** (optional): enter an `ANTHROPIC_API_KEY` in the sidebar
  (kept only in session memory, never written to disk) to get a cited
  answer via the Stage 2 generation pipeline.
- Selecting `openai` as the embedder reveals an `OPENAI_API_KEY` field.
  Selecting `local_minilm` shows a warning that it needs internet access
  on first use to download the model.

Verified in a headless test harness (Streamlit's `AppTest`): the app
starts with no exceptions, and switching any sidebar dropdown triggers a
clean rebuild with no exceptions. A live upload → ingest → retrieve →
generate walkthrough should still be run once in your own environment
before you rely on it for a demo.

## What's next
- **Re-ranking**: cross-encoder re-ranker over top-k hybrid results before
  generation, for even better precision
- **Real semantic embeddings**: swap `hashing`/`tfidf` for `local_minilm`
  or `openai` and re-run the same comparison in `rag_pipeline.py` — this
  is the single biggest quality jump available
- **Connectors**: API-based sources (Confluence, SharePoint) as a sixth
  extraction option
- **Post-hoc citation verification**: check that the model's cited claims
  actually appear in the cited chunk, rather than trusting the prompt
  instruction alone
