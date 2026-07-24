# Day 5 — RAG Ingestion: Chunking Strategies

Part of the RAG ingestion pipeline: **extraction → CHUNKING → embedding → vector storage**

Two strategies implemented and compared head-to-head on the same document:

| Strategy | How it splits | Cost | Best for |
|---|---|---|---|
| `recursive_chunk()` | Separator hierarchy: paragraph → line → sentence → word → char | Free, deterministic | Structured docs, production default |
| `semantic_chunk()` | Cosine similarity between consecutive sentences; splits at topic shifts | Vectorisation per doc | Unstructured prose, mixed-topic documents |

Both return identical `Chunk` objects (text, index, method, source, char offsets, extra metadata) so the embedding stage is agnostic to which produced them.

## Semantic similarity signal

Currently TF-IDF + cosine — dependency-light and fully offline. The swap point to real sentence embeddings is marked in the code:

```python
from sentence_transformers import SentenceTransfonceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(sentences)
```

The split *logic* is unchanged; only the similarity signal improves (lexical overlap → semantic meaning).

## Run

```bash
pip install scikit-learn
python chunking.py
```

## Next in the pipeline
- Embedding stage (all-MiniLM-L6-v2 → 384-dim vectors)
- Vector storage (ChromaDB / pgvector)
- Retrieval quality evaluation with RAGAS
