"""
RAG Pipeline — Retrieval Stage (pluggable)
==============================================
Sits on top of a storage backend (storage.py) and adds retrieval STRATEGY —
i.e. not just "what's stored" but "how do we pick the best k results".

  "similarity" — plain top-k by vector distance. What most tutorials show.
                 Can return near-duplicate chunks if several similar
                 passages all match the query well.

  "mmr"        — Maximal Marginal Relevance. Over-fetches candidates, then
                 greedily picks results that are both relevant to the query
                 AND dissimilar to what's already been picked. Reduces
                 redundancy — useful when your corpus has repeated/near-
                 duplicate content (e.g. several CV versions with the same
                 bullet points).

  "hybrid"     — Combines vector similarity with BM25 keyword search, then
                 merges rankings. Vector search alone can miss exact-match
                 terms (acronyms, product names, cert names like "AIGP")
                 if the embedder doesn't represent them well — this is
                 exactly the failure mode the HashingEmbedder in this
                 project showed. BM25 catches what pure vector search
                 misses; this is the standard production fix.
"""

import numpy as np
from rank_bm25 import BM25Okapi


def _similarity_retrieve(store, query_text: str, n_results: int):
    return store.query(query_text, n_results=n_results)


def _mmr_retrieve(store, query_text: str, n_results: int, fetch_k: int = 10, lambda_mult: float = 0.5):
    """
    Over-fetch `fetch_k` candidates by similarity, then greedily select
    `n_results` that balance relevance to the query against diversity from
    already-selected results.

    lambda_mult: 1.0 = pure relevance (same as plain similarity), 0.0 = pure
    diversity (ignores relevance entirely after the first pick). 0.5 is a
    reasonable default balance.
    """
    candidates = store.query(query_text, n_results=min(fetch_k, _store_size(store)))
    if not candidates:
        return []

    # Re-embed candidate texts to compute pairwise similarity for the
    # diversity term — reuses whatever embedder the store was built with.
    embed = store._embed if hasattr(store, "_embed") else store.collection._embedding_function
    texts = [c[0] for c in candidates]
    vectors = np.array(embed(texts))
    query_vec = np.array(embed([query_text])[0])

    def cosine(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-10
        return float(a @ b) / denom

    relevance = [cosine(v, query_vec) for v in vectors]

    selected_idx: list[int] = []
    remaining_idx = list(range(len(candidates)))

    while remaining_idx and len(selected_idx) < n_results:
        best_score, best_i = None, None
        for i in remaining_idx:
            diversity_penalty = 0.0
            if selected_idx:
                diversity_penalty = max(cosine(vectors[i], vectors[j]) for j in selected_idx)
            mmr_score = lambda_mult * relevance[i] - (1 - lambda_mult) * diversity_penalty
            if best_score is None or mmr_score > best_score:
                best_score, best_i = mmr_score, i
        selected_idx.append(best_i)
        remaining_idx.remove(best_i)

    return [candidates[i] for i in selected_idx]


def _hybrid_retrieve(store, query_text: str, n_results: int, fetch_k: int = 10, vector_weight: float = 0.5):
    """
    Combines vector similarity ranking with BM25 keyword ranking.

    Both rankings are converted to normalized scores (0-1, higher=better)
    and blended by `vector_weight` (1.0 = pure vector, 0.0 = pure BM25).
    This catches cases where the embedder misses an exact term match —
    e.g. a query for "AIGP" should score highly against a chunk containing
    "AIGP" via BM25 even if the vector embedder doesn't represent that
    acronym well.
    """
    vector_candidates = store.query(query_text, n_results=min(fetch_k, _store_size(store)))
    if not vector_candidates:
        return []

    texts = [c[0] for c in vector_candidates]
    metas = [c[1] for c in vector_candidates]
    vector_distances = np.array([c[2] for c in vector_candidates])

    # Normalize vector distances to a 0-1 relevance score (lower distance = higher score)
    if vector_distances.max() > vector_distances.min():
        vector_scores = 1 - (vector_distances - vector_distances.min()) / (vector_distances.max() - vector_distances.min())
    else:
        vector_scores = np.ones_like(vector_distances)

    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_raw = np.array(bm25.get_scores(query_text.lower().split()))
    if bm25_raw.max() > bm25_raw.min():
        bm25_scores = (bm25_raw - bm25_raw.min()) / (bm25_raw.max() - bm25_raw.min())
    else:
        bm25_scores = np.zeros_like(bm25_raw)

    blended = vector_weight * vector_scores + (1 - vector_weight) * bm25_scores
    order = np.argsort(-blended)  # descending — highest blended score first

    results = []
    for i in order[:n_results]:
        # report blended distance (1 - score) so the output shape matches
        # the other strategies (lower = better)
        results.append((texts[i], metas[i], float(1 - blended[i])))
    return results


def _store_size(store) -> int:
    if hasattr(store, "_documents"):
        return max(len(store._documents), 1)
    try:
        return max(store.collection.count(), 1)
    except Exception:
        return 20  # fallback fetch size if the backend doesn't expose a count


STRATEGIES = {
    "similarity": _similarity_retrieve,
    "mmr": _mmr_retrieve,
    "hybrid": _hybrid_retrieve,
}


def retrieve(store, query_text: str, strategy: str = "similarity", n_results: int = 3, **kwargs):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown retrieval strategy '{strategy}'. Options: {list(STRATEGIES)}")
    return STRATEGIES[strategy](store, query_text, n_results, **kwargs)
