"""
RAG Pipeline — Embedding Stage (pluggable)
=============================================
Registry of embedding strategies, all implementing Chroma's EmbeddingFunction
interface (a __call__(input: list[str]) -> list[list[float]]) so any of them
can be dropped into a Chroma collection interchangeably.

Options, roughly in order of retrieval quality (worst to best) and setup
cost (easiest to hardest):

  "hashing"   — HashingVectorizer. Fully offline, zero setup, zero cost.
                Lexical (word-overlap) only. Good for proving the pipeline
                works with no dependencies; not for real retrieval quality.

  "tfidf"     — TfidfVectorizer fit once on the corpus at ingest time.
                Still offline and free, but weights distinctive words more
                than HashingVectorizer does (idf term), so retrieval is
                usually noticeably better. Caveat: the vectorizer must be
                fit on your corpus BEFORE querying, and re-fit if the
                corpus changes significantly — it's corpus-dependent in a
                way hashing isn't.

  "local_minilm" — Chroma's bundled ONNX all-MiniLM-L6-v2 model. True
                semantic embeddings, runs on CPU, no API key. Needs
                internet access once to download the model (~90MB) —
                this was the option blocked in the original sandbox this
                project was built in; it should work fine in a normal dev
                environment.

  "openai"    — OpenAI's text-embedding-3-small via API. Best quality of
                these four, costs money per call, needs OPENAI_API_KEY.
"""

import os
from chromadb import EmbeddingFunction
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer


class HashingEmbedder(EmbeddingFunction):
    """Deterministic, offline, lexical-only. See module docstring."""

    def __init__(self, n_features: int = 384):
        self._vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False)

    def __call__(self, input):
        return self._vectorizer.transform(input).toarray().tolist()

    def name(self) -> str:
        return "hashing-embedder-v1"


class TfidfEmbedder(EmbeddingFunction):
    """Offline, lexical, but idf-weighted — must be fit on a corpus first.

    Usage differs from the other embedders: call `.fit(all_chunk_texts)`
    once before ingesting, because TF-IDF vectors are only comparable to
    each other if produced by the same fitted vectorizer. If you ingest
    incrementally (new docs after the initial fit), re-fit periodically or
    accept that new vocabulary won't be captured until you do.
    """

    def __init__(self, max_features: int = 2000):
        self._vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._fitted = False

    def fit(self, corpus: list[str]):
        self._vectorizer.fit(corpus)
        self._fitted = True

    def __call__(self, input):
        if not self._fitted:
            raise RuntimeError(
                "TfidfEmbedder must be fit on a corpus before use — call "
                "embedder.fit(list_of_chunk_texts) before ingesting or querying."
            )
        return self._vectorizer.transform(input).toarray().tolist()

    def name(self) -> str:
        return "tfidf-embedder-v1"


class LocalMiniLMEmbedder:
    """Thin factory for Chroma's bundled ONNX MiniLM embedder.

    Not instantiated directly here to avoid a network call (model download)
    at import time — call `build()` when you're ready to use it, in an
    environment with internet access.
    """

    @staticmethod
    def build():
        from chromadb.utils import embedding_functions
        return embedding_functions.DefaultEmbeddingFunction()


class OpenAIEmbedder:
    """Thin factory for OpenAI's embedding API. Needs OPENAI_API_KEY."""

    @staticmethod
    def build(model_name: str = "text-embedding-3-small"):
        from chromadb.utils import embedding_functions
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set — required for the 'openai' embedder.")
        return embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=model_name)


def get_embedder(name: str, corpus_for_fit: list[str] | None = None):
    """
    Build an embedder by name. `corpus_for_fit` is required for "tfidf"
    (see TfidfEmbedder docstring) and ignored by the others.
    """
    if name == "hashing":
        return HashingEmbedder()
    if name == "tfidf":
        embedder = TfidfEmbedder()
        if not corpus_for_fit:
            raise ValueError("'tfidf' embedder requires corpus_for_fit — a list of chunk texts to fit on.")
        embedder.fit(corpus_for_fit)
        return embedder
    if name == "local_minilm":
        return LocalMiniLMEmbedder.build()
    if name == "openai":
        return OpenAIEmbedder.build()
    raise ValueError(f"Unknown embedder '{name}'. Options: hashing, tfidf, local_minilm, openai")
