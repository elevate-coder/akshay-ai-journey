"""
RAG Pipeline — Streamlit Front End
=====================================
Interactive UI over the pluggable pipeline (extraction, chunking, embedding,
storage, retrieval) — pick a strategy per stage, ingest documents, run
queries, and optionally generate a cited answer.

Run with:
    streamlit run app.py

Notes on how this fits together:
- Uploaded files are written to a temp directory, then ingested through
  RAGPipeline exactly as rag_pipeline.py does from the command line — the
  UI is a thin layer over the same code, not a separate implementation.
- The pipeline is rebuilt automatically whenever you change a stage
  dropdown, since embedding/storage choice changes the underlying vector
  space (a hashing-embedded vector isn't comparable to a tfidf-embedded
  one) — old results wouldn't make sense to keep around after a switch.
- Generation (Stage 2) requires an Anthropic API key, entered in the
  sidebar. It's kept only in Streamlit's session state (memory, for this
  browser tab), never written to disk.
"""

import os
import shutil
import tempfile

import streamlit as st

from rag_pipeline import RAGPipeline
from rag_query import RAGQueryEngine

st.set_page_config(page_title="RAG Pipeline Explorer", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar — stage selection
# ---------------------------------------------------------------------------

st.sidebar.header("Pipeline configuration")

chunker = st.sidebar.selectbox(
    "Chunking strategy", ["recursive", "semantic"],
    help="recursive = fast, structural. semantic = splits on topic shifts (TF-IDF similarity).",
)
embedder_name = st.sidebar.selectbox(
    "Embedding method", ["hashing", "tfidf", "local_minilm", "openai"],
    help=(
        "hashing = offline, lexical only, zero setup.\n"
        "tfidf = offline, lexical, idf-weighted — usually better than hashing.\n"
        "local_minilm = real semantic embeddings, needs internet once to download the model.\n"
        "openai = real semantic embeddings via API, needs OPENAI_API_KEY, costs money."
    ),
)
store_name = st.sidebar.selectbox(
    "Vector store", ["chroma", "in_memory"],
    help="chroma = persistent, indexed. in_memory = brute-force numpy cosine similarity, nothing persisted.",
)
retrieval_strategy = st.sidebar.selectbox(
    "Retrieval strategy", ["similarity", "mmr", "hybrid"],
    help=(
        "similarity = plain top-k by distance.\n"
        "mmr = diversity-aware — avoids near-duplicate results.\n"
        "hybrid = blends vector similarity with BM25 keyword matching."
    ),
)

st.sidebar.divider()
st.sidebar.subheader("Generation (optional, Stage 2)")
api_key_input = st.sidebar.text_input("ANTHROPIC_API_KEY", type="password", help="Only kept in this session, never saved to disk.")
if api_key_input:
    os.environ["ANTHROPIC_API_KEY"] = api_key_input

if embedder_name == "openai":
    openai_key_input = st.sidebar.text_input("OPENAI_API_KEY", type="password")
    if openai_key_input:
        os.environ["OPENAI_API_KEY"] = openai_key_input

if embedder_name == "local_minilm":
    st.sidebar.warning("local_minilm downloads a ~90MB model on first use — needs internet access.")

# ---------------------------------------------------------------------------
# Session state — persist upload dir + pipeline across reruns, rebuild the
# pipeline whenever the stage config actually changes
# ---------------------------------------------------------------------------

if "upload_dir" not in st.session_state:
    st.session_state.upload_dir = tempfile.mkdtemp(prefix="rag_streamlit_")

current_config = (chunker, embedder_name, store_name, retrieval_strategy)
if st.session_state.get("config") != current_config:
    st.session_state.config = current_config
    st.session_state.pipeline = None  # force rebuild + re-ingest on next action
    st.session_state.ingested_files = set()

# ---------------------------------------------------------------------------
# Main area — upload, ingest, query
# ---------------------------------------------------------------------------

st.title("RAG Pipeline Explorer")
st.caption("Swap any stage independently and see how retrieval quality changes.")

col_upload, col_query = st.columns([1, 1.4])

with col_upload:
    st.subheader("1. Documents")
    uploaded = st.file_uploader(
        "Upload PDF / DOCX / TXT / MD / HTML files",
        type=["pdf", "docx", "txt", "md", "html", "htm"],
        accept_multiple_files=True,
    )

    if uploaded:
        new_files = []
        for f in uploaded:
            dest = os.path.join(st.session_state.upload_dir, f.name)
            if f.name not in st.session_state.ingested_files:
                with open(dest, "wb") as out:
                    out.write(f.getbuffer())
                new_files.append(f.name)

        if new_files or st.session_state.pipeline is None:
            with st.spinner(f"Ingesting with chunker={chunker}, embedder={embedder_name}, store={store_name}..."):
                pipeline = RAGPipeline(
                    chunker=chunker,
                    embedder_name=embedder_name,
                    store_name=store_name,
                    retrieval_strategy=retrieval_strategy,
                    persist_dir=os.path.join(st.session_state.upload_dir, "_chroma_store"),
                )
                pipeline.ingest_directory(st.session_state.upload_dir)
                st.session_state.pipeline = pipeline
                st.session_state.ingested_files.update(f.name for f in uploaded)

    if st.session_state.get("ingested_files"):
        st.success(f"Ingested: {', '.join(sorted(st.session_state.ingested_files))}")
    else:
        st.info("Upload at least one document to get started.")

    if st.button("Reset (clear uploaded docs + index)"):
        shutil.rmtree(st.session_state.upload_dir, ignore_errors=True)
        st.session_state.upload_dir = tempfile.mkdtemp(prefix="rag_streamlit_")
        st.session_state.pipeline = None
        st.session_state.ingested_files = set()
        st.rerun()

with col_query:
    st.subheader("2. Query")
    question = st.text_input("Ask a question about the uploaded documents")
    n_results = st.slider("Number of chunks to retrieve", 1, 10, 3)

    if st.button("Retrieve", type="primary", disabled=not st.session_state.get("pipeline")):
        with st.spinner("Retrieving..."):
            results = st.session_state.pipeline.retrieve(question, n_results=n_results)
        st.session_state.last_results = results
        st.session_state.last_question = question

    if st.session_state.get("last_results"):
        st.markdown(f"**Retrieved for:** _{st.session_state.last_question}_")
        for i, (doc, meta, dist) in enumerate(st.session_state.last_results, start=1):
            with st.expander(f"[{i}] {meta.get('source')} — distance {dist:.3f}", expanded=(i == 1)):
                st.write(doc)

        st.divider()
        st.subheader("3. Generate answer (optional)")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.caption("Enter an ANTHROPIC_API_KEY in the sidebar to enable this.")
        if st.button("Generate cited answer", disabled=not os.environ.get("ANTHROPIC_API_KEY")):
            with st.spinner("Calling the model..."):
                engine = RAGQueryEngine(st.session_state.pipeline)
                try:
                    result = engine.generate(st.session_state.last_question, n_results=n_results)
                    st.markdown("**Answer:**")
                    st.write(result.answer)
                    st.markdown("**Sources cited:**")
                    for s in result.sources:
                        st.caption(f"[{s.index}] {s.source} (chunk {s.chunk_id})")
                except Exception as e:
                    st.error(f"Generation failed: {e}")
