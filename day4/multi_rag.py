"""
Multi-Source RAG Assistant — day4
Ingests: local text file + live website → one ChromaDB → grounded answers with sources
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(Path(__file__).parent.parent / ".env")

# ── 1. LOAD FROM TWO SOURCES ──────────────────────────────
print("📁 Loading local file...")
local_docs = TextLoader(str(Path(__file__).parent / "company_notes.txt")).load()

print("🌐 Loading website...")
web_docs = WebBaseLoader("https://www.anthropic.com/news/claude-3-5-sonnet").load()

all_docs = local_docs + web_docs
print(f"   Loaded {len(all_docs)} documents")

# ── 2. SPLIT ──────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)
print(f"✂️  Split into {len(chunks)} chunks")

# ── 3. EMBED + STORE ──────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(Path(__file__).parent / "chroma_db"),
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("🗄️  Vector store ready")

# ── 4. CHAIN ──────────────────────────────────────────────
llm = ChatAnthropic(model="claude-sonnet-4-5", max_tokens=1024)

prompt = PromptTemplate.from_template(
    """Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have that information."
Always end your answer with a "Sources:" line listing which source(s) you used.

Context:
{context}

Question: {question}

Answer:"""
)

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── 5. INTERACTIVE LOOP ───────────────────────────────────
print("\n💬 Multi-Source RAG ready. Ask questions (type 'exit' to quit)\n")
while True:
    q = input("You: ").strip()
    if q.lower() in ("exit", "quit"):
        break
    print(f"\nClaude: {chain.invoke(q)}\n")