import os

# Set LangSmith variables directly — bypasses .env loading issues
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

os.environ["LANGSMITH_PROJECT"] = "akshay-ai-journey"
from dotenv import load_dotenv

#load_dotenv()
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

print("=" * 60)
print("LANGCHAIN — PROGRAM 1: DOCUMENT RAG")
print("=" * 60)

# ── IMPORTS — all correct modern LangChain paths ──
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── STEP 1: LOAD THE DOCUMENT ──
print("\n📄 Step 1: Loading document...")
loader = TextLoader("sample_document.txt")
documents = loader.load()
print(f"   Loaded {len(documents)} document(s)")
print(f"   Total characters: {len(documents[0].page_content)}")

# ── STEP 2: SPLIT INTO CHUNKS ──
print("\n✂️  Step 2: Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " "]
)
chunks = splitter.split_documents(documents)
print(f"   Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks[:3]):
    print(f"   Chunk {i+1}: {chunk.page_content[:80]}...")

# ── STEP 3: CREATE EMBEDDINGS ──
print("\n🔢 Step 3: Creating embeddings...")
print("   Loading HuggingFace embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("   Embedding model loaded.")

# ── STEP 4: STORE IN VECTOR DATABASE ──
print("\n💾 Step 4: Storing in Chroma vector database...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print(f"   Stored {len(chunks)} chunks in Chroma")

# ── STEP 5: BUILD THE RAG CHAIN ──
print("\n🔗 Step 5: Building RAG chain...")
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=500
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = PromptTemplate.from_template("""
You are an NTT DATA AI Services expert assistant.
Use ONLY the following context to answer the question.
If the answer is not in the context, say:
"I don't have that information in the provided documents."

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("   RAG chain ready.")

# ── STEP 6: ASK QUESTIONS ──
print("\n" + "=" * 60)
print("ASKING QUESTIONS AGAINST THE DOCUMENT")
print("=" * 60)

questions = [
    "What are NTT DATA's four AI capability areas?",
    "How much does an AI Engineering project cost?",
    "Which financial services clients does NTT DATA serve?",
    "What is the Yellow Belt curriculum about?",
    "Does NTT DATA offer quantum computing services?",
]

for q in questions:
    print(f"\n❓ Question: {q}")
    answer = rag_chain.invoke(q)
    print(f"💬 Answer: {answer}")

    import mlflow
mlflow.set_experiment("rag-documents")

with mlflow.start_run():
    mlflow.log_param("chunk_size", 500)
    mlflow.log_param("chunk_overlap", 50)
    mlflow.log_param("retrieval_k", 3)
    mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
    mlflow.log_param("llm", "claude-sonnet-4-6")
    # Run your questions, log quality scores
    mlflow.log_metric("questions_answered", 5)
    mlflow.log_metric("refusals", 1)  # quantum computing question