import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def read_document(file_path):
    with open(file_path, "r") as file:
        return file.read()


def chunk_text(text, chunk_size=300):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks


def build_prompt(question, chunks):
    context = "\n\n".join(chunks)

    prompt = f"""
You are a helpful GenAI assistant.

Use the context below to answer the user's question.

CONTEXT:
{context}

QUESTION:
{question}

Return your answer in JSON format:
{{
    "answer": "...",
    "confidence": "high/medium/low",
    "source_used": true
}}
"""
    return prompt


def fake_llm_call(prompt):
    response = {
        "answer": "Based on the provided document, Generative AI helps improve automation, decision-making, and knowledge retrieval.",
        "confidence": "medium",
        "source_used": True
    }

    return json.dumps(response, indent=4)


def save_log(question, response):
    os.makedirs("logs", exist_ok=True)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "response": json.loads(response)
    }

    with open("logs/mini_genai_log.json", "a") as file:
        file.write(json.dumps(log_entry, indent=4))
        file.write("\n")


def main():
    print("=" * 60)
    print("Mini GenAI App")
    print("=" * 60)

    file_path = "sample_document.txt"

    document = read_document(file_path)

    print("\nDocument loaded successfully.")

    chunks = chunk_text(document)

    print(f"Document split into {len(chunks)} chunks.")

    question = input("\nAsk a question about the document: ")

    prompt = build_prompt(question, chunks)

    print("\nGenerated Prompt:")
    print("-" * 60)
    print(prompt)

    response = fake_llm_call(prompt)

    print("\nLLM Response:")
    print("-" * 60)
    print(response)

    save_log(question, response)

    print("\nLog saved successfully.")


if __name__ == "__main__":
    main()