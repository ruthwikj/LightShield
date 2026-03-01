"""
RAG (Retrieval-Augmented Generation) System
Using Pinecone for vector storage and OpenAI for embeddings + generation.

Setup:
    pip install pinecone openai tiktoken python-dotenv

Usage:
    1. Set your API keys as environment variables (recommended):
       export PINECONE_API_KEY="your-key"
       export OPENAI_API_KEY="your-key"
    2. Run the script and follow the interactive menu.
"""

import os
import hashlib
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# ─── Configuration ───────────────────────────────────────────────────────────

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_NAME = "rag-index"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50
TOP_K = 5  # number of chunks to retrieve

# ─── RAG System Prompt ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

INSTRUCTIONS:
- Answer the user's question using ONLY the information in the provided context.
- If the context does not contain enough information to answer, say: "I don't have enough information in my knowledge base to answer that question."
- Be concise and accurate. Cite specific details from the context when possible.
- Do not make up information or use knowledge outside the provided context.
- If the context is partially relevant, answer what you can and note what's missing.

CONTEXT:
{context}
"""

# ─── Initialize Clients ─────────────────────────────────────────────────────

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)


# ─── Helper Functions ────────────────────────────────────────────────────────

def get_or_create_index():
    """Create Pinecone index if it doesn't exist."""
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created.")
    else:
        print(f"Using existing index '{INDEX_NAME}'.")
    return pc.Index(INDEX_NAME)


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def get_embedding(text: str) -> list[float]:
    """Get OpenAI embedding for a text string."""
    resp = openai_client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return resp.data[0].embedding


def generate_id(text: str) -> str:
    """Generate a deterministic ID from text content."""
    return hashlib.md5(text.encode()).hexdigest()


# ─── Core RAG Functions ─────────────────────────────────────────────────────

def ingest_text(index, text: str, source: str = "manual"):
    """Chunk, embed, and upsert text into Pinecone."""
    chunks = chunk_text(text)
    print(f"Splitting into {len(chunks)} chunks...")

    vectors = []
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        vid = generate_id(f"{source}_{i}_{chunk[:50]}")
        vectors.append({
            "id": vid,
            "values": emb,
            "metadata": {"text": chunk, "source": source, "chunk_index": i},
        })

    # Upsert in batches of 100
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i: i + 100])

    print(f"Ingested {len(vectors)} chunks from '{source}'.")


def ingest_file(index, filepath: str):
    """Read a text file and ingest it."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    source = os.path.basename(filepath)
    ingest_text(index, text, source=source)


def retrieve(index, query: str, top_k=TOP_K) -> list[dict]:
    """Retrieve the most relevant chunks for a query."""
    q_emb = get_embedding(query)
    results = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    return [
        {"text": m.metadata["text"], "source": m.metadata.get(
            "source", "?"), "score": m.score}
        for m in results.matches
    ]


def ask(index, question: str, top_k=TOP_K) -> str:
    """Full RAG pipeline: retrieve context, then generate an answer."""
    # 1. Retrieve
    docs = retrieve(index, question, top_k=top_k)
    if not docs:
        return "No relevant documents found in the knowledge base."

    # 2. Build context
    context = "\n\n---\n\n".join(
        f"[Source: {d['source']} | Relevance: {d['score']:.2f}]\n{d['text']}" for d in docs
    )

    # 3. Generate
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": question},
    ]
    resp = openai_client.chat.completions.create(
        model=CHAT_MODEL, messages=messages, temperature=0.2)
    answer = resp.choices[0].message.content

    return answer


# ─── Interactive CLI ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  RAG System — Pinecone + OpenAI")
    print("=" * 60)

    index = get_or_create_index()

    while True:
        print("\nOptions:")
        print("  1. Ingest a text file")
        print("  2. Ingest text directly")
        print("  3. Ask a question")
        print("  4. Search (retrieve only, no generation)")
        print("  5. Delete index and start fresh")
        print("  6. Exit")
        choice = input("\nChoice [1-6]: ").strip()

        if choice == "1":
            path = input("File path: ").strip()
            if os.path.isfile(path):
                ingest_file(index, path)
            else:
                print(f"File not found: {path}")

        elif choice == "2":
            print("Paste text below (enter a blank line to finish):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            text = "\n".join(lines)
            source = input("Source name (optional): ").strip() or "manual"
            ingest_text(index, text, source=source)

        elif choice == "3":
            q = input("Question: ").strip()
            if q:
                print("\nSearching & generating...\n")
                answer = ask(index, q)
                print(f"Answer:\n{answer}")

        elif choice == "4":
            q = input("Search query: ").strip()
            if q:
                docs = retrieve(index, q)
                for i, d in enumerate(docs, 1):
                    print(
                        f"\n--- Result {i} (score: {d['score']:.2f}, source: {d['source']}) ---")
                    print(d["text"])

        elif choice == "5":
            confirm = input("Delete index and all data? (yes/no): ").strip()
            if confirm.lower() == "yes":
                pc.delete_index(INDEX_NAME)
                print("Index deleted.")
                index = get_or_create_index()

        elif choice == "6":
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
