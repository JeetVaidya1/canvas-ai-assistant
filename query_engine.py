# query_engine.py

import os
from dotenv import load_dotenv
from openai import OpenAI
from vector_store import VectorStore

# Load keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Instantiate clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
vector_store   = VectorStore()

# You can customize this system prompt however you like
SYSTEM_PROMPT = """
You are a helpful tutor. Use the provided context from course materials to answer the user’s question as clearly and thoroughly as possible.
"""

def ask_question(question: str, course_id: str) -> str:
    # 1) Embed the user’s question
    emb_resp = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=[question]
    )
    q_emb = emb_resp.data[0].embedding

    # 2) Retrieve top‑5 relevant chunks from pgvector, with fallback
    try:
        results = vector_store.query(course_id, q_emb, top_k=5) or []
    except Exception:
        results = []

    # 3) Build context string; empty if no results
    context = "\n\n".join(r.get("content", "") for r in results)

    # 4) Call the chat API with context + question
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    chat_resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    return chat_resp.choices[0].message.content
