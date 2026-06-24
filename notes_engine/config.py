# notes_engine/config.py — Module-level clients, singletons, and env-overridable config
import os
from dotenv import load_dotenv
from providers import make_client
from vector_store import VectorStore

load_dotenv()
openai_client = make_client()
vector_store = VectorStore()

# ── Config (env-overridable) ─────────────────────────────────────────────────
MODEL_DEFAULT      = os.getenv("MODEL_DEFAULT", "gpt-5-mini")
MODEL_COMPLEX      = os.getenv("MODEL_COMPLEX", "gpt-5")
EMBED_MODEL        = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-large")
RAG_TOP_K_FILE     = int(os.getenv("RAG_TOP_K_FILE", "60"))   # chunks per file
MAX_TOK_DEFAULT    = int(os.getenv("MAX_TOKENS_DEFAULT", "5000"))
MAX_TOK_COMPLEX    = int(os.getenv("MAX_TOKENS_COMPLEX", "7000"))
ALLOW_GENERAL_FILL = os.getenv("ALLOW_GENERAL", "true").lower() == "true"
INCLUDE_FLASHCARDS = os.getenv("INCLUDE_FLASHCARDS", "true").lower() == "true"

# Friendly, modern tone
SYSTEM_STYLE = (
    "You are a friendly, sharp university note-taker. Write like a great TA: "
    "clear, compact, and conversational—not stiff. Keep paragraphs short, "
    "use tidy bullets where helpful, and avoid filler. No chain-of-thought."
)
