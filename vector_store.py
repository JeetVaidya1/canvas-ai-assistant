import os
from supabase import create_client
from dotenv import load_dotenv

# Load Supabase credentials from .env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client for both DB and Storage
db = create_client(SUPABASE_URL, SUPABASE_KEY)

class VectorStore:
    """
    A simple wrapper around Supabase pgvector for storing and querying embeddings.
    """
    def add(self, course_id: str, doc_name: str, chunk_id: int, embedding: list[float], content: str):
        """Insert one chunk's embedding into the embeddings table."""
        record = {
            "course_id": course_id,
            "doc_name": doc_name,
            "chunk_id": chunk_id,
            "embedding": embedding,
            "content": content
        }
        db.table("embeddings").insert(record).execute()

    def query(self, course_id: str, query_embedding: list[float], top_k: int = 5):
        """Return the top_k nearest chunks for the given course_id."""
        # Convert Python list to Postgres vector literal
        vector_literal = '<' + ','.join(str(x) for x in query_embedding) + '>'
        sql = f"""
            SELECT doc_name, chunk_id, content
            FROM embeddings
            WHERE course_id = '{course_id}'
            ORDER BY embedding <-> {vector_literal}::vector
            LIMIT {top_k};
        """
        result = db.rpc("sql", {"q": sql}).execute()
        return result.data or []
