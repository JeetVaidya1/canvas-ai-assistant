import os
from typing import List, Dict, Any, Optional, Iterable
from dotenv import load_dotenv
from supabase import create_client

# Load Supabase credentials
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

db = create_client(SUPABASE_URL, SUPABASE_KEY)

class VectorStore:
    """
    Vector store with metadata-aware inserts and enhanced querying for RAG.
    Compatible with text-embedding-3-large (3072-dim) and pgvector.
    """

    # ---------- INSERTS ----------
    def add(
        self,
        course_id: str,
        doc_name: str,
        chunk_id: int,
        embedding: List[float],
        content: str,
        page: Optional[int] = None,
        slide: Optional[int] = None,
        section: Optional[str] = None,
        sha256: Optional[str] = None,
    ) -> None:
        """Insert one chunk (with metadata) into 'embeddings'."""
        record = {
            "course_id": course_id,
            "doc_name": doc_name,
            "chunk_id": chunk_id,
            "embedding": embedding,      # pgvector[] column
            "content": content,
            "page": page,
            "slide": slide,
            "section": section,
            "sha256": sha256,
        }
        db.table("embeddings").insert(record).execute()

    def bulk_add(self, records: Iterable[Dict[str, Any]]) -> None:
        """
        Insert many records at once. Each record should include:
        course_id, doc_name, chunk_id, embedding, content, and optional page/slide/section/sha256.
        """
        payload = list(records)
        if not payload:
            return
        # Supabase can handle ~500 rows per insert comfortably; chunk if needed.
        BATCH = 500
        for i in range(0, len(payload), BATCH):
            db.table("embeddings").insert(payload[i : i + BATCH]).execute()

    # ---------- QUERIES ----------
    def query(self, course_id: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query most-similar chunks within a course via RPC 'match_embeddings_enhanced'.
        Falls back to basic RPC if needed.
        Returns content + metadata (page/slide/section) when available.
        """
        try:
            vector_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"
            resp = db.rpc(
                "match_embeddings_enhanced",
                {
                    "query_embedding": vector_literal,
                    "course_id_param": course_id,
                    "match_threshold": 0.1,
                    "match_count": top_k,
                },
            ).execute()

            results = []
            for row in (resp.data or []):
                results.append({
                    "content": row.get("content"),
                    "doc_name": row.get("doc_name"),
                    "chunk_id": row.get("chunk_id"),
                    "similarity": row.get("similarity", 0.0),
                    "course_id": row.get("course_id"),
                    "page": row.get("page"),
                    "slide": row.get("slide"),
                    "section": row.get("section"),
                    "sha256": row.get("sha256"),
                })
            return results
        except Exception as e:
            print(f"Vector query error (enhanced): {e}")
            return self._basic_query(course_id, query_embedding, top_k)

    def _basic_query(self, course_id: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Fallback RPC 'match_embeddings' (may not return similarity or metadata)."""
        try:
            vector_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"
            resp = db.rpc(
                "match_embeddings",
                {
                    "query_embedding": vector_literal,
                    "course_id_param": course_id,
                    "match_count": top_k,
                },
            ).execute()

            results = []
            for row in (resp.data or []):
                results.append({
                    "content": row.get("content"),
                    "doc_name": row.get("doc_name"),
                    "chunk_id": row.get("chunk_id"),
                    "similarity": row.get("similarity", 0.0),
                    "course_id": row.get("course_id"),
                    # Depending on your basic RPC/SELECT, these may be None:
                    "page": row.get("page"),
                    "slide": row.get("slide"),
                    "section": row.get("section"),
                    "sha256": row.get("sha256"),
                })
            return results
        except Exception as e:
            print(f"Basic query error: {e}")
            return []

    def query_by_metadata(
        self,
        course_id: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 100,
        order_by: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Simple metadata filter (no vector search). Useful for getting all chunks of a doc.
        Example: filters={"doc_name": "Lecture1.pdf"}
        """
        try:
            q = db.table("embeddings").select(
                "course_id, doc_name, chunk_id, content, page, slide, section, sha256"
            ).eq("course_id", course_id)

            for k, v in (filters or {}).items():
                q = q.eq(k, v)

            if order_by:
                q = q.order(order_by, desc=False)

            resp = q.limit(top_k).execute()
            return resp.data or []
        except Exception as e:
            print(f"query_by_metadata error: {e}")
            return []

    # ---------- UTIL / ADMIN ----------
    def get_document_stats(self, course_id: str) -> Dict[str, Any]:
        """Return chunk counts per document for a course."""
        try:
            resp = db.table("embeddings").select("doc_name, chunk_id").eq("course_id", course_id).execute()
            counts = {}
            for row in (resp.data or []):
                counts[row["doc_name"]] = counts.get(row["doc_name"], 0) + 1
            return {
                "total_chunks": sum(counts.values()),
                "total_documents": len(counts),
                "document_chunks": counts,
            }
        except Exception as e:
            print(f"Stats error: {e}")
            return {"total_chunks": 0, "total_documents": 0, "document_chunks": {}}

    def search_by_document(
        self,
        course_id: str,
        doc_name: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Vector search constrained to a specific document (RPC must support it)."""
        try:
            vector_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"
            resp = db.rpc(
                "match_embeddings_by_document",
                {
                    "query_embedding": vector_literal,
                    "course_id_param": course_id,
                    "doc_name_param": doc_name,
                    "match_count": top_k,
                },
            ).execute()

            results = []
            for row in (resp.data or []):
                results.append({
                    "content": row.get("content"),
                    "doc_name": row.get("doc_name"),
                    "chunk_id": row.get("chunk_id"),
                    "similarity": row.get("similarity", 0.0),
                    "course_id": row.get("course_id"),
                    "page": row.get("page"),
                    "slide": row.get("slide"),
                    "section": row.get("section"),
                    "sha256": row.get("sha256"),
                })
            return results
        except Exception as e:
            print(f"Document search error: {e}")
            return []

    def delete_by_course(self, course_id: str) -> bool:
        """Delete all embeddings for a course."""
        try:
            db.table("embeddings").delete().eq("course_id", course_id).execute()
            return True
        except Exception as e:
            print(f"Delete course error: {e}")
            return False

    def delete_by_document(self, course_id: str, doc_name: str) -> bool:
        """Delete all embeddings for a specific document."""
        try:
            db.table("embeddings").delete().eq("course_id", course_id).eq("doc_name", doc_name).execute()
            return True
        except Exception as e:
            print(f"Delete document error: {e}")
            return False
