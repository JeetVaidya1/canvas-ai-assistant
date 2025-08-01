import os
from supabase import create_client
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Load Supabase credentials from .env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client for DB operations
db = create_client(SUPABASE_URL, SUPABASE_KEY)

class VectorStore:
    """
    Enhanced vector store with advanced querying capabilities for RAG
    """
    def add(self, course_id: str, doc_name: str, chunk_id: int, embedding: list[float], content: str):
        """
        Insert one chunk's embedding into the embeddings table.
        """
        record = {
            "course_id": course_id,
            "doc_name": doc_name,
            "chunk_id": chunk_id,
            "embedding": embedding,
            "content": content
        }
        db.table("embeddings").insert(record).execute()

    def query(self, course_id: str, query_embedding: list[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Enhanced query method with better similarity search and metadata
        """
        try:
            # Convert Python list to Postgres vector literal format
            vector_literal = '[' + ','.join(str(x) for x in query_embedding) + ']'
            
            # Use cosine similarity for better semantic matching
            response = db.rpc(
                'match_embeddings_enhanced',
                {
                    'query_embedding': vector_literal,
                    'course_id_param': course_id,  # Fixed parameter name
                    'match_threshold': 0.1,  # Lower threshold to get more results
                    'match_count': top_k
                }
            ).execute()
            
            results = []
            for row in response.data:
                results.append({
                    'content': row['content'],
                    'doc_name': row['doc_name'],
                    'chunk_id': row['chunk_id'],
                    'similarity': row.get('similarity', 0.0),
                    'course_id': row['course_id']
                })
            
            return results
            
        except Exception as e:
            print(f"Vector query error: {e}")
            # Fallback to basic query if enhanced fails
            return self._basic_query(course_id, query_embedding, top_k)
    
    def _basic_query(self, course_id: str, query_embedding: list[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Fallback basic query method
        """
        try:
            vector_literal = '[' + ','.join(str(x) for x in query_embedding) + ']'
            
            response = db.rpc(
                'match_embeddings',
                {
                    'query_embedding': vector_literal,
                    'course_id_param': course_id,  # Fixed parameter name
                    'match_count': top_k
                }
            ).execute()
            
            results = []
            for row in response.data:
                results.append({
                    'content': row['content'],
                    'doc_name': row['doc_name'],
                    'chunk_id': row['chunk_id'],
                    'similarity': row.get('similarity', 0.0),
                    'course_id': row['course_id']
                })
            
            return results
            
        except Exception as e:
            print(f"Basic query error: {e}")
            return []

    def get_document_stats(self, course_id: str) -> Dict[str, Any]:
        """
        Get statistics about the documents in a course
        """
        try:
            response = db.table("embeddings").select(
                "doc_name, chunk_id"
            ).eq("course_id", course_id).execute()
            
            doc_counts = {}
            total_chunks = len(response.data)
            
            for row in response.data:
                doc_name = row['doc_name']
                doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
            
            return {
                'total_chunks': total_chunks,
                'total_documents': len(doc_counts),
                'document_chunks': doc_counts
            }
            
        except Exception as e:
            print(f"Stats error: {e}")
            return {'total_chunks': 0, 'total_documents': 0, 'document_chunks': {}}

    def search_by_document(self, course_id: str, doc_name: str, query_embedding: list[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search within a specific document
        """
        try:
            vector_literal = '[' + ','.join(str(x) for x in query_embedding) + ']'
            
            response = db.rpc(
                'match_embeddings_by_document',
                {
                    'query_embedding': vector_literal,
                    'course_id_param': course_id,  # Fixed parameter name
                    'doc_name_param': doc_name,    # Fixed parameter name
                    'match_count': top_k
                }
            ).execute()
            
            results = []
            for row in response.data:
                results.append({
                    'content': row['content'],
                    'doc_name': row['doc_name'],
                    'chunk_id': row['chunk_id'],
                    'similarity': row.get('similarity', 0.0),
                    'course_id': row['course_id']
                })
            
            return results
            
        except Exception as e:
            print(f"Document search error: {e}")
            return []

    def delete_by_course(self, course_id: str) -> bool:
        """
        Delete all embeddings for a course
        """
        try:
            db.table("embeddings").delete().eq("course_id", course_id).execute()
            return True
        except Exception as e:
            print(f"Delete course error: {e}")
            return False

    def delete_by_document(self, course_id: str, doc_name: str) -> bool:
        """
        Delete all embeddings for a specific document
        """
        try:
            db.table("embeddings").delete().eq("course_id", course_id).eq("doc_name", doc_name).execute()
            return True
        except Exception as e:
            print(f"Delete document error: {e}")
            return False