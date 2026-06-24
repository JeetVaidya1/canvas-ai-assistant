# notes_engine/retrieval.py — Vector-store retrieval and content extraction
from typing import Dict, List, Any

from .config import openai_client, vector_store, EMBED_MODEL, RAG_TOP_K_FILE
from .helpers import _is_dict, _safe_get


def _sanitize_chunks(raw: Any, file_name: str = "") -> List[Dict[str, Any]]:
    """Filter out None/invalid rows and rows with empty content."""
    if not raw:
        return []
    clean: List[Dict[str, Any]] = []
    for c in raw:
        if not _is_dict(c):
            continue
        # Optional: if a file filter is intended, enforce it here too
        if file_name:
            doc = _safe_get(c, "doc_name") or _safe_get(c, "file") or ""
            if doc != file_name:
                # keep only exact file matches when we're in filename mode
                continue
        text = (_safe_get(c, "content") or "").strip()
        if not text:
            continue
        clean.append(c)
    return clean

def _get_file_chunks_by_meta(course_id: str, file_name: str) -> List[Dict[str, Any]]:
    try:
        if hasattr(vector_store, "query_by_metadata"):
            raw = vector_store.query_by_metadata(
                course_id,
                filters={"doc_name": file_name},
                top_k=RAG_TOP_K_FILE
            ) or []
            chunks = _sanitize_chunks(raw)  # metadata route shouldn't need filename check
        else:
            emb = openai_client.embeddings.create(model=EMBED_MODEL, input=[file_name])
            raw = vector_store.query(course_id, emb.data[0].embedding, top_k=RAG_TOP_K_FILE) or []
            chunks = _sanitize_chunks(raw, file_name=file_name)

        def _sort_key(c):
            return (
                _safe_get(c, "page") or _safe_get(c, "page_num") or 0,
                _safe_get(c, "chunk_id") or _safe_get(c, "index") or 0
            )
        chunks.sort(key=_sort_key)
        return chunks
    except Exception as e:
        print(f"❌ Retrieval failed for {file_name}: {e}")
        return []

def extract_content_from_files(course_id: str, file_names: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for fname in file_names:
        chunks = _get_file_chunks_by_meta(course_id, fname) or []
        if not chunks:
            out[fname] = f"Content not found for {fname}"
            continue
        texts: List[str] = []
        for c in chunks:
            if not _is_dict(c):
                continue
            t = (_safe_get(c, "content") or "").strip()
            if t:
                texts.append(t)
        out[fname] = "\n\n".join(texts) if texts else f"Content not found for {fname}"
    return out
