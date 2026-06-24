# notes_engine/persistence.py — Supabase-backed notes CRUD
import os
import uuid
from typing import Dict, List, Any
from datetime import datetime

from .generation import extract_topics_from_content


def save_notes_to_db(course_id: str, title: str, content: str, source_files: List[str],
                     topic: str = "", note_id: str = None) -> Dict[str, Any]:
    try:
        from supabase import create_client
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        topics = extract_topics_from_content(content)
        word_count = len((content or "").split())
        reading_time = f"{max(1, word_count // 200)} min"

        note_data = {
            "course_id": course_id,
            "title": title,
            "content": content,
            "source_files": source_files,
            "topic_focus": topic,
            "topics": topics[:8],
            "word_count": word_count,
            "reading_time": reading_time,
            "updated_at": datetime.utcnow().isoformat()
        }

        if note_id:
            result = supabase.table("notes").update(note_data).eq("id", note_id).execute()
            saved_note = result.data[0] if result.data else None
        else:
            note_data["id"] = str(uuid.uuid4())
            note_data["created_at"] = datetime.utcnow().isoformat()
            result = supabase.table("notes").insert(note_data).execute()
            saved_note = result.data[0] if result.data else None

        if saved_note:
            return {"status": "success", "note": saved_note}
        else:
            return {"status": "error", "message": "Failed to save note to database"}

    except Exception as e:
        print(f"❌ Note saving error: {e}")
        return {"status": "error", "message": f"Database error: {str(e)}"}

def get_notes_from_db(course_id: str) -> List[Dict[str, Any]]:
    try:
        from supabase import create_client
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        res = supabase.table("notes").select("*").eq("course_id", course_id).order("updated_at", desc=True).execute()
        return res.data if res.data else []
    except Exception as e:
        print(f"❌ Notes retrieval error: {e}")
        return []

def delete_note_from_db(note_id: str) -> bool:
    try:
        from supabase import create_client
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        res = supabase.table("notes").delete().eq("id", note_id).execute()
        return len(res.data) > 0 if res.data else True
    except Exception as e:
        print(f"❌ Note deletion error: {e}")
        return False
