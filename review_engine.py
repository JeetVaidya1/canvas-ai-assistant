"""Review engine — the per-user, mistake-driven spaced-repetition queue.

This is the spine of the closed loop. Every wrong quiz/exam answer seeds a
``review_items`` row (due today) for the concept the student missed. The queue
is then scheduled with SM-2 (reused from ``flashcard_engine``), so mistakes
resurface on a spacing schedule until mastered — the student never has to decide
what to review; the system decides from evidence.
"""
from __future__ import annotations

import os
import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import create_client

from flashcard_engine import sm2, DEFAULT_EASE

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
_supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def seed_from_mistake(user_id: str, course_id: str, concept: str, prompt: str,
                      answer: str, explanation: str = "", source: str = "quiz") -> Optional[str]:
    """Add a review item for a missed concept (due today). Idempotent per
    (user, course, prompt): if an active item already exists, do nothing.

    Returns the item id (new or existing), or None on failure.
    """
    try:
        existing = (_supabase.table("review_items").select("id")
                    .eq("user_id", user_id).eq("course_id", course_id)
                    .eq("prompt", prompt).eq("status", "active")
                    .limit(1).execute().data)
        if existing:
            return existing[0]["id"]

        item_id = str(uuid.uuid4())
        _supabase.table("review_items").insert({
            "id": item_id,
            "user_id": user_id,
            "course_id": course_id,
            "concept": concept or "general",
            "prompt": prompt,
            "answer": answer,
            "explanation": explanation,
            "source": source,
            "ease": DEFAULT_EASE,
            "interval": 0,
            "repetitions": 0,
            "due_date": date.today().isoformat(),
            "status": "active",
            "created_at": datetime.utcnow().isoformat(),
        }).execute()
        return item_id
    except Exception as e:  # noqa: BLE001  must never break grading
        print(f"seed_from_mistake failed: {e}")
        return None


def get_due(course_id: str, user_id: str, include_all: bool = False) -> Dict[str, Any]:
    """Active review items for a course/user, due first.

    ``include_all`` returns upcoming items too (for a full queue view); by default
    only items due today or earlier are returned.
    """
    items = (_supabase.table("review_items").select("*")
             .eq("course_id", course_id).eq("user_id", user_id)
             .eq("status", "active").execute().data or [])
    today = date.today().isoformat()

    enriched = []
    for it in items:
        due = (it.get("due_date") is None) or (str(it["due_date"])[:10] <= today)
        if not include_all and not due:
            continue
        enriched.append({
            "id": it["id"],
            "concept": it.get("concept"),
            "prompt": it.get("prompt"),
            "answer": it.get("answer"),
            "explanation": it.get("explanation"),
            "source": it.get("source"),
            "due": due,
            "due_date": it.get("due_date"),
        })

    enriched.sort(key=lambda c: (not c["due"], c["due_date"] or ""))
    due_count = sum(1 for c in enriched if c["due"])
    return {"items": enriched, "due_count": due_count, "total": len(enriched)}


def due_count(course_id: str, user_id: str) -> int:
    """Cheap count of items due now (for badges)."""
    return get_due(course_id, user_id)["due_count"]


def grade(item_id: str, user_id: str, grade_value: int) -> Dict[str, Any]:
    """Apply an SM-2 review to a queue item. A high grade that schedules the item
    far out effectively 'graduates' it; we keep status active so it keeps cycling."""
    rows = (_supabase.table("review_items").select("*")
            .eq("id", item_id).eq("user_id", user_id).limit(1).execute().data)
    if not rows:
        raise KeyError(f"Review item {item_id} not found")
    prev = rows[0]

    state = sm2(grade_value, float(prev.get("ease") or DEFAULT_EASE),
                int(prev.get("interval") or 0), int(prev.get("repetitions") or 0))
    _supabase.table("review_items").update({
        **state,
        "last_reviewed": datetime.utcnow().isoformat(),
    }).eq("id", item_id).execute()

    return {"item_id": item_id, **state}
