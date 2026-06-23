"""Shared class courses — the distribution wedge.

The first student to build a course can publish it to a browsable catalog with a
short share code. Classmates join the same grounded course (same materials, same
generated artifacts) while keeping their own per-user mastery and review state.
This is the network-effect loop: one upload seeds a whole class.
"""
from __future__ import annotations

import os
import secrets
import string
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
_supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

_CODE_ALPHABET = string.ascii_uppercase + string.digits  # no lowercase to avoid confusion


def _new_code(length: int = 6) -> str:
    """A short, unique, human-typeable share code."""
    for _ in range(20):
        code = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(length))
        exists = _supabase.table("shared_courses").select("share_code").eq("share_code", code).execute().data
        if not exists:
            return code
    # Extremely unlikely; widen on collision.
    return "".join(secrets.choice(_CODE_ALPHABET) for _ in range(length + 2))


def _course_title(course_id: str) -> str:
    r = _supabase.table("courses").select("title").eq("course_id", course_id).limit(1).execute()
    return (r.data[0].get("title") if r.data else None) or course_id


def publish(course_id: str, user_id: str, subject: str = "", school: str = "",
            term: str = "", description: str = "") -> Dict[str, Any]:
    """Publish (or update) a course in the shared catalog. Idempotent per course:
    re-publishing keeps the same share_code."""
    existing = _supabase.table("shared_courses").select("*").eq("course_id", course_id).limit(1).execute().data
    if existing:
        row = existing[0]
        _supabase.table("shared_courses").update({
            "title": _course_title(course_id),
            "subject": subject or row.get("subject"),
            "school": school or row.get("school"),
            "term": term or row.get("term"),
            "description": description or row.get("description"),
        }).eq("course_id", course_id).execute()
        return {"course_id": course_id, "share_code": row["share_code"], "republished": True}

    code = _new_code()
    _supabase.table("shared_courses").insert({
        "course_id": course_id,
        "share_code": code,
        "title": _course_title(course_id),
        "subject": subject,
        "school": school,
        "term": term,
        "description": description,
        "published_by": user_id,
        "join_count": 0,
        "created_at": datetime.utcnow().isoformat(),
    }).execute()
    return {"course_id": course_id, "share_code": code, "republished": False}


def get_share_info(course_id: str) -> Optional[Dict[str, Any]]:
    r = _supabase.table("shared_courses").select("*").eq("course_id", course_id).limit(1).execute()
    return r.data[0] if r.data else None


def catalog(query: str = "", limit: int = 50) -> List[Dict[str, Any]]:
    """Browse published courses, most-joined first. ``query`` filters title/subject/school."""
    rows = (_supabase.table("shared_courses").select("*")
            .order("join_count", desc=True).limit(limit).execute().data or [])
    q = (query or "").strip().lower()
    if q:
        def hit(r: Dict[str, Any]) -> bool:
            return any(q in str(r.get(f) or "").lower() for f in ("title", "subject", "school", "term"))
        rows = [r for r in rows if hit(r)]
    return [{
        "course_id": r["course_id"],
        "share_code": r["share_code"],
        "title": r.get("title"),
        "subject": r.get("subject"),
        "school": r.get("school"),
        "term": r.get("term"),
        "description": r.get("description"),
        "join_count": r.get("join_count", 0),
    } for r in rows]


def join_by_code(share_code: str, user_id: str) -> Dict[str, Any]:
    """Join a published course by its share code. Idempotent per (user, course)."""
    code = (share_code or "").strip().upper()
    found = _supabase.table("shared_courses").select("*").eq("share_code", code).limit(1).execute().data
    if not found:
        raise KeyError("No class found for that code.")
    shared = found[0]
    course_id = shared["course_id"]

    already = (_supabase.table("course_memberships").select("id")
               .eq("user_id", user_id).eq("course_id", course_id).limit(1).execute().data)
    newly_joined = False
    if not already:
        _supabase.table("course_memberships").insert({
            "user_id": user_id,
            "course_id": course_id,
            "role": "member",
            "joined_at": datetime.utcnow().isoformat(),
        }).execute()
        _supabase.table("shared_courses").update(
            {"join_count": (shared.get("join_count", 0) or 0) + 1}
        ).eq("course_id", course_id).execute()
        newly_joined = True

    return {"course_id": course_id, "title": shared.get("title"), "newly_joined": newly_joined}


def my_memberships(user_id: str) -> List[str]:
    rows = (_supabase.table("course_memberships").select("course_id")
            .eq("user_id", user_id).execute().data or [])
    return [r["course_id"] for r in rows]
