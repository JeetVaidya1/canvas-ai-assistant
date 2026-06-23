"""Context pack — a paste-ready study brief for any AI (Claude Project, Custom
GPT, Cursor, plain chat).

Inverts the wrapper story: instead of wrapping a model, Vindexa emits the
structured, grounded substrate — the course's key material plus *this student's*
weak areas — that makes any model useful for this specific course.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from supabase import create_client

from rag.retrieval import retrieve

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
_supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def _course_title(course_id: str) -> str:
    try:
        r = _supabase.table("courses").select("title").eq("course_id", course_id).limit(1).execute()
        if r.data and r.data[0].get("title"):
            return r.data[0]["title"]
    except Exception:  # noqa: BLE001
        pass
    return course_id


def _weak_topics(course_id: str, user_id: str) -> List[Dict[str, Any]]:
    rows = (_supabase.table("learning_progress").select("topic, mastery_level")
            .eq("user_id", user_id).eq("course_id", course_id).execute().data or [])
    rows = [r for r in rows if r.get("topic")]
    rows.sort(key=lambda r: float(r.get("mastery_level") or 0.0))
    return [{"topic": r["topic"], "mastery_pct": round(float(r.get("mastery_level") or 0.0) * 100)} for r in rows[:6]]


def build_context_pack(course_id: str, user_id: str) -> str:
    """Return a Markdown context pack the student can paste into any AI."""
    title = _course_title(course_id)
    weak = _weak_topics(course_id, user_id)

    lines: List[str] = [
        f"# Study context: {title}",
        "",
        "You are my study tutor for the course above. Use the grounded course "
        "excerpts below as the source of truth. Prioritize my weak areas. When you "
        "quiz me, base questions on this material and explain mistakes against it.",
        "",
    ]

    if weak:
        lines.append("## My weak areas (focus here)")
        for w in weak:
            lines.append(f"- {w['topic']} — currently ~{w['mastery_pct']}% mastery")
        lines.append("")

    # Curate grounded excerpts: for each weak topic (or the course generally),
    # pull the top passages so the model has the actual source material.
    queries = [w["topic"] for w in weak] or ["key concepts, definitions, and theorems of this course"]
    seen = set()
    lines.append("## Grounded course excerpts")
    for q in queries[:5]:
        for r in retrieve(q, course_id, top_k=3):
            key = (r.get("doc_name"), r.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            doc = r.get("doc_name", "source")
            page = r.get("page") or r.get("slide")
            cite = f"{doc}" + (f", p.{page}" if page else "")
            content = (r.get("content") or "").strip()[:600]
            if content:
                lines.append(f"### {q} — [{cite}]")
                lines.append(content)
                lines.append("")

    lines.append("---")
    lines.append("Start by asking me 3 questions targeting my weakest area.")
    return "\n".join(lines)
