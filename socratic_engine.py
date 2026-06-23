"""Socratic tutor — the answer to "why not just use ChatGPT".

It will NOT hand over the answer. Grounded in the student's own course material,
it asks one guiding question at a time, builds on what the student says, and only
scaffolds toward the answer (never states it outright) — unless the student
explicitly gives up, in which case it gives a hint, not the full solution.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv

from providers import make_client
from rag.retrieval import retrieve

load_dotenv()
_client = make_client()

_SYSTEM = (
    "You are a Socratic tutor for a specific course. Your goal is to make the student "
    "reason their way to understanding — you must NOT give away the final answer.\n"
    "RULES:\n"
    "- Ask ONE focused, guiding question at a time. Keep it short.\n"
    "- Build on what the student just said; acknowledge correct steps, gently probe wrong ones.\n"
    "- Stay grounded in the COURSE CONTEXT provided; use its terminology and examples.\n"
    "- Never state the final answer or do the work for them. If they're stuck, give a small hint "
    "or a simpler sub-question, not the solution.\n"
    "- Only if the student EXPLICITLY gives up (e.g. 'just tell me', 'I give up') may you reveal "
    "a worked explanation — and even then, end by checking their understanding.\n"
    "- If the question is outside the course material, say so briefly and steer back."
)


def _context(course_id: str, query: str) -> str:
    rows = retrieve(query, course_id, top_k=5)
    parts = []
    for r in rows:
        doc = r.get("doc_name", "source")
        page = r.get("page") or r.get("slide")
        head = f"[{doc}" + (f", p.{page}" if page else "") + "]"
        parts.append(f"{head} {(r.get('content') or '').strip()[:600]}")
    return "\n\n".join(parts) if parts else "(no specific course material found)"


def respond(course_id: str, message: str, history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
    """Return the tutor's next Socratic turn.

    ``history`` is a list of {role: 'user'|'assistant', content: str}. The latest
    student ``message`` is appended. Retrieval is keyed off the most recent student
    turn so grounding tracks the conversation.
    """
    history = history or []
    grounding = _context(course_id, message)

    messages = [{"role": "system", "content": f"{_SYSTEM}\n\nCOURSE CONTEXT:\n{grounding}"}]
    # Keep the last few turns for continuity without unbounded growth.
    for turn in history[-8:]:
        role = "assistant" if turn.get("role") == "assistant" else "user"
        messages.append({"role": role, "content": str(turn.get("content", ""))})
    messages.append({"role": "user", "content": message})

    resp = _client.chat.completions.create(
        model=os.getenv("MODEL_COMPLEX"),
        messages=messages,
        temperature=0.4,
        max_tokens=500,
    )
    reply = (resp.choices[0].message.content or "").strip()
    return {"reply": reply}
