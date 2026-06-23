"""Grounded 'explain my mistake' — why a wrong answer is wrong, cited to the
student's own course materials.

On a miss, retrieve the most relevant passages and have the model explain, in a
sentence or two, why the student's answer is wrong and the correct one is right —
anchored to a specific document/page so the student can go read the source.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from providers import structured_call
from rag.retrieval import retrieve

MISTAKE_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {
            "type": "string",
            "description": "1-2 sentences: why the student's answer is wrong and the correct one is right, grounded in the sources.",
        },
        "source_doc": {"type": ["string", "null"], "description": "Document name from the provided sources."},
        "source_page": {"type": ["integer", "null"], "description": "Page/slide number, or null."},
    },
    "required": ["explanation"],
}


def _context(results) -> str:
    parts = []
    for i, r in enumerate(results[:4], 1):
        doc = r.get("doc_name", "unknown")
        page = r.get("page") or r.get("slide")
        head = f"[Source {i}: {doc}" + (f", page {page}" if page else "") + "]"
        parts.append(f"{head}\n{(r.get('content') or '').strip()[:700]}")
    return "\n\n".join(parts)


def explain_mistake(course_id: str, question_text: str, concept: str,
                    selected_text: str, correct_text: str) -> Dict[str, Any]:
    """Return {explanation, source:{doc_name, page}} grounded in course material.

    Best-effort: returns an empty explanation on any failure (callers should treat
    it as optional enrichment, never as required).
    """
    try:
        query = f"{concept} {question_text}".strip() or question_text
        results = retrieve(query, course_id, top_k=4)
        if not results:
            return {"explanation": "", "source": {"doc_name": None, "page": None}}

        prompt = (
            "A student answered a question incorrectly. Using ONLY the course sources below, "
            "explain in 1-2 sentences why their answer is wrong and why the correct answer is right. "
            "Be specific and reference the relevant idea. Cite the source you used.\n\n"
            f"QUESTION:\n{question_text}\n\n"
            f"STUDENT'S ANSWER (wrong):\n{selected_text}\n\n"
            f"CORRECT ANSWER:\n{correct_text}\n\n"
            f"COURSE SOURCES:\n{_context(results)}"
        )
        out = structured_call(
            [{"role": "user", "content": prompt}],
            schema=MISTAKE_SCHEMA,
            tool_name="mistake_explanation",
            model=os.getenv("MODEL_DEFAULT"),
            max_tokens=300,
        )
        if not isinstance(out, dict):
            return {"explanation": "", "source": {"doc_name": None, "page": None}}
        return {
            "explanation": out.get("explanation", ""),
            "source": {"doc_name": out.get("source_doc"), "page": out.get("source_page")},
        }
    except Exception as e:  # noqa: BLE001
        print(f"explain_mistake failed: {e}")
        return {"explanation": "", "source": {"doc_name": None, "page": None}}
