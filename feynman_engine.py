"""Feynman mode — "explain it back to me".

The student explains a concept in their own words; the model grades that
explanation against the grounded course material, surfacing what they got right,
what they missed, and what they got wrong. Gaps and misconceptions are fed back
into the spaced-repetition review queue, so the single highest-retention study
technique closes the loop.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from providers import structured_call
from rag.retrieval import retrieve

FEYNMAN_SCHEMA = {
    "type": "object",
    "properties": {
        "score_pct": {"type": "integer", "description": "0-100: how complete & correct the explanation is."},
        "verdict": {"type": "string", "enum": ["solid", "partial", "shaky"]},
        "strengths": {"type": "array", "items": {"type": "string"}, "description": "What the explanation got right."},
        "gaps": {"type": "array", "items": {"type": "string"}, "description": "Important points the student missed."},
        "misconceptions": {"type": "array", "items": {"type": "string"}, "description": "Things the student got wrong."},
        "summary": {"type": "string", "description": "One sentence of coaching."},
    },
    "required": ["score_pct", "verdict", "strengths", "gaps", "misconceptions", "summary"],
}


def _grounding(course_id: str, concept: str) -> List[Dict[str, Any]]:
    return retrieve(concept, course_id, top_k=5)


def evaluate(course_id: str, concept: str, explanation: str, user_id: str = "anonymous") -> Dict[str, Any]:
    """Grade a Feynman explanation and seed review items for its gaps."""
    rows = _grounding(course_id, concept)
    context = "\n\n".join(
        f"[{r.get('doc_name','source')}" + (f", p.{r.get('page') or r.get('slide')}]" if (r.get('page') or r.get('slide')) else "]")
        + f" {(r.get('content') or '').strip()[:600]}"
        for r in rows
    ) or "(no specific course material found)"

    prompt = (
        f"A student is using the Feynman technique to learn \"{concept}\". Grade their "
        "explanation against the COURSE MATERIAL (the source of truth). Judge meaning, not "
        "wording. Be specific and encouraging but honest about gaps and misconceptions.\n\n"
        f"CONCEPT: {concept}\n\n"
        f"STUDENT'S EXPLANATION:\n{explanation}\n\n"
        f"COURSE MATERIAL:\n{context}"
    )
    out = structured_call(
        [{"role": "user", "content": prompt}],
        schema=FEYNMAN_SCHEMA,
        tool_name="feynman_grade",
        model=os.getenv("MODEL_COMPLEX"),
        max_tokens=900,
    )
    if not isinstance(out, dict):
        out = {"score_pct": 0, "verdict": "shaky", "strengths": [], "gaps": [],
               "misconceptions": [], "summary": "Could not grade the explanation."}

    # Close the loop: weak spots become spaced-repetition review items.
    seeded = 0
    weak = list(out.get("gaps", [])) + list(out.get("misconceptions", []))
    if weak:
        try:
            import review_engine
            reference = (rows[0].get("content", "")[:300] if rows else "")
            for item in weak[:3]:
                review_engine.seed_from_mistake(
                    user_id=user_id,
                    course_id=course_id,
                    concept=concept,
                    prompt=f"Re-explain {concept} — make sure you cover: {item}",
                    answer=reference or f"Review {concept} in your course materials.",
                    explanation=f"From a Feynman self-check, this point needs work: {item}",
                    source="feynman",
                )
                seeded += 1
        except Exception as e:  # noqa: BLE001
            print(f"feynman review seeding failed: {e}")

    out["review_items_added"] = seeded
    return out
