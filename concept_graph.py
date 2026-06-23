"""Concept prerequisite graph — a per-course DAG of what must be learned before what.

Extracted once from the course material via guaranteed-schema tool use and cached
in ``concept_graphs``. Overlaid with the student's mastery, it answers the
question generic tools can't: *"you keep failing X because you never mastered its
prerequisite Y."*
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import create_client

from providers import structured_call

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
_supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

WEAK_THRESHOLD = 0.6

CONCEPT_GRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The course's key concepts (8-20), foundational first.",
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "prerequisite": {"type": "string", "description": "Concept that must be understood first."},
                    "concept": {"type": "string", "description": "Concept that depends on the prerequisite."},
                },
                "required": ["prerequisite", "concept"],
            },
            "description": "Directed prerequisite edges (prerequisite -> concept). Keep it a DAG.",
        },
    },
    "required": ["concepts", "edges"],
}


def _course_sample(course_id: str) -> str:
    try:
        from deps import exam_generator
        return exam_generator.get_course_content_sample(course_id) or ""
    except Exception as e:  # noqa: BLE001
        print(f"concept-graph content sample failed: {e}")
        return ""


def build_graph(course_id: str) -> Dict[str, Any]:
    """Extract and persist the prerequisite graph for a course."""
    context = _course_sample(course_id)
    prompt = (
        "From the course materials below, build a concept prerequisite graph.\n"
        "- List the key concepts (8-20), foundational ones first.\n"
        "- Add directed edges prerequisite -> concept where one concept must be "
        "understood before another. Keep it acyclic. Only include edges grounded "
        "in the material.\n\n"
        f"COURSE MATERIALS:\n{context[:6000]}"
    )
    out = structured_call(
        [{"role": "user", "content": prompt}],
        schema=CONCEPT_GRAPH_SCHEMA,
        tool_name="concept_graph",
        model=os.getenv("MODEL_COMPLEX"),
        max_tokens=1500,
    )
    concepts = [str(c).strip() for c in (out.get("concepts") or []) if str(c).strip()]
    edges = [
        {"prerequisite": str(e["prerequisite"]).strip(), "concept": str(e["concept"]).strip()}
        for e in (out.get("edges") or [])
        if e.get("prerequisite") and e.get("concept")
        and str(e["prerequisite"]).strip() != str(e["concept"]).strip()
    ]
    graph = {"concepts": concepts, "edges": edges}

    _supabase.table("concept_graphs").upsert({
        "course_id": course_id,
        "graph": graph,
        "created_at": datetime.utcnow().isoformat(),
    }).execute()
    return graph


def get_graph(course_id: str) -> Optional[Dict[str, Any]]:
    resp = _supabase.table("concept_graphs").select("graph").eq("course_id", course_id).limit(1).execute()
    return resp.data[0]["graph"] if resp.data else None


def _mastery_map(course_id: str, user_id: str) -> Dict[str, float]:
    rows = (_supabase.table("learning_progress").select("topic, mastery_level")
            .eq("user_id", user_id).eq("course_id", course_id).execute().data or [])
    return {(r["topic"] or "").strip().lower(): float(r.get("mastery_level") or 0.0)
            for r in rows if r.get("topic")}


def _match(concept: str, mastery: Dict[str, float]) -> Optional[float]:
    c = concept.strip().lower()
    if c in mastery:
        return mastery[c]
    for k, v in mastery.items():
        if k and (k in c or c in k):
            return v
    return None


def graph_with_mastery(course_id: str, user_id: str) -> Dict[str, Any]:
    """Return the graph annotated with mastery + 'blockers': weak concepts whose
    prerequisites are also weak (fix the prerequisite first)."""
    graph = get_graph(course_id)
    if not graph:
        # Lazily build on first request so the client needn't orchestrate it.
        try:
            graph = build_graph(course_id)
        except Exception as e:  # noqa: BLE001
            print(f"concept graph lazy build failed: {e}")
            return {"concepts": [], "edges": [], "blockers": [], "exists": False}

    mastery = _mastery_map(course_id, user_id)

    def m(concept: str) -> Optional[float]:
        return _match(concept, mastery)

    nodes = []
    for c in graph.get("concepts", []):
        val = m(c)
        nodes.append({"concept": c, "mastery_pct": round((val or 0.0) * 100, 1), "has_data": val is not None})

    blockers = []
    for e in graph.get("edges", []):
        pre, con = e["prerequisite"], e["concept"]
        pre_m, con_m = m(pre), m(con)
        # The dependent concept is weak AND its prerequisite is also weak.
        if con_m is not None and con_m < WEAK_THRESHOLD and (pre_m is None or pre_m < WEAK_THRESHOLD):
            blockers.append({
                "concept": con,
                "prerequisite": pre,
                "concept_pct": round((con_m or 0.0) * 100, 1),
                "prerequisite_pct": round((pre_m or 0.0) * 100, 1),
            })

    # Strongest signal first: lowest prerequisite mastery.
    blockers.sort(key=lambda b: b["prerequisite_pct"])
    return {"concepts": nodes, "edges": graph.get("edges", []), "blockers": blockers, "exists": True}
