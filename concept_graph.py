"""Concept prerequisite graph — derived from Course Brain topics.

V3: the graph is no longer an ad-hoc LLM extraction. Nodes are the course's
``course_topics`` rows and edges come from their ``prereq_slugs``, so the
graph, quiz topics, mastery, readiness, and planner all share ONE topic
taxonomy. Overlaid with the student's mastery it answers *"you keep failing X
because you never mastered its prerequisite Y."*

Response shapes are unchanged from the stored-graph era:
``{"concepts": [...], "edges": [{"prerequisite", "concept"}]}`` for the raw
graph and ``{"concepts": nodes, "edges", "blockers", "exists"}`` annotated.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

from dotenv import load_dotenv
from supabase import create_client

import course_brain

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
_supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

WEAK_THRESHOLD = 0.6


def _graph_from_topics(topics: Sequence[course_brain.Topic]) -> Dict[str, Any]:
    """Nodes = topic names (teaching order); edges = prereq_slugs, as names."""
    name_by_slug = {t.slug: t.name for t in topics}
    concepts = [t.name for t in topics]
    edges = [
        {"prerequisite": name_by_slug[pre], "concept": t.name}
        for t in topics
        for pre in t.prereq_slugs
        if pre in name_by_slug and name_by_slug[pre] != t.name
    ]
    return {"concepts": concepts, "edges": edges}


def build_graph(course_id: str) -> Dict[str, Any]:
    """(Re)build the graph by re-synthesizing the Course Brain topics."""
    topics = course_brain.synthesize_topics(course_id)
    if not topics:
        raise RuntimeError(f"No course content to build a concept graph for {course_id}")
    return _graph_from_topics(topics)


def get_graph(course_id: str) -> Optional[Dict[str, Any]]:
    """Return the graph derived from stored course topics, or None if absent."""
    topics = course_brain.get_topics(course_id, auto_generate=False)
    return _graph_from_topics(topics) if topics else None


def _mastery_map(course_id: str, user_id: str) -> Dict[str, float]:
    rows = (_supabase.table("learning_progress").select("topic, mastery_level")
            .eq("user_id", user_id).eq("course_id", course_id).execute().data or [])
    return {(r["topic"] or "").strip().lower(): float(r.get("mastery_level") or 0.0)
            for r in rows if r.get("topic")}


def _match(concept: str, mastery: Dict[str, float]) -> Optional[float]:
    """Bridge legacy mastery labels onto Course Brain concept names."""
    return course_brain.match_mastery(concept, mastery)


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

