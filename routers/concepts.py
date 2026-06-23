from fastapi import APIRouter, HTTPException
from deps import *  # noqa: F401,F403  shared state, engines, helpers, stdlib re-exports

import concept_graph

router = APIRouter()


@router.post("/api/concept-graph/{course_id}")
async def build_concept_graph_endpoint(course_id: str):
    """(Re)build and persist the course's concept prerequisite graph."""
    try:
        return concept_graph.build_graph(course_id)
    except Exception as e:
        print(f"Concept graph build failed: {e}")
        raise HTTPException(500, detail=f"Concept graph build failed: {e}")


@router.get("/api/concept-graph/{course_id}/{user_id}")
async def get_concept_graph_endpoint(course_id: str, user_id: str):
    """Return the concept graph annotated with the user's mastery + blockers."""
    try:
        return concept_graph.graph_with_mastery(course_id, user_id)
    except Exception as e:
        print(f"Concept graph fetch failed: {e}")
        raise HTTPException(500, detail=f"Concept graph fetch failed: {e}")
