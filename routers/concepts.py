from fastapi import APIRouter, Depends, HTTPException
from auth import current_user_id

import logging

logger = logging.getLogger(__name__)

import concept_graph

router = APIRouter()


@router.post("/api/concept-graph/{course_id}")
async def build_concept_graph_endpoint(course_id: str):
    """(Re)build and persist the course's concept prerequisite graph."""
    try:
        return concept_graph.build_graph(course_id)
    except Exception as e:
        print(f"Concept graph build failed: {e}")
        logger.exception("Concept graph build failed")
        raise HTTPException(500, detail="Concept graph build failed")


@router.get("/api/concept-graph/{course_id}/{user_id}")
async def get_concept_graph_endpoint(course_id: str, user_id: str, _auth: str = Depends(current_user_id)):
    user_id = _auth
    """Return the concept graph annotated with the user's mastery + blockers."""
    try:
        return concept_graph.graph_with_mastery(course_id, user_id)
    except Exception as e:
        print(f"Concept graph fetch failed: {e}")
        logger.exception("Concept graph fetch failed")
        raise HTTPException(500, detail="Concept graph fetch failed")
