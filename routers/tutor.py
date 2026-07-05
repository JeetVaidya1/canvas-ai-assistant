from fastapi import APIRouter, Depends, Form, HTTPException
from auth import current_user_id
from rate_limit import ai_rate_limit

import logging

logger = logging.getLogger(__name__)

import json
import socratic_engine
import feynman_engine

router = APIRouter()


@router.post("/api/socratic", dependencies=[Depends(ai_rate_limit)])
async def socratic_endpoint(
    course_id: str = Form(...),
    message: str = Form(...),
    history: str = Form("[]"),   # JSON: [{"role": "...", "content": "..."}]
):
    """One Socratic tutor turn — grounded, never gives the answer away."""
    if not course_id or not message.strip():
        raise HTTPException(400, detail="course_id and message are required")
    try:
        hist = json.loads(history)
        if not isinstance(hist, list):
            hist = []
    except json.JSONDecodeError:
        hist = []
    try:
        return socratic_engine.respond(course_id, message.strip(), hist)
    except Exception as e:
        print(f"Socratic turn failed: {e}")
        logger.exception("Socratic turn failed")
        raise HTTPException(500, detail="Socratic turn failed")


@router.post("/api/feynman", dependencies=[Depends(ai_rate_limit)])
async def feynman_endpoint(
    course_id: str = Form(...),
    concept: str = Form(...),
    explanation: str = Form(...),
    user_id: str = Depends(current_user_id),
):
    """Grade a Feynman-technique explanation against the course material and seed
    review items for the gaps."""
    if not (course_id and concept.strip() and explanation.strip()):
        raise HTTPException(400, detail="course_id, concept, and explanation are required")
    try:
        return feynman_engine.evaluate(course_id, concept.strip(), explanation.strip(), user_id)
    except Exception as e:
        print(f"Feynman grading failed: {e}")
        logger.exception("Feynman grading failed")
        raise HTTPException(500, detail="Feynman grading failed")
