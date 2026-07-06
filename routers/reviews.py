from fastapi import APIRouter, Depends, Form, HTTPException
from auth import current_user_id

import logging

logger = logging.getLogger(__name__)

import review_engine

router = APIRouter()


@router.get("/api/reviews/{course_id}")
async def get_reviews_endpoint(course_id: str, user_id: str = Depends(current_user_id), include_all: bool = False):
    """The mistake-driven review queue for a course/user, due items first."""
    try:
        return review_engine.get_due(course_id, user_id, include_all=include_all)
    except Exception as e:
        print(f"Review queue fetch failed: {e}")
        logger.exception("Review queue fetch failed")
        raise HTTPException(500, detail="Review queue fetch failed")


@router.post("/api/reviews/{item_id}/grade")
async def grade_review_endpoint(
    item_id: str,
    grade: int = Form(...),
    user_id: str = Depends(current_user_id),
):
    """Apply an SM-2 review (grade 0-5) to a queue item."""
    if not 0 <= grade <= 5:
        raise HTTPException(400, detail="grade must be between 0 and 5")
    try:
        return review_engine.grade(item_id, user_id, grade)
    except KeyError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        print(f"Review grade failed: {e}")
        logger.exception("Review grade failed")
        raise HTTPException(500, detail="Review grade failed")
