from fastapi import APIRouter, Form, HTTPException
from deps import *  # noqa: F401,F403  shared state, engines, helpers, stdlib re-exports

import review_engine

router = APIRouter()


@router.get("/api/reviews/{course_id}")
async def get_reviews_endpoint(course_id: str, user_id: str = "anonymous", include_all: bool = False):
    """The mistake-driven review queue for a course/user, due items first."""
    try:
        return review_engine.get_due(course_id, user_id, include_all=include_all)
    except Exception as e:
        print(f"Review queue fetch failed: {e}")
        raise HTTPException(500, detail=f"Review queue fetch failed: {e}")


@router.post("/api/reviews/{item_id}/grade")
async def grade_review_endpoint(
    item_id: str,
    grade: int = Form(...),
    user_id: str = Form("anonymous"),
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
        raise HTTPException(500, detail=f"Review grade failed: {e}")
