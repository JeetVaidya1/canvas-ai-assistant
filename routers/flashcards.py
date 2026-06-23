from fastapi import APIRouter, Form, HTTPException
from deps import *  # noqa: F401,F403  shared state, engines, helpers, stdlib re-exports

import json
import flashcard_engine

router = APIRouter()


@router.post("/flashcards/save")
async def save_flashcards_endpoint(
    course_id: str = Form(...),
    cards: str = Form(...),          # JSON: [{"q": "...", "a": "..."}]
    note_id: str | None = Form(None),
):
    """Persist a batch of flashcards into the course review deck."""
    if not course_id:
        raise HTTPException(400, detail="Course ID is required")
    try:
        parsed = json.loads(cards)
    except json.JSONDecodeError:
        raise HTTPException(400, detail="Invalid cards JSON")
    if not isinstance(parsed, list):
        raise HTTPException(400, detail="cards must be a JSON array")

    try:
        return flashcard_engine.save_cards(course_id, parsed, note_id)
    except Exception as e:
        print(f"Flashcard save failed: {e}")
        raise HTTPException(500, detail=f"Flashcard save failed: {e}")


@router.get("/flashcards/{course_id}")
async def get_flashcards_endpoint(course_id: str, user_id: str = Depends(current_user_id)):
    """Return the course deck with per-user SR state, due cards first."""
    try:
        return flashcard_engine.get_deck(course_id, user_id)
    except Exception as e:
        print(f"Flashcard deck fetch failed: {e}")
        raise HTTPException(500, detail=f"Flashcard deck fetch failed: {e}")


@router.post("/flashcards/review")
async def review_flashcard_endpoint(
    card_id: str = Form(...),
    grade: int = Form(...),
    user_id: str = Depends(current_user_id),
):
    """Apply an SM-2 review (grade 0-5) and return the updated schedule."""
    if not 0 <= grade <= 5:
        raise HTTPException(400, detail="grade must be between 0 and 5")
    try:
        return flashcard_engine.review_card(card_id, user_id, grade)
    except Exception as e:
        print(f"Flashcard review failed: {e}")
        raise HTTPException(500, detail=f"Flashcard review failed: {e}")
