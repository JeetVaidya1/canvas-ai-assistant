from fastapi import APIRouter, Form, HTTPException
from deps import *  # noqa: F401,F403  shared state, engines, helpers, stdlib re-exports

import json
import socratic_engine

router = APIRouter()


@router.post("/api/socratic")
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
        raise HTTPException(500, detail=f"Socratic turn failed: {e}")
