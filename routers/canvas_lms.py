from fastapi import APIRouter, Form, HTTPException
from deps import *  # noqa: F401,F403  shared state, engines, helpers, stdlib re-exports

import canvas_engine

router = APIRouter()


@router.post("/api/import-canvas")
async def import_canvas_endpoint(
    canvas_base_url: str = Form(...),     # e.g. https://canvas.instructure.com
    canvas_token: str = Form(...),
    canvas_course_id: str = Form(...),
    course_id: str = Form(...),           # the Vindexa course to import into
    ingest_materials: bool = Form(True),
):
    """Import a Canvas course: syllabus + due dates + materials, with exam-date
    detection for the planner."""
    if not (canvas_base_url and canvas_token and canvas_course_id and course_id):
        raise HTTPException(400, detail="base URL, token, Canvas course id, and course id are required")
    try:
        return canvas_engine.import_course(
            canvas_base_url, canvas_token, canvas_course_id, course_id, ingest_materials
        )
    except Exception as e:
        print(f"Canvas import failed: {e}")
        raise HTTPException(502, detail=f"Canvas import failed: {e}")
