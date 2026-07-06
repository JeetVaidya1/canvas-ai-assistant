from fastapi import APIRouter, BackgroundTasks, Form, HTTPException

import logging

logger = logging.getLogger(__name__)

import canvas_engine
import course_brain

router = APIRouter()


@router.post("/api/import-canvas")
async def import_canvas_endpoint(
    background_tasks: BackgroundTasks,
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
        result = canvas_engine.import_course(
            canvas_base_url, canvas_token, canvas_course_id, course_id, ingest_materials
        )
        # Materials (possibly) changed -> rebuild the Course Brain off-request.
        if ingest_materials:
            background_tasks.add_task(course_brain.rebuild_topics_safely, course_id)
        return result
    except Exception as e:
        print(f"Canvas import failed: {e}")
        logger.exception("Canvas import failed")
        raise HTTPException(502, detail="Canvas import failed")
