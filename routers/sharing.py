from fastapi import APIRouter, Depends, Form, HTTPException
from auth import current_user_id

import logging

logger = logging.getLogger(__name__)

import sharing_engine

router = APIRouter()


@router.post("/api/courses/publish")
async def publish_course_endpoint(
    course_id: str = Form(...),
    user_id: str = Depends(current_user_id),
    subject: str = Form(""),
    school: str = Form(""),
    term: str = Form(""),
    description: str = Form(""),
):
    """Publish a course to the shared class catalog (returns a share code)."""
    if not course_id:
        raise HTTPException(400, detail="course_id is required")
    try:
        return sharing_engine.publish(course_id, user_id, subject, school, term, description)
    except Exception as e:
        print(f"Publish failed: {e}")
        logger.exception("Publish failed")
        raise HTTPException(500, detail="Publish failed")


@router.get("/api/shared-courses")
async def catalog_endpoint(q: str = ""):
    """Browse published courses (most-joined first)."""
    try:
        return {"courses": sharing_engine.catalog(q)}
    except Exception as e:
        print(f"Catalog fetch failed: {e}")
        logger.exception("Catalog fetch failed")
        raise HTTPException(500, detail="Catalog fetch failed")


@router.get("/api/courses/{course_id}/share")
async def share_info_endpoint(course_id: str):
    """Return a course's share info, or null if it isn't published."""
    try:
        return sharing_engine.get_share_info(course_id)
    except Exception as e:
        print(f"Share info fetch failed: {e}")
        logger.exception("Share info fetch failed")
        raise HTTPException(500, detail="Share info fetch failed")


@router.post("/api/courses/join")
async def join_course_endpoint(
    share_code: str = Form(...),
    user_id: str = Depends(current_user_id),
):
    """Join a published course by share code."""
    try:
        return sharing_engine.join_by_code(share_code, user_id)
    except KeyError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        print(f"Join failed: {e}")
        logger.exception("Join failed")
        raise HTTPException(500, detail="Join failed")
