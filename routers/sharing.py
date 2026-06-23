from fastapi import APIRouter, Form, HTTPException
from deps import *  # noqa: F401,F403  shared state, engines, helpers, stdlib re-exports

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
        raise HTTPException(500, detail=f"Publish failed: {e}")


@router.get("/api/shared-courses")
async def catalog_endpoint(q: str = ""):
    """Browse published courses (most-joined first)."""
    try:
        return {"courses": sharing_engine.catalog(q)}
    except Exception as e:
        print(f"Catalog fetch failed: {e}")
        raise HTTPException(500, detail=f"Catalog fetch failed: {e}")


@router.get("/api/courses/{course_id}/share")
async def share_info_endpoint(course_id: str):
    """Return a course's share info, or null if it isn't published."""
    try:
        return sharing_engine.get_share_info(course_id)
    except Exception as e:
        print(f"Share info fetch failed: {e}")
        raise HTTPException(500, detail=f"Share info fetch failed: {e}")


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
        raise HTTPException(500, detail=f"Join failed: {e}")
