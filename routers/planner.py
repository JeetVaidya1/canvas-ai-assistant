from fastapi import APIRouter, Form, HTTPException
from deps import *  # noqa: F401,F403  shared state, engines, helpers, stdlib re-exports

import planner_engine

router = APIRouter()


@router.post("/api/generate-study-plan")
async def generate_study_plan_endpoint(
    course_id: str = Form(...),
    days_available: int | None = Form(None),
    hours_per_day: float | None = Form(None),
    exam_date: str | None = Form(None),
    user_id: str = Depends(current_user_id),
    mode: str = Form("balanced"),
):
    """Generate and persist a day-by-day study plan for a course."""
    if not course_id:
        raise HTTPException(400, detail="Course ID is required")

    validation = await validate_course_for_practice(course_id)
    if validation.get("status") != "valid":
        raise HTTPException(400, detail=validation.get("error") or "Course not ready for planning")

    try:
        return planner_engine.generate_study_plan(
            course_id, days_available, hours_per_day, exam_date, user_id, mode
        )
    except Exception as e:
        print(f"Study plan generation failed: {e}")
        raise HTTPException(500, detail=f"Study plan generation failed: {e}")


@router.post("/api/replan")
async def replan_endpoint(
    course_id: str = Form(...),
    days_available: int | None = Form(None),
    hours_per_day: float | None = Form(None),
    exam_date: str | None = Form(None),
    user_id: str = Depends(current_user_id),
):
    """Re-generate the plan from the student's current state (weak areas first,
    prerequisites respected, due reviews folded in)."""
    if not course_id:
        raise HTTPException(400, detail="Course ID is required")
    try:
        return planner_engine.generate_study_plan(
            course_id, days_available, hours_per_day, exam_date, user_id, mode="weak_first"
        )
    except Exception as e:
        print(f"Replan failed: {e}")
        raise HTTPException(500, detail=f"Replan failed: {e}")


@router.get("/api/study-plan/{course_id}")
async def get_study_plan_endpoint(course_id: str):
    """Return the latest persisted study plan for a course, or null."""
    try:
        return planner_engine.get_latest_plan(course_id)
    except Exception as e:
        print(f"Study plan fetch failed: {e}")
        raise HTTPException(500, detail=f"Study plan fetch failed: {e}")
