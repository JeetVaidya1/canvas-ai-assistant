"""Practice routes: problem generation + Course Brain topic endpoints.

Topics are served from the ``course_topics`` table (course_brain.py) —
synthesized from real chunk content on first request and rebuilt on demand.
The legacy response shapes (``{"topics": [names], ...}``) are preserved for
the existing frontend; the new ``/api/courses/{course_id}/topics`` routes
return full topic objects (slug/name/description/doc_coverage/prereq_slugs/
position) for the Course Brief and mastery grid.
"""
import logging

from fastapi import APIRouter, Depends, Form, HTTPException

import course_brain
from auth import current_user_id, get_current_user, require_course_access
from deps import (
    get_intelligent_fallback_topics,
    practice_generator,
    validate_course_for_practice,
)
from rate_limit import ai_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/generate-practice", dependencies=[Depends(ai_rate_limit)])
async def generate_practice_problems(
    course_id: str = Form(...),
    topic: str = Form(...),
    difficulty: str = Form("adaptive"),
    count: int = Form(5),
    user_id: str = Depends(current_user_id)
):
    """Generate practice problems. difficulty='adaptive' routes off the user's
    mastery of the topic (easy/medium/hard)."""
    try:
        problems = practice_generator.generate_practice_problems(
            course_id, topic, difficulty, count, user_id
        )
        return {"problems": problems}
    except Exception as e:
        print(f"Practice generation error: {e}")
        # Return fallback problems
        return {"problems": [{
            "question": f"Sample practice question about {topic}",
            "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
            "correct_answer": "A",
            "explanation": "This is a sample explanation",
            "estimated_time": "3-5 minutes",
            "difficulty": difficulty,
            "topic": topic
        }]}


@router.get("/practice-topics/{course_id}")
async def get_practice_topics(course_id: str, user=Depends(get_current_user)):
    """Course Brain topic names for practice (auto-synthesized on first call)."""
    require_course_access(course_id, user)
    try:
        validation_result = await validate_course_for_practice(course_id)
        if validation_result["error"]:
            return validation_result

        try:
            topics = course_brain.topic_names(course_id, auto_generate=True)
            if not topics:
                logger.warning("Course Brain returned no topics for %s; using fallback", course_id)
                topics = await get_intelligent_fallback_topics(course_id)
                return {
                    "topics": topics,
                    "course_files_count": validation_result["files_count"],
                    "extraction_method": "intelligent_fallback",
                    "fallback": True,
                    "status": "partial_success",
                }
            return {
                "topics": topics,
                "course_files_count": validation_result["files_count"],
                "extraction_method": "course_brain",
                "status": "success",
            }
        except Exception as e:
            logger.exception("Course Brain topic fetch failed for %s", course_id)
            fallback_topics = await get_intelligent_fallback_topics(course_id)
            return {
                "topics": fallback_topics,
                "error": f"Extraction failed, using fallback: {str(e)}",
                "fallback": True,
                "status": "partial_success",
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Complete failure in get_practice_topics for %s", course_id)
        return {
            "topics": ["System Error"],
            "error": f"System error: {str(e)}",
            "status": "error",
        }


@router.post("/regenerate-practice-topics", dependencies=[Depends(ai_rate_limit)])
async def regenerate_practice_topics(course_id: str = Form(...),
                                     user=Depends(get_current_user)):
    """Force a Course Brain rebuild (fresh synthesis from course content)."""
    require_course_access(course_id, user)
    try:
        validation = await validate_course_for_practice(course_id)
        if validation["error"]:
            return {
                "status": "error",
                "message": validation["error"],
                "topics": validation["topics"],
            }

        topics = [t.name for t in course_brain.synthesize_topics(course_id)]
        if not topics:
            fallback = await get_intelligent_fallback_topics(course_id)
            return {
                "status": "partial_success",
                "topics": fallback,
                "message": f"Used intelligent fallback - generated {len(fallback)} topics for course {course_id}",
                "fallback": True,
            }
        return {
            "status": "success",
            "topics": topics,
            "message": f"Successfully regenerated {len(topics)} topics for course {course_id}",
            "extraction_method": "course_brain",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Topic regeneration failed for %s", course_id)
        fallback_topics = await get_intelligent_fallback_topics(course_id)
        return {
            "status": "error",
            "message": str(e),
            "topics": fallback_topics,
            "fallback": True,
        }


# ---- Course Brain full-object routes (V3) ----------------------------------
# The frontend Course Brief + topic mastery grid build on these.

@router.get("/api/courses/{course_id}/topics")
async def get_course_topics(course_id: str, user=Depends(get_current_user)):
    """Full Course Brain topic objects, teaching order (synthesized on first call)."""
    require_course_access(course_id, user)
    try:
        topics = course_brain.get_topics(course_id, auto_generate=True)
    except Exception:
        logger.exception("Course topics fetch failed for %s", course_id)
        raise HTTPException(500, detail="Failed to load course topics")
    return {
        "course_id": course_id,
        "topics": [t.to_dict() for t in topics],
        "count": len(topics),
    }


@router.post("/api/courses/{course_id}/topics/rebuild",
             dependencies=[Depends(ai_rate_limit)])
async def rebuild_course_topics(course_id: str, user=Depends(get_current_user)):
    """Rebuild the Course Brain for a course (idempotent delete+insert)."""
    require_course_access(course_id, user)
    try:
        topics = course_brain.synthesize_topics(course_id)
    except Exception:
        logger.exception("Course topics rebuild failed for %s", course_id)
        raise HTTPException(500, detail="Topic rebuild failed")
    if not topics:
        raise HTTPException(
            409,
            detail="No ingested content to build topics from. Upload course materials first.",
        )
    return {
        "course_id": course_id,
        "status": "success",
        "topics": [t.to_dict() for t in topics],
        "count": len(topics),
    }
