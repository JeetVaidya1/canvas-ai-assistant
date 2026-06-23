from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import Response, StreamingResponse
from deps import *  # noqa: F401,F403  shared state, engines, helpers, stdlib re-exports

router = APIRouter()


@router.get("/analytics/{course_id}/{user_id}")
async def get_learning_analytics(course_id: str, user_id: str):
    """Get learning analytics for a student in a specific course"""
    try:
        print(f"Getting analytics for user {user_id} in course {course_id}")
        
        analytics = analytics_engine.get_learning_analytics(user_id, course_id)
        
        # Add course-specific context
        analytics["course_id"] = course_id
        
        return {"analytics": analytics}
    except Exception as e:
        print(f"Analytics error for course {course_id}, user {user_id}: {e}")
        return {"analytics": {
            "topics_progress": [],
            "study_streak": 0,
            "weak_areas": [],
            "study_recommendations": [f"Start studying {course_id} to see analytics!"],
            "total_questions": 0,
            "avg_confidence": 0.0,
            "study_time_trend": [],
            "course_id": course_id
        }}


@router.get("/analytics-topics/{course_id}")  
async def get_analytics_topics(course_id: str):
    """Get topics that have been studied in this course"""
    try:
        # Get topics from learning progress table for this specific course  
        from supabase import create_client
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY") 
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Get unique topics from learning progress for this course
        progress_query = supabase.table("learning_progress") \
            .select("topic") \
            .eq("course_id", course_id) \
            .execute()
        
        if progress_query.data:
            studied_topics = list(set([item["topic"] for item in progress_query.data]))
            print(f"Found studied topics for course {course_id}: {studied_topics}")
            return {"topics": studied_topics}
        else:
            # If no progress yet, try to get topics from course content
            topics = practice_generator.extract_topics_from_course(course_id)
            return {"topics": topics}
            
    except Exception as e:
        print(f"Failed to get analytics topics for course {course_id}: {e}")
        return {"topics": []}


@router.post("/track-interaction")
async def track_user_interaction(
    user_id: str = Form(...),
    course_id: str = Form(...),
    question: str = Form(...),
    answer: str = Form(...),
    confidence: float = Form(...),
    response_time: int = Form(...)
):
    """Track a user interaction for analytics"""
    try:
        success = analytics_engine.track_interaction(
            user_id, course_id, question, answer, confidence, response_time
        )
        return {"success": success}
    except Exception as e:
        print(f"Tracking error: {e}")
        return {"success": False}


@router.post("/track-practice-session")
async def track_practice_session(
    user_id: str = Form(...),
    course_id: str = Form(...),
    topic: str = Form(...),
    problems_attempted: int = Form(...),
    problems_correct: int = Form(...),
    duration_minutes: int = Form(...),
    difficulty_level: str = Form(...)
):
    """Track a completed practice session"""
    try:
        success = analytics_engine.track_practice_session(
            user_id, course_id, topic, problems_attempted,
            problems_correct, duration_minutes, difficulty_level
        )
        return {"status": "success" if success else "error"}

    except Exception as e:
        print(f"Failed to track practice session: {e}")
        return {"status": "error", "message": str(e)}

