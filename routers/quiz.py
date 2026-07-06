from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, Depends
from datetime import datetime

from auth import current_user_id
from deps import supabase, validate_course_for_practice
from quiz_assistant_engine import assist_with_quiz_question
from rate_limit import ai_rate_limit

import logging

logger = logging.getLogger(__name__)

import quiz_engine

router = APIRouter()


@router.post("/quiz-assist", dependencies=[Depends(ai_rate_limit)])
async def quiz_assist_endpoint(
    question: str = Form(...),
    course_id: str = Form(...),
    session_id: str | None = Form(None),
    user_id: str = Depends(current_user_id)
):
    """Quiz assistance endpoint - handles any quiz question"""
    
    print(f"🎯 Quiz assistance request for course: {course_id}")
    
    # Validate inputs
    if not question.strip():
        raise HTTPException(400, detail="Question cannot be empty")
        
    if not course_id:
        raise HTTPException(400, detail="Course ID is required")
    
    # Check if course exists and has files
    try:
        course_check = supabase.table("courses").select("*").eq("course_id", course_id).execute()
        if not course_check.data:
            raise HTTPException(400, detail="Course not found")
            
        files_check = supabase.table("files").select("filename").eq("course_id", course_id).execute()
        if not files_check.data:
            return {
                "status": "error",
                "answer": "No course materials found",
                "explanation": "Please upload course materials first before using quiz assistance. I need your course content to provide accurate answers.",
                "confidence": 0.0,
                "question_type": "unknown",
                "study_tips": ["Upload your course materials (PDFs, slides, notes) to get started"],
                "similar_concepts": [],
                "estimated_time": "",
                "relevant_sources": []
            }
            
    except Exception as e:
        print(f"Course validation error: {e}")
        raise HTTPException(500, detail="Course validation failed")
    
    # Create session if needed
    actual_session_id = session_id
    if not actual_session_id:
        try:
            resp = supabase.table("chat_sessions").insert({
                "user_id": user_id,
                "course_id": course_id,
                "title": f"Quiz: {question[:50]}...",
                "created_at": datetime.utcnow().isoformat()
            }).execute()
            actual_session_id = resp.data[0]["id"]
        except Exception as e:
            print(f"Session creation failed: {e}")
            actual_session_id = None
    
    # Log the question
    if actual_session_id:
        try:
            supabase.table("messages").insert({
                "session_id": actual_session_id,
                "role": "user",
                "content": f"[QUIZ] {question}",
                "timestamp": datetime.utcnow().isoformat()
            }).execute()
        except Exception as e:
            print(f"Question logging failed: {e}")
    
    # Get quiz assistance response
    try:
        response = assist_with_quiz_question(question, course_id, actual_session_id)
        
        # Log the response
        if actual_session_id and response.get('status') == 'success':
            try:
                assistant_message = f"QUIZ ANSWER: {response['answer']}\n\nEXPLANATION: {response['explanation']}\n\nCONFIDENCE: {response['confidence']:.0%}"
                
                supabase.table("messages").insert({
                    "session_id": actual_session_id,
                    "role": "assistant", 
                    "content": assistant_message,
                    "timestamp": datetime.utcnow().isoformat()
                }).execute()
            except Exception as e:
                print(f"Response logging failed: {e}")
        
        # Add session info
        if actual_session_id:
            response["session_id"] = actual_session_id
            
        return response
        
    except Exception as e:
        print(f"Quiz assistance failed: {e}")
        return {
            "status": "error",
            "answer": "I encountered an error processing your quiz question.",
            "explanation": "Please try rephrasing your question or check if it's formatted correctly.",
            "confidence": 0.0,
            "question_type": "unknown",
            "study_tips": ["Try rephrasing the question", "Include all answer choices for multiple choice"],
            "similar_concepts": [],
            "estimated_time": "",
            "relevant_sources": []
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quiz runner: generate (two-phase) -> answer one-at-a-time -> submit
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/quiz/generate", dependencies=[Depends(ai_rate_limit)])
async def generate_quiz_endpoint(
    background_tasks: BackgroundTasks,
    course_id: str = Form(...),
    topic: str | None = Form(None),
    num_questions: int = Form(10),
    difficulty: str = Form("medium"),
    user_id: str = Depends(current_user_id),
):
    """Two-phase quiz generation (fast start).

    Returns immediately with the session + the first few questions (never the
    answer key). If more questions were requested, generation_status is
    'generating' and the remainder is written by a background task — clients
    poll GET /quiz/{quiz_id}/questions until it reads 'ready' (or 'partial').
    """
    if not course_id:
        raise HTTPException(400, detail="Course ID is required")

    validation = await validate_course_for_practice(course_id)
    if validation.get("status") != "valid":
        raise HTTPException(400, detail=validation.get("error") or "Course not ready for quizzes")

    num_questions = max(1, min(int(num_questions), 20))
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "medium"
    clean_topic = (topic or "").strip() or None

    try:
        result = quiz_engine.generate_quiz(
            course_id, clean_topic, num_questions, difficulty, user_id=user_id,
        )
    except Exception:
        logger.exception("Quiz generation failed")
        raise HTTPException(500, detail="Quiz generation failed")

    background_spec = result.pop("_background", None)
    if background_spec:
        background_tasks.add_task(quiz_engine.generate_remaining_questions, **background_spec)
    return result


# NOTE: static path — declared before the /quiz/{quiz_id}/... routes so it can
# never be captured by a dynamic segment.
@router.get("/quiz/in-progress")
async def in_progress_quizzes_endpoint(
    course_id: str = "",
    user_id: str = Depends(current_user_id),
):
    """Resume everywhere: the user's unfinished quizzes in a course.

    Returns up to 3 sessions, newest first, each with answered/available
    counts so the client can render "Resume (4/10)" style entries.
    """
    clean_course_id = (course_id or "").strip()
    if not clean_course_id:
        raise HTTPException(400, detail="Course ID is required")

    try:
        sessions = quiz_engine.get_in_progress_quizzes(clean_course_id, user_id)
    except Exception:
        logger.exception("In-progress quiz lookup failed")
        raise HTTPException(500, detail="Couldn't fetch in-progress quizzes")
    return {"sessions": sessions}


@router.get("/quiz/{quiz_id}/responses")
async def quiz_responses_endpoint(
    quiz_id: str,
    user_id: str = Depends(current_user_id),
):
    """Saved answers (latest per question) for resuming a quiz mid-session."""
    try:
        return quiz_engine.get_quiz_responses(quiz_id, user_id)
    except KeyError:
        raise HTTPException(404, detail="Quiz not found")
    except Exception:
        logger.exception("Quiz responses fetch failed")
        raise HTTPException(500, detail="Couldn't fetch quiz responses")


@router.get("/quiz/{quiz_id}/questions")
async def quiz_questions_endpoint(
    quiz_id: str,
    user_id: str = Depends(current_user_id),
):
    """Sanitized questions (no answer key) + generation progress.

    Poll while generation_status == 'generating'; stop on 'ready' or 'partial'.
    """
    try:
        return quiz_engine.get_quiz_questions(quiz_id, user_id)
    except KeyError:
        raise HTTPException(404, detail="Quiz not found")
    except Exception:
        logger.exception("Quiz questions fetch failed")
        raise HTTPException(500, detail="Couldn't fetch quiz questions")


@router.post("/quiz/{quiz_id}/answer")
async def answer_quiz_endpoint(
    quiz_id: str,
    background_tasks: BackgroundTasks,
    question_id: str = Form(...),
    selected: str = Form(...),
    time_taken: float = Form(0.0),
    confidence: str | None = Form(None),
    user_id: str = Depends(current_user_id),
):
    """Grade one answer; returns correctness + explanation + source.

    ``confidence`` is the optional pre-reveal tap ('sure'|'thinkso'|'guessing'),
    stored for the end-of-quiz calibration read-out. Wrong answers get their
    grounded mistake explanation + review seeding in the background so grading
    latency is identical for right and wrong answers.
    """
    clean_confidence = (confidence or "").strip().lower() or None
    if clean_confidence is not None and clean_confidence not in quiz_engine.CONFIDENCE_LEVELS:
        raise HTTPException(
            400, detail="confidence must be one of: sure, thinkso, guessing")

    try:
        result = quiz_engine.grade_answer(
            quiz_id, question_id, selected, time_taken, user_id,
            confidence=clean_confidence,
        )
    except KeyError as e:
        raise HTTPException(404, detail=str(e))
    except Exception:
        logger.exception("Grading failed")
        raise HTTPException(500, detail="Grading failed")

    if not result.get("is_correct"):
        background_tasks.add_task(
            quiz_engine.followup_wrong_answer, quiz_id, question_id, selected, user_id,
        )
    return result


@router.post("/quiz/{quiz_id}/submit")
async def submit_quiz_endpoint(
    quiz_id: str,
    user_id: str = Depends(current_user_id),
):
    """Finalize a quiz; returns score, per-topic breakdown, weak areas, and the
    confidence-calibration read-out."""
    try:
        return quiz_engine.submit_quiz(quiz_id, user_id)
    except Exception:
        logger.exception("Submit failed")
        raise HTTPException(500, detail="Submit failed")

