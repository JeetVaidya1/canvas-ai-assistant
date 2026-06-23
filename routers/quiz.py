from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import Response, StreamingResponse
from deps import *  # noqa: F401,F403  shared state, engines, helpers, stdlib re-exports

import quiz_engine

router = APIRouter()


@router.post("/quiz-assist")
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
# Quiz runner: generate -> answer one-at-a-time -> submit (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/quiz/generate")
async def generate_quiz_endpoint(
    course_id: str = Form(...),
    topic: str | None = Form(None),
    num_questions: int = Form(10),
    difficulty: str = Form("medium"),
):
    """Generate a grounded MCQ quiz and persist it. Returns quiz_id + questions
    (without the answer key)."""
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
        return quiz_engine.generate_quiz(course_id, clean_topic, num_questions, difficulty)
    except Exception as e:
        print(f"Quiz generation failed: {e}")
        raise HTTPException(500, detail=f"Quiz generation failed: {e}")


@router.post("/quiz/{quiz_id}/answer")
async def answer_quiz_endpoint(
    quiz_id: str,
    question_id: str = Form(...),
    selected: str = Form(...),
    time_taken: float = Form(0.0),
    user_id: str = Depends(current_user_id),
):
    """Grade one answer; returns correctness + explanation + source."""
    try:
        return quiz_engine.grade_answer(quiz_id, question_id, selected, time_taken, user_id)
    except KeyError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        print(f"Quiz answer grading failed: {e}")
        raise HTTPException(500, detail=f"Grading failed: {e}")


@router.post("/quiz/{quiz_id}/submit")
async def submit_quiz_endpoint(
    quiz_id: str,
    user_id: str = Depends(current_user_id),
):
    """Finalize a quiz; returns score, per-topic breakdown, and weak areas."""
    try:
        return quiz_engine.submit_quiz(quiz_id, user_id)
    except Exception as e:
        print(f"Quiz submit failed: {e}")
        raise HTTPException(500, detail=f"Submit failed: {e}")

