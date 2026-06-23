from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import Response, StreamingResponse
from deps import *  # noqa: F401,F403  shared state, engines, helpers, stdlib re-exports

router = APIRouter()


@router.post("/quiz-assist")
async def quiz_assist_endpoint(
    question: str = Form(...),
    course_id: str = Form(...),
    session_id: str | None = Form(None),
    user_id: str = Form("anonymous")
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

