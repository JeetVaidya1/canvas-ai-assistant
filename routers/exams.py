from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import Response, StreamingResponse
from deps import *  # noqa: F401,F403  shared state, engines, helpers, stdlib re-exports

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/upload-past-paper", dependencies=[Depends(ai_rate_limit)])
async def upload_past_paper(
    course_id: str = Form(...),
    file: UploadFile = File(...),
    user_id: str = Depends(current_user_id)
):
    """Upload and analyze a past paper"""
    try:
        print(f"📄 Uploading past paper for course: {course_id}")
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(400, detail="Only PDF files are supported for past papers")
        
        # Check if course exists
        course_check = supabase.table("courses").select("*").eq("course_id", course_id).execute()
        if not course_check.data:
            raise HTTPException(400, detail="Course not found")
        
        # Read file content
        content = await file.read()
        
        # Analyze the past paper
        analysis = exam_generator.analyze_past_paper(content, file.filename)
        
        if analysis.get("error"):
            return {"status": "error", "message": analysis["error"]}
        
        # Save analysis to database
        exam_generator.save_past_paper_analysis(course_id, analysis)
        
        # Store the file for future reference
        storage_path = f"{course_id}/past_papers/{file.filename}"
        try:
            public_url = upload_file("course-files", content, storage_path)
            
            # Save file metadata
            supabase.table("past_papers").insert({
                "course_id": course_id,
                "filename": file.filename,
                "storage_path": storage_path,
                "analysis_data": analysis,
                "uploaded_by": user_id,
                "uploaded_at": datetime.utcnow().isoformat()
            }).execute()
            
        except Exception as e:
            print(f"Storage warning: {e}")
        
        return {
            "status": "success",
            "message": f"Successfully analyzed {file.filename}",
            "analysis": analysis,
            "questions_found": len(analysis.get("extracted_questions", [])),
            "exam_structure": analysis.get("analysis", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Past paper upload failed: {e}")
        logger.exception("Upload failed")
        raise HTTPException(500, detail="Upload failed")


@router.post("/api/generate-practice-exam", dependencies=[Depends(ai_rate_limit)])
async def generate_practice_exam(
    course_id: str = Form(...),
    exam_type: str = Form("practice"),
    question_count: int = Form(10),
    time_limit: int = Form(120),
    difficulty: str = Form("mixed"),
    question_types: str = Form('["multiple_choice", "calculation", "short_answer"]'),
    topic_focus: str = Form(""),
    user_id: str = Depends(current_user_id)
):
    """Generate a practice exam"""
    try:
        # Enhanced debugging
        print(f"🎯 EXAM GENERATION REQUEST:")
        print(f"   Course ID: {course_id}")
        print(f"   Question Count: {question_count}")
        print(f"   Difficulty: {difficulty}")
        print(f"   User ID: {user_id}")
        
        # Parse question types with better error handling
        try:
            question_types_list = json.loads(question_types)
            print(f"   Question Types: {question_types_list}")
        except json.JSONDecodeError as e:
            print(f"   ⚠️ Invalid question_types JSON: {question_types}")
            question_types_list = ["multiple_choice", "calculation", "short_answer"]
        
        # Check if course has files with detailed logging
        try:
            result = supabase.table("files").select("*").eq("course_id", course_id).execute()
            file_count = len(result.data) if result.data else 0
            print(f"   📁 Files found for course: {file_count}")
            
            if file_count == 0:
                print(f"   ❌ No files found for course {course_id}")
                raise HTTPException(400, detail=f"No files found for course '{course_id}'. Upload course materials first.")
                
            # List the files for debugging
            for file in result.data[:3]:  # Show first 3 files
                print(f"      - {file.get('filename', 'unknown')}")
                
        except HTTPException:
            raise
        except Exception as e:
            print(f"   ⚠️ Could not check course files: {e}")
        
        # Build exam specifications
        exam_specs = {
            "name": f"{exam_type.title()} Exam - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "exam_type": exam_type,
            "question_count": question_count,
            "time_limit": time_limit,
            "difficulty": difficulty,
            "question_types": question_types_list,
            "topic_focus": topic_focus,
            "course_id": course_id,
            "created_by": user_id
        }
        
        print(f"   📋 Exam specs prepared: {exam_specs['name']}")
        
        # Generate the exam with timing
        start_time = datetime.now()
        result = exam_generator.generate_practice_exam(course_id, exam_specs)
        generation_time = (datetime.now() - start_time).total_seconds()
        
        print(f"   ⏱️ Generation took: {generation_time:.2f} seconds")
        
        if result.get("status") == "error":
            print(f"   ❌ Generation failed: {result.get('message')}")
            raise HTTPException(500, detail=result.get("message", "Exam generation failed"))
        
        exam_data = result["exam"]
        
        print(f"   ✅ Generated exam with {len(exam_data['questions'])} questions")
        
        # Enhanced response
        response_data = {
            "status": "success",
            "exam": exam_data,
            "message": f"Generated {exam_data['question_count']} question exam",
            "debug": {
                "generation_time_seconds": generation_time,
                "course_files_count": file_count,
                "request_timestamp": datetime.now().isoformat()
            }
        }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"   ❌ Exam generation failed: {e}")
        import traceback
        traceback.print_exc()
        logger.exception("Generation failed")
        raise HTTPException(500, detail="Generation failed")


@router.post("/api/create-exam-session")
async def create_exam_session(
    exam_data: str = Form(...),
    user_id: str = Depends(current_user_id),
    course_id: str = Form(...)
):
    """Create a new exam session"""
    try:
        print(f"📝 Creating exam session for user: {user_id}")
        
        # Parse exam data
        try:
            exam_obj = json.loads(exam_data)
        except json.JSONDecodeError:
            raise HTTPException(400, detail="Invalid exam data format")
        
        # Create session
        result = exam_session_manager.create_exam_session(user_id, course_id, exam_obj)
        
        if result.get("status") == "error":
            raise HTTPException(500, detail=result.get("message", "Session creation failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Session creation failed: {e}")
        logger.exception("Session creation failed")
        raise HTTPException(500, detail="Session creation failed")


@router.post("/api/start-exam-session/{session_id}")
async def start_exam_session(session_id: str):
    """Start an exam session (begin timing)"""
    try:
        result = exam_session_manager.start_exam_session(session_id)
        
        if result.get("status") == "error":
            raise HTTPException(400, detail=result.get("message"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Session start failed: {e}")
        logger.exception("Failed to start session")
        raise HTTPException(500, detail="Failed to start session")


@router.post("/api/pause-exam-session/{session_id}")
async def pause_exam_session(session_id: str):
    """Pause/unpause an exam session"""
    try:
        result = exam_session_manager.pause_exam_session(session_id)
        
        if result.get("status") == "error":
            raise HTTPException(400, detail=result.get("message"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Session pause failed: {e}")
        logger.exception("Failed to pause session")
        raise HTTPException(500, detail="Failed to pause session")


@router.post("/api/save-exam-answer")
async def save_exam_answer(
    session_id: str = Form(...),
    question_id: str = Form(...),
    answer: str = Form(...)
):
    """Save an answer to an exam question"""
    try:
        result = exam_session_manager.save_answer(session_id, question_id, answer)
        
        if result.get("status") == "error":
            raise HTTPException(400, detail=result.get("message"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Answer save failed: {e}")
        logger.exception("Failed to save answer")
        raise HTTPException(500, detail="Failed to save answer")


@router.post("/api/navigate-exam-question")
async def navigate_exam_question(
    session_id: str = Form(...),
    question_index: int = Form(...)
):
    """Navigate to a specific question in the exam"""
    try:
        result = exam_session_manager.navigate_to_question(session_id, question_index)
        
        if result.get("status") == "error":
            raise HTTPException(400, detail=result.get("message"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Navigation failed: {e}")
        logger.exception("Navigation failed")
        raise HTTPException(500, detail="Navigation failed")


@router.post("/api/submit-exam/{session_id}")
async def submit_exam(session_id: str):
    """Submit and score the exam"""
    try:
        result = exam_session_manager.submit_exam(session_id)
        
        if result.get("status") == "error":
            raise HTTPException(400, detail=result.get("message"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Exam submission failed: {e}")
        logger.exception("Submission failed")
        raise HTTPException(500, detail="Submission failed")


@router.get("/api/exam-session/{session_id}")
async def get_exam_session(session_id: str):
    """Get current exam session state"""
    try:
        result = exam_session_manager.get_session(session_id)
        
        if result.get("status") == "error":
            raise HTTPException(404, detail=result.get("message"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Get session failed: {e}")
        logger.exception("Failed to get session")
        raise HTTPException(500, detail="Failed to get session")


@router.get("/api/exam-history/{user_id}")
async def get_exam_history(user_id: str, course_id: Optional[str] = None):
    """Get user's exam history"""
    try:
        history = exam_session_manager.get_user_exam_history(user_id, course_id)
        
        return {
            "status": "success",
            "exams": history,
            "total_exams": len(history)
        }
        
    except Exception as e:
        print(f"❌ Get exam history failed: {e}")
        logger.exception("Failed to get exam history")
        raise HTTPException(500, detail="Failed to get exam history")


@router.delete("/api/exam-session/{session_id}")
async def delete_exam_session(session_id: str):
    """Delete an exam session"""
    try:
        success = exam_session_manager.delete_session(session_id)
        
        if success:
            return {"status": "success", "message": "Session deleted"}
        else:
            raise HTTPException(404, detail="Session not found or deletion failed")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Delete session failed: {e}")
        logger.exception("Deletion failed")
        raise HTTPException(500, detail="Deletion failed")


@router.get("/api/past-papers/{course_id}")
async def get_past_papers(course_id: str):
    """Get list of past papers for a course"""
    try:
        result = supabase.table("past_papers").select("*").eq("course_id", course_id).order("uploaded_at", desc=True).execute()
        
        papers = []
        for paper in result.data or []:
            papers.append({
                "id": paper["id"],
                "filename": paper["filename"],
                "uploaded_at": paper["uploaded_at"],
                "analysis_summary": {
                    "total_questions": len(paper.get("analysis_data", {}).get("extracted_questions", [])),
                    "exam_type": paper.get("analysis_data", {}).get("analysis", {}).get("exam_type", "unknown"),
                    "difficulty": paper.get("analysis_data", {}).get("analysis", {}).get("difficulty_level", "unknown")
                }
            })
        
        return {
            "status": "success",
            "past_papers": papers,
            "total": len(papers)
        }
        
    except Exception as e:
        print(f"❌ Get past papers failed: {e}")
        logger.exception("Failed to get past papers")
        raise HTTPException(500, detail="Failed to get past papers")


@router.post("/api/solve-exam-question", dependencies=[Depends(ai_rate_limit)])
async def solve_exam_question(
    course_id: str = Form(...),
    question_text: str = Form(...),
    want_hint: bool = Form(False),
    pdf_file: UploadFile = File(None),
    past_paper_id: str = Form(None),
    pages: str = Form("[]")
):
    """Solve one question with GPT-5 Vision + RAG"""
    try:
        try:
            page_list = json.loads(pages) if pages else []
            if not isinstance(page_list, list):
                page_list = []
        except Exception:
            page_list = []

        file_bytes = None

        # Option A: direct upload
        if pdf_file is not None:
            if not pdf_file.filename.lower().endswith(".pdf"):
                raise HTTPException(400, "pdf_file must be a PDF")
            file_bytes = await pdf_file.read()

        # Option B: fetch from supabase storage using past_paper_id
        elif past_paper_id:
            try:
                record = supabase.table("past_papers").select("*").eq("id", past_paper_id).single().execute()
                if not record.data:
                    raise HTTPException(404, "Past paper not found")
                storage_path = record.data["storage_path"]
                file_bytes = download_file("course-files", storage_path)
            except Exception as e:
                print(f"Storage download failed: {e}")

        # Solve the question
        result = exam_generator.solve_question_with_vision(
            course_id=course_id,
            question_text=question_text,
            file_bytes=file_bytes,
            pages=page_list,
            want_hint=want_hint
        )

        if result.get("status") == "error":
            raise HTTPException(500, result.get("message", "Solve failed"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ solve_exam_question failed: {e}")
        logger.exception("Solve failed")
        raise HTTPException(500, detail="Solve failed")


@router.get("/api/exam-analytics/{course_id}/{user_id}")
async def get_exam_analytics(course_id: str, user_id: str):
    """Get detailed exam analytics for a user in a course"""
    try:
        # Get exam history
        exam_history = exam_session_manager.get_user_exam_history(user_id, course_id)
        
        # Calculate analytics
        analytics = calculate_exam_analytics(exam_history)
        
        return {
            "status": "success",
            "analytics": analytics,
            "exam_count": len(exam_history)
        }
        
    except Exception as e:
        print(f"❌ Exam analytics failed: {e}")
        logger.exception("Analytics failed")
        raise HTTPException(500, detail="Analytics failed")


@router.get("/api/admin/auto-submit-expired-exams")
async def auto_submit_expired_exams():
    """Admin endpoint to auto-submit expired exams"""
    try:
        expired_count = exam_session_manager.auto_submit_expired_exams()
        return {
            "status": "success",
            "message": f"Auto-submitted {expired_count} expired exams"
        }
    except Exception as e:
        print(f"❌ Auto-submit failed: {e}")
        logger.exception("Auto-submit failed")
        raise HTTPException(500, detail="Auto-submit failed")


@router.get("/api/exam-status")
async def exam_status():
    """Check if exam system is properly initialized"""
    try:
        # Test exam generator
        exam_gen_ok = exam_generator is not None
        session_mgr_ok = exam_session_manager is not None
        
        # Test database connection
        db_ok = False
        try:
            test_result = supabase.table("courses").select("course_id").limit(1).execute()
            db_ok = True
        except Exception as e:
            print(f"DB test failed: {e}")
        
        return {
            "status": "success",
            "exam_generator_ready": exam_gen_ok,
            "session_manager_ready": session_mgr_ok,
            "database_connected": db_ok,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

