# check_endpoints.py - Check what exam endpoints you actually have
import re

def check_main_py_endpoints():
    """Check what exam endpoints are actually in your main.py"""
    print("🔍 Checking your main.py for exam endpoints...")
    
    try:
        with open("main.py", "r") as f:
            content = f.read()
        
        # Look for all @app.post and @app.get decorators
        endpoint_pattern = r'@app\.(post|get|put|delete)\(["\']([^"\']+)["\'].*?\)\s*\n\s*async def\s+(\w+)'
        matches = re.findall(endpoint_pattern, content, re.MULTILINE)
        
        print("\n📍 Found endpoints:")
        exam_endpoints = []
        for method, path, func_name in matches:
            print(f"  {method.upper()} {path} -> {func_name}")
            if any(exam_word in path.lower() for exam_word in ['exam', 'past-paper']):
                exam_endpoints.append((method, path, func_name))
        
        print(f"\n🎯 Exam-related endpoints found: {len(exam_endpoints)}")
        for method, path, func_name in exam_endpoints:
            print(f"  ✅ {method.upper()} {path}")
        
        # Check specifically for the endpoints your frontend is calling
        required_endpoints = [
            '/api/upload-past-paper',
            '/api/generate-practice-exam',
            '/api/create-exam-session',
            '/api/past-papers',
            '/api/exam-history'
        ]
        
        print(f"\n📋 Required endpoint check:")
        for required in required_endpoints:
            found = any(required in path for _, path, _ in matches)
            status = "✅" if found else "❌"
            print(f"  {status} {required}")
        
        # Check for exam imports
        print(f"\n🔧 Import check:")
        if "from exam_generator import ExamGenerator" in content:
            print("  ✅ ExamGenerator imported")
        else:
            print("  ❌ ExamGenerator not imported")
            
        if "from exam_session_manager import ExamSessionManager" in content:
            print("  ✅ ExamSessionManager imported")
        else:
            print("  ❌ ExamSessionManager not imported")
        
        # Check if they're initialized
        if "exam_generator = ExamGenerator()" in content:
            print("  ✅ exam_generator initialized")
        else:
            print("  ❌ exam_generator not initialized")
            
        if "exam_session_manager = ExamSessionManager()" in content:
            print("  ✅ exam_session_manager initialized")
        else:
            print("  ❌ exam_session_manager not initialized")
            
        return exam_endpoints
        
    except FileNotFoundError:
        print("❌ main.py not found!")
        return []
    except Exception as e:
        print(f"❌ Error reading main.py: {e}")
        return []

def suggest_missing_endpoints():
    """Show what endpoints need to be added"""
    print("\n💡 Missing endpoints need to be added to main.py:")
    
    endpoints_to_add = '''
# Add these to your main.py:

@app.post("/api/upload-past-paper")
async def upload_past_paper(
    course_id: str = Form(...),
    file: UploadFile = File(...),
    user_id: str = Form("anonymous")
):
    """Upload and analyze a past paper"""
    try:
        content = await file.read()
        analysis = exam_generator.analyze_past_paper(content, file.filename)
        
        if analysis.get("error"):
            return {"status": "error", "message": analysis["error"]}
        
        return {
            "status": "success",
            "message": f"Successfully analyzed {file.filename}",
            "analysis": analysis,
            "questions_found": len(analysis.get("extracted_questions", [])),
            "exam_structure": analysis.get("analysis", {})
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Upload failed: {str(e)}")

@app.post("/api/generate-practice-exam")
async def generate_practice_exam(
    course_id: str = Form(...),
    exam_type: str = Form("practice"),
    question_count: int = Form(10),
    time_limit: int = Form(120),
    difficulty: str = Form("mixed"),
    question_types: str = Form('["multiple_choice", "calculation", "short_answer"]'),
    topic_focus: str = Form(""),
    user_id: str = Form("anonymous")
):
    """Generate a practice exam"""
    try:
        exam_specs = {
            "name": f"{exam_type.title()} Exam - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "exam_type": exam_type,
            "question_count": question_count,
            "time_limit": time_limit,
            "difficulty": difficulty,
            "question_types": json.loads(question_types),
            "topic_focus": topic_focus,
            "course_id": course_id,
            "created_by": user_id
        }
        
        result = exam_generator.generate_practice_exam(course_id, exam_specs)
        
        if result.get("status") == "error":
            raise HTTPException(500, detail=result.get("message", "Exam generation failed"))
        
        return {
            "status": "success",
            "exam": result["exam"],
            "message": f"Generated {result['exam']['question_count']} question exam"
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Generation failed: {str(e)}")

@app.get("/api/past-papers/{course_id}")
async def get_past_papers(course_id: str):
    """Get list of past papers for a course"""
    return {
        "status": "success",
        "past_papers": [],
        "total": 0
    }

@app.get("/api/exam-history/{user_id}")
async def get_exam_history(user_id: str, course_id: Optional[str] = None):
    """Get user's exam history"""
    return {
        "status": "success",
        "exams": [],
        "total_exams": 0
    }
'''
    
    print(endpoints_to_add)

if __name__ == "__main__":
    print("🚀 Endpoint Checker")
    print("=" * 40)
    
    exam_endpoints = check_main_py_endpoints()
    
    if len(exam_endpoints) == 0:
        print("\n❌ No exam endpoints found!")
        suggest_missing_endpoints()
    else:
        print(f"\n✅ Found {len(exam_endpoints)} exam endpoints")
        print("\nNow run the server and test:")
        print("  uvicorn main:app --reload")
        print("  python debug_live_api.py")