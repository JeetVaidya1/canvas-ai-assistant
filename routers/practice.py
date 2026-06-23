from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import Response, StreamingResponse
from deps import *  # noqa: F401,F403  shared state, engines, helpers, stdlib re-exports

router = APIRouter()


@router.post("/generate-practice")
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
async def get_practice_topics(course_id: str):
    """Get available topics for practice based on ACTUAL course content - works for any subject"""
    try:
        print(f"🔍 Getting practice topics for course: {course_id}")
        
        # Validate course exists and has content
        validation_result = await validate_course_for_practice(course_id)
        if validation_result["error"]:
            return validation_result
        
        # Extract topics using the generic practice generator
        try:
            print(f"📖 Starting topic extraction for course: {course_id}")
            topics = practice_generator.extract_topics_from_course(course_id)
            
            if not topics or len(topics) == 0:
                print("⚠️ No topics extracted, using intelligent fallback")
                topics = await get_intelligent_fallback_topics(course_id)
            
            print(f"✅ Successfully extracted {len(topics)} topics: {topics}")
            
            return {
                "topics": topics,
                "course_files_count": validation_result["files_count"],
                "extraction_method": "generic_multi_strategy",
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Topic extraction failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Intelligent fallback based on course content
            fallback_topics = await get_intelligent_fallback_topics(course_id)
            
            return {
                "topics": fallback_topics,
                "error": f"Extraction failed, using fallback: {str(e)}",
                "fallback": True,
                "status": "partial_success"
            }
        
    except Exception as e:
        print(f"❌ Complete failure in get_practice_topics: {e}")
        return {
            "topics": ["System Error"],
            "error": f"System error: {str(e)}",
            "status": "error"
        }


@router.post("/regenerate-practice-topics")
async def regenerate_practice_topics(course_id: str = Form(...)):
    """Force regeneration of practice topics for any course"""
    try:
        print(f"🔄 Force regenerating topics for course: {course_id}")
        
        # Validate course first
        validation = await validate_course_for_practice(course_id)
        if validation["error"]:
            return {
                "status": "error",
                "message": validation["error"],
                "topics": validation["topics"]
            }
        
        # Force fresh extraction
        topics = practice_generator.extract_topics_from_course(course_id)
        
        if not topics:
            topics = await get_intelligent_fallback_topics(course_id)
            return {
                "status": "partial_success",
                "topics": topics,
                "message": f"Used intelligent fallback - generated {len(topics)} topics for course {course_id}",
                "fallback": True
            }
        
        return {
            "status": "success",
            "topics": topics,
            "message": f"Successfully regenerated {len(topics)} topics for course {course_id}",
            "extraction_method": "full_analysis"
        }
        
    except Exception as e:
        print(f"❌ Topic regeneration failed: {e}")
        fallback_topics = await get_intelligent_fallback_topics(course_id)
        return {
            "status": "error",
            "message": str(e),
            "topics": fallback_topics,
            "fallback": True
        }

