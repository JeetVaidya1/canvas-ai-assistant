from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import Response, StreamingResponse
from deps import *  # noqa: F401,F403  shared state, engines, helpers, stdlib re-exports

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/generate-notes", dependencies=[Depends(ai_rate_limit)])
async def generate_notes_endpoint(
    course_id: str = Form(...),
    file_names: str = Form(...),  # JSON string of file names list
    topic: str = Form(""),
    style: str = Form("detailed"),
    user_id: str = Depends(current_user_id)
):
    """Generate comprehensive notes from lecture files"""
    
    print(f"📝 Notes generation request for course: {course_id}")
    
    # Validate inputs
    if not course_id:
        raise HTTPException(400, detail="Course ID is required")
    
    try:
        import json
        file_list = json.loads(file_names)
        if not file_list:
            raise HTTPException(400, detail="At least one file must be selected")
    except json.JSONDecodeError:
        raise HTTPException(400, detail="Invalid file names format")
    
    # Validate style
    if style not in ["detailed", "summary", "outline"]:
        style = "detailed"
    
    # Check if course exists
    try:
        course_check = supabase.table("courses").select("*").eq("course_id", course_id).execute()
        if not course_check.data:
            raise HTTPException(400, detail="Course not found")
    except Exception as e:
        print(f"Course validation error: {e}")
        raise HTTPException(500, detail="Course validation failed")
    
    # Generate notes
    try:
        print(f"🎯 Generating {style} notes for files: {file_list}")
        if topic:
            print(f"📖 Topic focus: {topic}")
            
        result = generate_notes_from_files(course_id, file_list, topic, style)
        
        if result.get("status") == "error":
            return {
                "status": "error",
                "message": result.get("message", "Notes generation failed"),
                "notes": result.get("notes", ""),
                "suggested_title": "Error - Generation Failed",
                "word_count": 0,
                "reading_time": "0 min",
                "topics": [],
                "source_files": file_list
            }
        
        print(f"✅ Generated {result.get('word_count', 0)} word notes")
        
        return {
            "status": "success",
            "notes": result.get("notes", ""),
            "suggested_title": result.get("suggested_title", "Generated Notes"),
            "word_count": result.get("word_count", 0),
            "reading_time": result.get("reading_time", "0 min"),
            "topics": result.get("topics", []),
            # Structured flashcards the engine already generated (schema tool-call).
            # Previously dropped here, forcing the UI to regex-scrape markdown -> "0 cards".
            "flashcards": result.get("flashcards", []),
            "source_files": result.get("source_files", file_list)
        }
        
    except Exception as e:
        print(f"❌ Notes generation failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "status": "error",
            "message": "Notes generation failed",
            "notes": "An error occurred while generating your notes. Please try again with different files or check that your selected files contain readable content.",
            "suggested_title": "Error - Generation Failed",
            "word_count": 0,
            "reading_time": "0 min",
            "topics": [],
            "source_files": file_list
        }


@router.post("/save-notes")
async def save_notes_endpoint(
    course_id: str = Form(...),
    title: str = Form(...),
    content: str = Form(...),
    source_files: str = Form(...),  # JSON string of file names
    topic: str = Form(""),
    note_id: str = Form(None),
    user_id: str = Depends(current_user_id)
):
    """Save notes to database"""
    
    print(f"💾 Saving notes: {title}")
    
    # Validate inputs
    if not course_id or not title.strip() or not content.strip():
        raise HTTPException(400, detail="Course ID, title, and content are required")
    
    try:
        import json
        file_list = json.loads(source_files)
    except json.JSONDecodeError:
        raise HTTPException(400, detail="Invalid source files format")
    
    # Save notes
    try:
        result = save_notes_to_db(course_id, title.strip(), content, file_list, topic, note_id)
        
        if result.get("status") == "success":
            print(f"✅ Notes saved successfully: {title}")
            return {
                "status": "success",
                "message": "Notes saved successfully",
                "note": result.get("note")
            }
        else:
            print(f"❌ Notes saving failed: {result.get('message')}")
            raise HTTPException(500, detail=result.get("message", "Failed to save notes"))
            
    except Exception as e:
        print(f"❌ Notes saving error: {e}")
        logger.exception("Notes saving failed")
        raise HTTPException(500, detail="Notes saving failed")


@router.put("/notes/{note_id}")
async def update_note_endpoint(
    note_id: str,
    course_id: str = Form(...),
    title: str = Form(...),
    content: str = Form(...),
    source_files: str = Form("[]"),  # JSON string of file names
    topic: str = Form(""),
):
    """Update a saved note in place (edit-and-resave)."""
    if not note_id:
        raise HTTPException(400, detail="Note ID is required")
    if not course_id or not title.strip() or not content.strip():
        raise HTTPException(400, detail="Course ID, title, and content are required")

    try:
        import json
        file_list = json.loads(source_files)
    except json.JSONDecodeError:
        raise HTTPException(400, detail="Invalid source files format")

    try:
        result = save_notes_to_db(course_id, title.strip(), content, file_list, topic, note_id)
        if result.get("status") == "success" and result.get("note"):
            return {"status": "success", "message": "Notes updated", "note": result.get("note")}
        raise HTTPException(404, detail=result.get("message", "Note not found or update failed"))
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Notes update error: {e}")
        logger.exception("Notes update failed")
        raise HTTPException(500, detail="Notes update failed")


@router.get("/notes/{course_id}")
async def get_notes_endpoint(course_id: str):
    """Get all saved notes for a course"""
    
    print(f"📖 Fetching notes for course: {course_id}")
    
    if not course_id:
        raise HTTPException(400, detail="Course ID is required")
    
    try:
        notes = get_notes_from_db(course_id)
        print(f"✅ Found {len(notes)} notes")
        
        return {
            "status": "success",
            "notes": notes
        }
        
    except Exception as e:
        print(f"❌ Notes retrieval error: {e}")
        logger.exception("Failed to retrieve notes")
        raise HTTPException(500, detail="Failed to retrieve notes")


@router.delete("/notes/{note_id}")
async def delete_note_endpoint(note_id: str):
    """Delete a saved note"""
    
    print(f"🗑️ Deleting note: {note_id}")
    
    if not note_id:
        raise HTTPException(400, detail="Note ID is required")
    
    try:
        success = delete_note_from_db(note_id)
        
        if success:
            print(f"✅ Note deleted successfully")
            return {
                "status": "success",
                "message": "Note deleted successfully"
            }
        else:
            print(f"❌ Note deletion failed")
            raise HTTPException(500, detail="Failed to delete note")
            
    except Exception as e:
        print(f"❌ Note deletion error: {e}")
        logger.exception("Note deletion failed")
        raise HTTPException(500, detail="Note deletion failed")

