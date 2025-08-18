from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from query_engine import ask_question
import os
import json
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from storage import upload_file           
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query
from supabase import create_client
from datetime import datetime
import os
from dotenv import load_dotenv
from datetime import datetime
from query_engine import ask_question
from storage import upload_file
from ingest import process_file
from quiz_assistant_engine import assist_with_quiz_question
from notes_engine import generate_notes_from_files, save_notes_to_db, get_notes_from_db, delete_note_from_db
from learning_analytics import LearningAnalyticsEngine
from practice_generator import PracticeGenerator
from typing import Dict, List, Any, Optional
import asyncio
from fastapi import Form, UploadFile, File
from fastapi import HTTPException
from exam_generator import ExamGenerator
from exam_session_manager import ExamSessionManager
from typing import Optional

analytics_engine = LearningAnalyticsEngine()
practice_generator = PracticeGenerator()

# NEW: Try to import enhanced modules
try:
    from enhanced_ingest import process_file_enhanced, delete_file_from_course as enhanced_delete_file
    from enhanced_query_engine import enhanced_ask_question
    ENHANCED_MODE = True
    print("✅ Enhanced multimodal system loaded!")
except ImportError as e:
    print(f"⚠️ Enhanced system not available: {e}")
    ENHANCED_MODE = False

try:
    from conversational_rag_engine import conversational_ask_question
    CONVERSATIONAL_MODE = True
    print("✅ Conversational RAG system loaded!")
except ImportError as e:
    print(f"⚠️ Conversational RAG not available: {e}")
    CONVERSATIONAL_MODE = False

# 1. Load the secret note (.env)
load_dotenv()

# 2. Read the URL & KEY from that note
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# 3. Make a Supabase "client" to talk to the database
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

@app.post("/upload/{course_id}")
async def upload(course_id: str, file: UploadFile = File(...)):
    # 1) Read the file bytes
    content = await file.read()

    # 2) Upload to Supabase Storage (overwrite if exists)
    storage_path = f"{course_id}/{file.filename}"
    public_url = upload_file("course-files", content, storage_path)

    # 3) Record metadata in Supabase
    try:
        result = supabase.table("files").insert({
            "course_id":   course_id,
            "filename":    file.filename,
            "storage_path": storage_path,
            "file_type":   file.filename.rsplit(".", 1)[-1],
            "uploaded_at": datetime.utcnow().isoformat()
        }).execute()
        metadata = result.data
    except Exception as e:
        raise HTTPException(500, detail=f"DB insert failed: {e}")

    # 4) **Enhanced processing with fallback**
    try:
        if ENHANCED_MODE:
            print("🚀 Using enhanced multimodal processing...")
            chunks = process_file_enhanced(file.filename, content, course_id)
        else:
            chunks = process_file(file.filename, content, course_id)
    except Exception as e:
        print(f"⚠️ Processing failed, using fallback: {e}")
        chunks = process_file(file.filename, content, course_id)

    # 5) Return everything
    return {
        "url":    public_url,
        "meta":   metadata,
        "chunks": chunks
    }

# Allow frontend access (CORS setup)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Lock this down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COURSE_DB_PATH = "courses.json"

if not os.path.exists(COURSE_DB_PATH):
    with open(COURSE_DB_PATH, "w") as f:
        json.dump({}, f)

def load_courses():
    with open(COURSE_DB_PATH) as f:
        return json.load(f)

def save_courses(courses):
    with open(COURSE_DB_PATH, "w") as f:
        json.dump(courses, f, indent=2)

@app.get("/")
def health_check():
    status_message = "✅ API is running"
    if ENHANCED_MODE:
        status_message += " (Enhanced Mode Active 🚀)"
    return {"status": status_message}

@app.post("/create-course")
def create_course(course_id: str = Form(...), title: str = Form(...)):
    # Check if course already exists in Supabase
    try:
        existing = supabase.table("courses").select("*").eq("course_id", course_id).execute()
        if existing.data:
            raise HTTPException(400, detail="Course already exists")
    except Exception as e:
        if "Course already exists" in str(e):
            raise e
        # If it's just a DB connection issue, continue with local storage
        pass

    # Create local directories
    os.makedirs(f"data/{course_id}", exist_ok=True)
    os.makedirs(f"vectorstores/{course_id}", exist_ok=True)

    # Save to local JSON file (for backward compatibility)
    courses = load_courses()
    courses[course_id] = {"title": title, "files": []}
    save_courses(courses)

    # **NEW: Also save to Supabase**
    try:
        supabase.table("courses").insert({
            "course_id": course_id,
            "title": title
        }).execute()
    except Exception as e:
        # If Supabase fails, at least we have local storage
        print(f"Warning: Failed to save to Supabase: {e}")

    return {"status": "ok", "message": f"Created course {title}"}

@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    course_id: str = Form(...)
):
    print(f"🚀 Enhanced upload request for course: {course_id}")
    print(f"📁 Number of files: {len(files)}")
    
    # Check if course exists
    try:
        print(f"🔍 Checking if course {course_id} exists...")
        course_check = supabase.table("courses").select("*").eq("course_id", course_id).execute()
        print(f"📊 Course check result: {course_check.data}")
        if not course_check.data:
            print("❌ Course not found!")
            raise HTTPException(400, detail="Course not found")
        print("✅ Course exists!")
    except Exception as e:
        print(f"❌ Course check failed: {e}")
        raise HTTPException(400, detail=f"Invalid course_id: {e}")

    uploaded_files = []
    chunks_preview = []
    errors = []
    
    for i, file in enumerate(files):
        print(f"\n📄 Processing file {i+1}/{len(files)}: {file.filename}")
        
        try:
            # 1) Check if file already exists
            print("🔍 Checking for existing file...")
            existing_file = supabase.table("files").select("*").eq("course_id", course_id).eq("filename", file.filename).execute()
            print(f"📊 Existing file check: {len(existing_file.data)} matches found")
            
            if existing_file.data:
                print(f"🔄 File {file.filename} already exists, replacing...")
                
                # Delete existing file completely
                print("🗑️ Deleting from embeddings...")
                supabase.table("embeddings").delete().eq("course_id", course_id).eq("doc_name", file.filename).execute()
                
                print("🗑️ Deleting from files table...")
                supabase.table("files").delete().eq("course_id", course_id).eq("filename", file.filename).execute()
                
                # Delete from storage
                storage_path = f"{course_id}/{file.filename}"
                try:
                    print(f"🗑️ Deleting from storage: {storage_path}")
                    supabase.storage.from_("course-files").remove([storage_path])
                except Exception as storage_error:
                    print(f"⚠️ Storage deletion warning: {storage_error}")

            # 2) Read the file bytes  
            print("📖 Reading file content...")
            content = await file.read()
            print(f"📏 File size: {len(content)} bytes")

            # 3) Upload to Supabase Storage
            storage_path = f"{course_id}/{file.filename}"
            print(f"☁️ Uploading to storage: {storage_path}")
            try:
                public_url = upload_file("course-files", content, storage_path)
                print(f"✅ Storage upload successful: {public_url}")
            except Exception as e:
                print(f"❌ Storage upload failed: {e}")
                errors.append(f"Storage upload failed for {file.filename}: {e}")
                continue

            # 4) Record metadata in Supabase files table
            print("💾 Saving file metadata to database...")
            try:
                file_record = {
                    "course_id": course_id,
                    "filename": file.filename,
                    "storage_path": storage_path,
                    "file_type": file.filename.rsplit(".", 1)[-1] if "." in file.filename else "unknown",
                    "uploaded_at": datetime.utcnow().isoformat()
                }
                print(f"📝 File record: {file_record}")
                
                result = supabase.table("files").insert(file_record).execute()
                print(f"✅ Database insert successful: {result.data}")
                file_metadata = result.data[0] if result.data else {}
            except Exception as e:
                print(f"❌ Database insert failed: {e}")
                errors.append(f"Database insert failed for {file.filename}: {e}")
                continue

            # 5) **ENHANCED: Process file for vector embeddings with multimodal support**
            print("🧠 Processing file for AI embeddings...")
            try:
                if ENHANCED_MODE:
                    print("🚀 Using enhanced multimodal processing...")
                    chunks = process_file_enhanced(file.filename, content, course_id)
                    print(f"✅ Enhanced processing successful: {len(chunks)} chunks")
                else:
                    print("📝 Using basic processing...")
                    chunks = process_file(file.filename, content, course_id)
                    print(f"✅ Basic processing successful: {len(chunks)} chunks")
                
                chunks_preview.extend(chunks[:2])  # Preview first 2 chunks per file
            except Exception as e:
                print(f"❌ Processing failed: {e}")
                # Fallback to basic processing if enhanced fails
                try:
                    if ENHANCED_MODE:
                        print("🔄 Falling back to basic processing...")
                    chunks = process_file(file.filename, content, course_id)
                    chunks_preview.extend(chunks[:2])
                    print(f"✅ Fallback processing successful")
                except Exception as e2:
                    print(f"❌ All processing failed: {e2}")
                    chunks_preview.append({"chunk": f"Processing failed for {file.filename}: {e2}"})

            # 6) Also save to local storage for backward compatibility
            print("💿 Saving to local storage...")
            try:
                courses = load_courses()
                if course_id not in courses:
                    courses[course_id] = {"title": "Unknown", "files": []}
                
                if file.filename not in courses[course_id]["files"]:
                    courses[course_id]["files"].append(file.filename)
                
                # Save actual file locally
                file_path = f"data/{course_id}/{file.filename}"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(content)
                
                save_courses(courses)
                print("✅ Local storage successful")
            except Exception as e:
                print(f"⚠️ Local storage warning: {e}")

            uploaded_files.append({
                "filename": file.filename,
                "url": public_url,
                "metadata": file_metadata,
                "status": "success"
            })
            
            print(f"🎉 File {file.filename} processed successfully!")

        except Exception as e:
            print(f"💥 Failed to process {file.filename}: {str(e)}")
            import traceback
            traceback.print_exc()
            errors.append(f"Failed to process {file.filename}: {str(e)}")
            continue

    # Return results with success/error info
    response = {
        "status": "completed", 
        "message": f"Processed {len(uploaded_files)} files",
        "files": uploaded_files,
        "chunks": chunks_preview
    }
    
    if errors:
        response["errors"] = errors
        response["status"] = "partial" if uploaded_files else "failed"
    
    print(f"📤 Enhanced processing complete!")
    return response

@app.post("/ask")
async def ask_endpoint(
    question: str = Form(...),
    course_id: str = Form(...),
    session_id: str | None = Form(None),
    user_id: str = Form("anonymous")
):
    # 1) Create a new chat_session if none was provided
    if not session_id:
        try:
            resp = supabase.table("chat_sessions").insert({
                "user_id": user_id,
                "course_id": course_id,
                "title": question[:50],
                "created_at": datetime.utcnow().isoformat()
            }).execute()
            session_id = resp.data[0]["id"]
        except Exception as e:
            raise HTTPException(500, detail=f"Couldn't create session: {e}")

    # 2) Record the user's question
    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "role": "user",
            "content": question,
            "timestamp": datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        raise HTTPException(500, detail=f"Couldn't save question: {e}")

    # 3) **NEW: Generate conversational answer with context awareness**
    try:
        if CONVERSATIONAL_MODE:
            print("🧠 Using conversational RAG with context awareness...")
            answer = conversational_ask_question(question, course_id, session_id)
            print("✅ Conversational answer generated!")
        elif ENHANCED_MODE:
            print("🤖 Using enhanced question answering...")
            answer = enhanced_ask_question(question, course_id)
        else:
            print("📝 Using basic question answering...")
            answer = ask_question(question, course_id)
    except Exception as e:
        print(f"❌ All QA methods failed, using fallback: {e}")
        answer = "I'm having trouble processing your question. Could you please rephrase it or try asking in a different way?"

    # 4) Record the assistant's answer
    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        raise HTTPException(500, detail=f"Couldn't save answer: {e}")

    return {
        "session_id": session_id,
        "question": question,
        "answer": answer
    }

@app.get("/list-courses")
def list_courses():
    try:
        # Fetch from Supabase first
        resp = supabase.table("courses").select("*").order("created_at", desc=True).execute()
        courses_data = resp.data
        
        # Convert to the format expected by frontend
        courses = [{"course_id": c["course_id"], "title": c["title"]} for c in courses_data]
        return {"courses": courses}
    
    except Exception as e:
        print(f"Supabase error: {e}")
        # Fallback to local JSON file
        courses_file = "courses.json"
        if not os.path.exists(courses_file):
            return {"courses": []}
        
        with open(courses_file, "r") as f:
            courses = json.load(f)
        
        return {"courses": [{"course_id": cid, "title": data["title"]} for cid, data in courses.items()]}

@app.get("/list-files")
def list_files(course_id: str):
    try:
        # Fetch from Supabase files table
        resp = supabase.table("files").select("filename").eq("course_id", course_id).execute()
        files = [row["filename"] for row in resp.data]
        return {"files": files}
    
    except Exception as e:
        print(f"Supabase error: {e}")
        # Fallback to local JSON
        folder_path = os.path.join("vectorstores", course_id, "files.json")
        if not os.path.exists(folder_path):
            return {"files": []}
        
        with open(folder_path, "r") as f:
            file_list = json.load(f)
        
        return {"files": file_list}

from ingest import delete_file_from_course, delete_course

@app.post("/delete-file")
async def delete_file(course_id: str = Form(...), filename: str = Form(...)):
    try:
        # Delete from Supabase files table
        supabase.table("files").delete().eq("course_id", course_id).eq("filename", filename).execute()
        
        # Delete from Supabase storage
        storage_path = f"{course_id}/{filename}"
        try:
            supabase.storage.from_("course-files").remove([storage_path])
        except Exception as e:
            print(f"Storage deletion failed (file may not exist): {e}")
        
        # **ENHANCED: Use enhanced delete if available**
        try:
            if ENHANCED_MODE:
                deleted = enhanced_delete_file(course_id, filename)
            else:
                deleted = delete_file_from_course(course_id, filename)
        except Exception as e:
            print(f"Vector store deletion failed: {e}")
            deleted = False
        
        # Clean up local files if they exist
        courses = load_courses()
        if course_id in courses and filename in courses[course_id]["files"]:
            courses[course_id]["files"].remove(filename)
            save_courses(courses)
        
        return {"status": "ok", "message": f"Deleted {filename} from {course_id}"}
        
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to delete file: {e}")

@app.post("/delete-course")
async def delete_entire_course(course_id: str = Form(...)):
    try:
        # Delete all files for this course from Supabase files table
        files_result = supabase.table("files").select("filename, storage_path").eq("course_id", course_id).execute()
        
        # Delete from storage
        if files_result.data:
            storage_paths = [row["storage_path"] for row in files_result.data]
            if storage_paths:
                try:
                    supabase.storage.from_("course-files").remove(storage_paths)
                except Exception as e:
                    print(f"Storage deletion failed: {e}")
        
        # Delete files metadata from database
        supabase.table("files").delete().eq("course_id", course_id).execute()
        
        # Delete course from courses table
        supabase.table("courses").delete().eq("course_id", course_id).execute()
        
        # Delete from vector store
        success = delete_course(course_id)
        
        # Clean up local files
        courses = load_courses()
        if course_id in courses:
            del courses[course_id]
            save_courses(courses)
            
            # Clean up local directories
            data_path = f"data/{course_id}"
            if os.path.exists(data_path):
                shutil.rmtree(data_path)
                
            vectorstore_path = f"vectorstores/{course_id}"
            if os.path.exists(vectorstore_path):
                shutil.rmtree(vectorstore_path)
        
        return {"status": "ok", "message": f"Deleted course {course_id}"}
        
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to delete course: {e}")

@app.get("/sessions")
def list_sessions(user_id: str = Query(..., description="User ID to filter by")):
    """
    List all sessions for a user, newest first.
    """
    try:
        resp = supabase.table("chat_sessions") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .execute()
        sessions = resp.data
    except Exception as e:
        raise HTTPException(500, detail=f"Couldn't fetch sessions: {e}")
    return {"sessions": sessions}

@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: str):
    """
    Fetch all messages in a session, oldest first.
    """
    try:
        resp = supabase.table("messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("timestamp", desc=False) \
            .execute()
        messages = resp.data
    except Exception as e:
        raise HTTPException(500, detail=f"Couldn't fetch messages: {e}")
    return {"messages": messages}

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """
    Delete a chat session and all its messages.
    """
    try:
        # First delete all messages in the session
        supabase.table("messages").delete().eq("session_id", session_id).execute()
        
        # Then delete the session itself
        supabase.table("chat_sessions").delete().eq("id", session_id).execute()
        
        return {"status": "ok", "message": "Session deleted successfully"}
    except Exception as e:
        raise HTTPException(500, detail=f"Couldn't delete session: {e}")

@app.get("/system-status")
def get_system_status():
    """Get current system capabilities and status"""
    return {
        "enhanced_mode": ENHANCED_MODE,
        "conversational_mode": CONVERSATIONAL_MODE,
        "capabilities": {
            "multimodal_processing": ENHANCED_MODE,
            "image_extraction": ENHANCED_MODE,
            "enhanced_formatting": ENHANCED_MODE,
            "question_classification": ENHANCED_MODE,
            "quiz_assistance": True,
            "intelligent_parsing": True,
            "confidence_scoring": True,
            "study_recommendations": True,
            "notes_generation": True,
            "comprehensive_notes": True,
            "notes_management": True,
            "learning_analytics": True,  # NEW!
            "practice_mode": True,       # NEW!
            "progress_tracking": True,   # NEW!
            "adaptive_difficulty": True, # NEW!
            "spaced_repetition": True    # NEW!
        },
        "version": "3.0.0" if ENHANCED_MODE else "2.0.0"
    }

@app.post("/quiz-assist")
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
@app.post("/generate-notes")
async def generate_notes_endpoint(
    course_id: str = Form(...),
    file_names: str = Form(...),  # JSON string of file names list
    topic: str = Form(""),
    style: str = Form("detailed"),
    user_id: str = Form("anonymous")
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

@app.post("/save-notes")
async def save_notes_endpoint(
    course_id: str = Form(...),
    title: str = Form(...),
    content: str = Form(...),
    source_files: str = Form(...),  # JSON string of file names
    topic: str = Form(""),
    note_id: str = Form(None),
    user_id: str = Form("anonymous")
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
        raise HTTPException(500, detail=f"Notes saving failed: {str(e)}")

@app.get("/notes/{course_id}")
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
        raise HTTPException(500, detail=f"Failed to retrieve notes: {str(e)}")

@app.delete("/notes/{note_id}")
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
        raise HTTPException(500, detail=f"Note deletion failed: {str(e)}")

@app.get("/analytics/{course_id}/{user_id}")
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

@app.get("/analytics-topics/{course_id}")  
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

@app.post("/generate-practice")
async def generate_practice_problems(
    course_id: str = Form(...),
    topic: str = Form(...),
    difficulty: str = Form("medium"),
    count: int = Form(5)
):
    """Generate practice problems"""
    try:
        problems = practice_generator.generate_practice_problems(course_id, topic, difficulty, count)
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

@app.post("/track-interaction")
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

@app.post("/track-practice-session")
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
        # For now, just update learning progress
        confidence = problems_correct / problems_attempted if problems_attempted > 0 else 0.5
        analytics_engine.update_learning_progress(user_id, course_id, topic, confidence)
        
        return {"status": "success", "session_id": "temp_session_id"}
        
    except Exception as e:
        print(f"Failed to track practice session: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/practice-topics/{course_id}")
async def get_practice_topics(course_id: str):
    """Get available topics for practice based on ACTUAL course content"""
    try:
        print(f"Getting topics for course: {course_id}")
        
        # Use the practice generator to extract real topics
        topics = practice_generator.extract_topics_from_course(course_id)
        
        print(f"Extracted topics: {topics}")
        
        return {"topics": topics}
        
    except Exception as e:
        print(f"Failed to get practice topics for course {course_id}: {e}")
        # Return fallback topics with clear indication
        return {
            "topics": [
                "Course Content Analysis", 
                "General Review",
                "Key Concepts"
            ],
            "error": "Could not analyze course content for topics"
        }

@app.get("/health/rag")
def rag_health():
    try:
        row = supabase.table("embeddings").select("course_id, embedding").limit(1).execute()
        if not row.data:
            return {"ok": False, "reason": "no embeddings yet"}
        course_id = row.data[0]["course_id"]
        e_txt = str(row.data[0]["embedding"])  # vector -> text (PostgREST serializes it)
        res = supabase.rpc("match_embeddings", {
            "query_embedding": e_txt,
            "course_id_param": course_id,
            "match_count": 1
        }).execute()
        return {"ok": True, "rows": len(res.data or []), "course_id": course_id}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/practice-topics/{course_id}")
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

async def validate_course_for_practice(course_id: str) -> dict:
    """Validate that a course exists and has content for practice generation"""
    try:
        # Check if course exists
        course_check = supabase.table("courses").select("*").eq("course_id", course_id).execute()
        if not course_check.data:
            print(f"❌ Course {course_id} not found")
            return {
                "error": "Course not found",
                "topics": ["Course Not Found"],
                "status": "error"
            }
        
        # Check if course has uploaded files
        files_check = supabase.table("files").select("filename").eq("course_id", course_id).execute()
        if not files_check.data:
            print(f"❌ No files found for course {course_id}")
            return {
                "error": "No files uploaded. Please upload course materials first.",
                "topics": ["No Files Uploaded"],
                "status": "error"
            }
        
        # Check if files have been processed (have embeddings)
        embeddings_check = supabase.table("embeddings").select("id").eq("course_id", course_id).limit(1).execute()
        if not embeddings_check.data:
            print(f"⚠️ Course {course_id} files not yet processed for AI analysis")
            return {
                "error": "Course files are still being processed. Please try again in a moment.",
                "topics": ["Processing Files"],
                "status": "processing"
            }
        
        print(f"✅ Course {course_id} validation passed - {len(files_check.data)} files found")
        return {
            "error": None,
            "files_count": len(files_check.data),
            "status": "valid"
        }
        
    except Exception as e:
        print(f"❌ Course validation error: {e}")
        return {
            "error": f"Validation error: {str(e)}",
            "topics": ["Validation Error"],
            "status": "error"
        }

async def get_intelligent_fallback_topics(course_id: str) -> list:
    """Generate intelligent fallback topics based on available course info"""
    try:
        # Try to get course title for subject hints
        course_info = supabase.table("courses").select("title").eq("course_id", course_id).execute()
        course_title = ""
        if course_info.data:
            course_title = course_info.data[0].get("title", "").lower()
        
        # Try to get some filenames for topic hints
        files_info = supabase.table("files").select("filename").eq("course_id", course_id).execute()
        filenames = []
        if files_info.data:
            filenames = [f["filename"].lower() for f in files_info.data]
        
        # Generate subject-appropriate fallback topics
        fallback_topics = generate_subject_fallback_topics(course_title, filenames)
        
        print(f"📋 Generated intelligent fallback topics: {fallback_topics}")
        return fallback_topics
        
    except Exception as e:
        print(f"❌ Fallback topic generation failed: {e}")
        return [
            "Course Fundamentals",
            "Key Concepts",
            "Core Material",
            "Main Topics",
            "Essential Knowledge"
        ]

def generate_subject_fallback_topics(course_title: str, filenames: list) -> list:
    """Generate fallback topics based on course title and filenames - subject aware"""
    
    # Combine course title and filenames for analysis
    text_to_analyze = f"{course_title} {' '.join(filenames)}"
    
    # Subject detection patterns (can be expanded)
    subject_patterns = {
        "computer_science": {
            "keywords": ["programming", "algorithm", "data", "structure", "software", "code", "java", "python", "cs", "computer"],
            "topics": ["Programming Fundamentals", "Algorithm Analysis", "Data Structures", "Software Development", "Problem Solving", "Computational Thinking"]
        },
        "mathematics": {
            "keywords": ["calculus", "algebra", "geometry", "statistics", "math", "equation", "theorem", "proof", "derivative", "integral"],
            "topics": ["Mathematical Concepts", "Problem Solving", "Theoretical Foundations", "Applied Mathematics", "Mathematical Analysis", "Computational Methods"]
        },
        "biology": {
            "keywords": ["biology", "cell", "organism", "genetics", "evolution", "ecology", "physiology", "anatomy", "molecular", "bio"],
            "topics": ["Biological Systems", "Cell Biology", "Genetics and Evolution", "Physiology", "Ecological Concepts", "Molecular Biology"]
        },
        "chemistry": {
            "keywords": ["chemistry", "chemical", "reaction", "molecule", "atom", "organic", "inorganic", "lab", "compound", "chem"],
            "topics": ["Chemical Principles", "Molecular Structure", "Chemical Reactions", "Organic Chemistry", "Inorganic Chemistry", "Laboratory Techniques"]
        },
        "physics": {
            "keywords": ["physics", "mechanics", "thermodynamics", "electromagnetic", "quantum", "force", "energy", "motion", "wave"],
            "topics": ["Mechanics", "Thermodynamics", "Electromagnetism", "Wave Physics", "Modern Physics", "Applied Physics"]
        },
        "history": {
            "keywords": ["history", "historical", "century", "war", "civilization", "culture", "society", "period", "ancient", "modern"],
            "topics": ["Historical Events", "Cultural Analysis", "Historical Periods", "Social Movements", "Historical Methods", "Comparative History"]
        },
        "literature": {
            "keywords": ["literature", "poetry", "novel", "author", "writing", "literary", "text", "analysis", "criticism", "english"],
            "topics": ["Literary Analysis", "Literary Themes", "Writing Techniques", "Literary History", "Critical Reading", "Textual Interpretation"]
        },
        "psychology": {
            "keywords": ["psychology", "behavior", "cognitive", "mental", "brain", "learning", "memory", "perception", "psych"],
            "topics": ["Cognitive Psychology", "Behavioral Psychology", "Research Methods", "Psychological Theories", "Human Development", "Mental Processes"]
        },
        "economics": {
            "keywords": ["economics", "market", "economy", "finance", "business", "trade", "money", "supply", "demand", "econ"],
            "topics": ["Economic Principles", "Market Analysis", "Microeconomics", "Macroeconomics", "Economic Policy", "Financial Systems"]
        },
        "engineering": {
            "keywords": ["engineering", "design", "system", "technical", "mechanical", "electrical", "civil", "project", "analysis"],
            "topics": ["Engineering Design", "System Analysis", "Technical Problem Solving", "Engineering Principles", "Project Management", "Applied Engineering"]
        }
    }
    
    # Detect subject based on keywords
    detected_subject = None
    max_matches = 0
    
    for subject, info in subject_patterns.items():
        matches = sum(1 for keyword in info["keywords"] if keyword in text_to_analyze)
        if matches > max_matches:
            max_matches = matches
            detected_subject = subject
    
    # Return subject-specific topics or generic ones
    if detected_subject and max_matches > 0:
        return subject_patterns[detected_subject]["topics"]
    else:
        # Generic academic topics
        return [
            "Course Fundamentals",
            "Key Concepts and Definitions", 
            "Core Principles",
            "Practical Applications",
            "Theoretical Foundations",
            "Problem-Solving Methods"
        ]

@app.get("/debug-course-content/{course_id}")
async def debug_course_content(course_id: str):
    """Debug endpoint to see what content is available for any course"""
    try:
        # Get course info
        course_info = supabase.table("courses").select("*").eq("course_id", course_id).execute()
        course_data = course_info.data[0] if course_info.data else None
        
        # Get files info
        files_result = supabase.table("files").select("*").eq("course_id", course_id).execute()
        files_info = files_result.data or []
        
        # Get embeddings info
        embeddings_result = supabase.table("embeddings").select("doc_name, content, page, slide").eq("course_id", course_id).limit(15).execute()
        embeddings_info = embeddings_result.data or []
        
        # Analyze content diversity
        content_analysis = analyze_course_content_diversity(embeddings_info)
        
        # Sample content from different sources
        sample_content = []
        seen_docs = set()
        for emb in embeddings_info:
            doc_name = emb.get("doc_name", "unknown")
            if doc_name not in seen_docs and len(sample_content) < 5:
                content = emb.get("content", "")
                sample_content.append({
                    "doc": doc_name,
                    "page": emb.get("page"),
                    "slide": emb.get("slide"),
                    "content_preview": content[:300] + "..." if len(content) > 300 else content,
                    "content_length": len(content)
                })
                seen_docs.add(doc_name)
        
        return {
            "course_id": course_id,
            "course_info": {
                "title": course_data.get("title") if course_data else "Unknown",
                "created_at": course_data.get("created_at") if course_data else None
            },
            "files_summary": {
                "count": len(files_info),
                "files": [{"name": f["filename"], "type": f.get("file_type"), "uploaded": f.get("uploaded_at")} for f in files_info]
            },
            "content_analysis": content_analysis,
            "sample_content": sample_content,
            "vector_store_status": {
                "populated": len(embeddings_info) > 0,
                "total_chunks": len(embeddings_info),
                "unique_documents": len(seen_docs)
            }
        }
        
    except Exception as e:
        return {"error": str(e), "course_id": course_id}

def analyze_course_content_diversity(embeddings_info: list) -> dict:
    """Analyze the diversity and richness of course content"""
    if not embeddings_info:
        return {"status": "no_content"}
    
    # Count documents
    doc_counts = {}
    total_content_length = 0
    page_info = {"has_pages": False, "page_range": []}
    slide_info = {"has_slides": False, "slide_range": []}
    
    for emb in embeddings_info:
        doc_name = emb.get("doc_name", "unknown")
        content = emb.get("content", "")
        page = emb.get("page")
        slide = emb.get("slide")
        
        # Count by document
        doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
        total_content_length += len(content)
        
        # Track page info
        if page:
            page_info["has_pages"] = True
            page_info["page_range"].append(page)
        
        # Track slide info
        if slide:
            slide_info["has_slides"] = True
            slide_info["slide_range"].append(slide)
    
    # Calculate content richness
    avg_chunk_length = total_content_length / len(embeddings_info) if embeddings_info else 0
    content_richness = "rich" if avg_chunk_length > 800 else "moderate" if avg_chunk_length > 400 else "sparse"
    
    return {
        "total_chunks": len(embeddings_info),
        "unique_documents": len(doc_counts),
        "document_distribution": doc_counts,
        "average_chunk_length": round(avg_chunk_length),
        "content_richness": content_richness,
        "page_info": {
            "has_pages": page_info["has_pages"],
            "page_range": f"{min(page_info['page_range'])}-{max(page_info['page_range'])}" if page_info["page_range"] else None
        },
        "slide_info": {
            "has_slides": slide_info["has_slides"],
            "slide_range": f"{min(slide_info['slide_range'])}-{max(slide_info['slide_range'])}" if slide_info["slide_range"] else None
        }
    }

@app.post("/regenerate-practice-topics")
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

@app.get("/course-subject-detection/{course_id}")
async def detect_course_subject(course_id: str):
    """Detect what subject area a course covers - useful for UI and analytics"""
    try:
        # Get course info
        course_info = supabase.table("courses").select("title").eq("course_id", course_id).execute()
        course_title = course_info.data[0].get("title", "") if course_info.data else ""
        
        # Get filenames
        files_info = supabase.table("files").select("filename").eq("course_id", course_id).execute()
        filenames = [f["filename"] for f in files_info.data] if files_info.data else []
        
        # Get sample content
        embeddings_sample = supabase.table("embeddings").select("content").eq("course_id", course_id).limit(5).execute()
        sample_content = " ".join([e["content"][:200] for e in embeddings_sample.data]) if embeddings_sample.data else ""
        
        # Analyze subject
        combined_text = f"{course_title} {' '.join(filenames)} {sample_content}".lower()
        
        # Subject detection logic
        subject_scores = {}
        subject_patterns = {
            "Computer Science": ["programming", "algorithm", "data", "structure", "software", "code", "java", "python", "cs"],
            "Mathematics": ["calculus", "algebra", "geometry", "statistics", "math", "equation", "theorem", "proof"],
            "Biology": ["biology", "cell", "organism", "genetics", "evolution", "ecology", "physiology", "bio"],
            "Chemistry": ["chemistry", "chemical", "reaction", "molecule", "atom", "organic", "inorganic", "chem"],
            "Physics": ["physics", "mechanics", "thermodynamics", "electromagnetic", "quantum", "force", "energy"],
            "History": ["history", "historical", "century", "war", "civilization", "culture", "society", "period"],
            "Literature": ["literature", "poetry", "novel", "author", "writing", "literary", "text", "analysis"],
            "Psychology": ["psychology", "behavior", "cognitive", "mental", "brain", "learning", "memory", "psych"],
            "Economics": ["economics", "market", "economy", "finance", "business", "trade", "money", "econ"],
            "Engineering": ["engineering", "design", "system", "technical", "mechanical", "electrical", "civil"]
        }
        
        for subject, keywords in subject_patterns.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                subject_scores[subject] = score
        
        # Find best match
        if subject_scores:
            detected_subject = max(subject_scores, key=subject_scores.get)
            confidence = subject_scores[detected_subject]
        else:
            detected_subject = "General Studies"
            confidence = 0
        
        return {
            "course_id": course_id,
            "detected_subject": detected_subject,
            "confidence_score": confidence,
            "all_scores": subject_scores,
            "course_title": course_title,
            "files_analyzed": len(filenames)
        }
        
    except Exception as e:
        return {
            "course_id": course_id,
            "detected_subject": "Unknown",
            "error": str(e)
        }

@app.get("/debug-topic-extraction/{course_id}")
async def debug_topic_extraction(course_id: str):
    """Debug endpoint to see exactly what's happening in topic extraction"""
    try:
        print(f"🔍 DEBUGGING topic extraction for course: {course_id}")
        
        # Step 1: Check files in database
        files_result = supabase.table("files").select("filename").eq("course_id", course_id).execute()
        filenames = [f["filename"] for f in files_result.data] if files_result.data else []
        print(f"📁 Files in database: {filenames}")
        
        # Step 2: Test filename extraction manually
        filename_topics = []
        for filename in filenames:
            extracted_topic = extract_topic_from_filename_debug(filename)
            filename_topics.append({
                "original_filename": filename,
                "extracted_topic": extracted_topic
            })
        
        print(f"📝 Filename topic extraction results: {filename_topics}")
        
        # Step 3: Check vector store content
        embeddings_result = supabase.table("embeddings").select("doc_name, content").eq("course_id", course_id).limit(10).execute()
        embeddings_info = embeddings_result.data or []
        
        content_sample = []
        for emb in embeddings_info[:3]:
            content_sample.append({
                "doc_name": emb.get("doc_name"),
                "content_preview": emb.get("content", "")[:200] + "..."
            })
        
        # Step 4: Try the practice generator methods individually
        debug_results = {
            "course_id": course_id,
            "files_found": len(filenames),
            "filenames": filenames,
            "filename_extraction_results": filename_topics,
            "embeddings_found": len(embeddings_info),
            "content_sample": content_sample,
        }
        
        # Step 5: Test each extraction method
        try:
            # Test filename extraction
            clean_filename_topics = [item["extracted_topic"] for item in filename_topics if item["extracted_topic"]]
            debug_results["clean_filename_topics"] = clean_filename_topics
            
            # Test practice generator filename method
            pg_filename_topics = practice_generator.extract_topics_from_filenames(course_id)
            debug_results["practice_generator_filename_topics"] = pg_filename_topics
            
            # Test content extraction if we have embeddings
            if embeddings_info:
                from vector_store import VectorStore
                vector_store = VectorStore()
                pg_content_topics = practice_generator.extract_topics_from_content(course_id, vector_store)
                debug_results["practice_generator_content_topics"] = pg_content_topics
            
            # Test full extraction
            full_topics = practice_generator.extract_topics_from_course(course_id)
            debug_results["full_extraction_result"] = full_topics
            
        except Exception as e:
            debug_results["extraction_error"] = str(e)
            import traceback
            debug_results["extraction_traceback"] = traceback.format_exc()
        
        return debug_results
        
    except Exception as e:
        return {
            "error": str(e),
            "course_id": course_id
        }

def extract_topic_from_filename_debug(filename: str) -> str:
    """Debug version of filename topic extraction with detailed logging"""
    print(f"  🔍 Processing filename: {filename}")
    
    # Remove file extension
    clean_name = re.sub(r'\.(pdf|docx|pptx|txt|md)$', '', filename, flags=re.IGNORECASE)
    print(f"    After extension removal: {clean_name}")
    
    # Remove common academic prefixes
    clean_name = re.sub(r'^(lecture|chapter|week|unit|lesson|section|module|assignment|homework|hw|lab|tutorial)\s*\d*\s*[-_:]?\s*', '', clean_name, flags=re.IGNORECASE)
    print(f"    After prefix removal: {clean_name}")
    
    # Remove common suffixes
    clean_name = re.sub(r'\s*(part|section|chapter)\s*\d+$', '', clean_name, flags=re.IGNORECASE)
    clean_name = re.sub(r'\s*(in_class_activity|activity|exercise|solutions?|notes?)$', '', clean_name, flags=re.IGNORECASE)
    print(f"    After suffix removal: {clean_name}")
    
    # Clean up separators and formatting
    clean_name = re.sub(r'[-_]+', ' ', clean_name)
    clean_name = re.sub(r'\s+', ' ', clean_name)
    clean_name = clean_name.strip()
    print(f"    After separator cleanup: {clean_name}")
    
    # Capitalize properly
    if len(clean_name) > 2:
        # Handle special cases like "BSTs" or acronyms
        words = clean_name.split()
        formatted_words = []
        for word in words:
            if len(word) <= 4 and word.isupper():
                formatted_words.append(word)  # Keep acronyms as-is
            else:
                formatted_words.append(word.capitalize())
        
        result = ' '.join(formatted_words)
        print(f"    Final result: {result}")
        return result
    
    print(f"    Final result: (empty - too short)")
    return ""

# Add this debug endpoint to your main.py to see what content is being found

@app.get("/debug-practice-content/{course_id}/{topic}")
async def debug_practice_content(course_id: str, topic: str):
    """Debug what content is found when generating practice questions for a topic"""
    try:
        from vector_store import VectorStore
        from openai import OpenAI
        
        vector_store = VectorStore()
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        print(f"🔍 Debugging content retrieval for topic: '{topic}' in course: {course_id}")
        
        # Test the search queries that practice generator uses
        search_queries = [
            topic,
            f"{topic} examples",
            f"{topic} concepts", 
            f"{topic} definition"
        ]
        
        debug_results = {
            "course_id": course_id,
            "topic": topic,
            "search_results": {},
            "combined_results": [],
            "context_preview": "",
            "total_chunks_found": 0
        }
        
        all_results = []
        
        for query in search_queries:
            try:
                print(f"  🔍 Searching for: '{query}'")
                
                # Create embedding
                emb_resp = openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=[query]
                )
                
                # Search vector store
                results = vector_store.query(course_id, emb_resp.data[0].embedding, top_k=5)
                
                search_info = {
                    "query": query,
                    "results_count": len(results) if results else 0,
                    "results": []
                }
                
                if results:
                    for i, result in enumerate(results):
                        search_info["results"].append({
                            "doc_name": result.get("doc_name", "unknown"),
                            "page": result.get("page"),
                            "similarity": result.get("similarity", 0),
                            "content_preview": result.get("content", "")[:200] + "..." if result.get("content") else "",
                            "content_length": len(result.get("content", ""))
                        })
                        all_results.append(result)
                    
                    print(f"    ✅ Found {len(results)} results")
                else:
                    print(f"    ❌ No results found")
                
                debug_results["search_results"][query] = search_info
                
            except Exception as e:
                print(f"    ❌ Search failed: {e}")
                debug_results["search_results"][query] = {
                    "query": query,
                    "error": str(e),
                    "results_count": 0
                }
        
        # Deduplicate results (same logic as practice generator)
        seen_content = set()
        unique_results = []
        
        for result in all_results:
            content = result.get("content", "").strip()
            content_hash = hash(content[:200])
            
            if content and content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
                if len(unique_results) >= 8:
                    break
        
        debug_results["combined_results"] = [
            {
                "doc_name": r.get("doc_name"),
                "page": r.get("page"),
                "content_preview": r.get("content", "")[:300] + "..." if r.get("content") else "",
                "content_length": len(r.get("content", ""))
            }
            for r in unique_results
        ]
        
        debug_results["total_chunks_found"] = len(unique_results)
        
        # Create context preview (same as practice generator)
        if unique_results:
            context_parts = [f"COURSE MATERIALS ABOUT {topic.upper()}:"]
            
            for i, result in enumerate(unique_results[:6], 1):
                content = result.get("content", "").strip()
                doc = result.get("doc_name", "unknown")
                page = result.get("page")
                
                source_info = f"[Source {i}: {doc}"
                if page:
                    source_info += f", page {page}"
                source_info += "]"
                
                context_parts.append(f"\n{source_info}")
                context_parts.append(content[:500] + "..." if len(content) > 500 else content)
                context_parts.append("---")
            
            debug_results["context_preview"] = "\n".join(context_parts)[:2000] + "..." if len("\n".join(context_parts)) > 2000 else "\n".join(context_parts)
        else:
            debug_results["context_preview"] = "No relevant content found"
        
        # Check if we have any files that should contain this topic
        files_result = supabase.table("files").select("filename").eq("course_id", course_id).execute()
        relevant_files = []
        
        if files_result.data:
            topic_lower = topic.lower()
            for file_info in files_result.data:
                filename = file_info["filename"].lower()
                if any(word in filename for word in topic_lower.split()):
                    relevant_files.append(file_info["filename"])
        
        debug_results["relevant_files"] = relevant_files
        debug_results["should_have_content"] = len(relevant_files) > 0
        
        return debug_results
        
    except Exception as e:
        return {
            "error": str(e),
            "course_id": course_id,
            "topic": topic
        }

# Also add this simpler endpoint to check what's in the vector store
@app.get("/debug-vector-content/{course_id}")
async def debug_vector_content(course_id: str, limit: int = 20):
    """See what content is actually in the vector store for a course"""
    try:
        # Get sample of embeddings
        embeddings_result = supabase.table("embeddings").select("doc_name, content, page, slide").eq("course_id", course_id).limit(limit).execute()
        
        if not embeddings_result.data:
            return {
                "error": "No embeddings found for this course",
                "course_id": course_id
            }
        
        # Group by document
        by_document = {}
        for emb in embeddings_result.data:
            doc_name = emb.get("doc_name", "unknown")
            if doc_name not in by_document:
                by_document[doc_name] = []
            
            by_document[doc_name].append({
                "page": emb.get("page"),
                "slide": emb.get("slide"),
                "content_preview": emb.get("content", "")[:200] + "..." if emb.get("content") else "",
                "content_length": len(emb.get("content", ""))
            })
        
        return {
            "course_id": course_id,
            "total_chunks_sampled": len(embeddings_result.data),
            "documents_found": list(by_document.keys()),
            "content_by_document": by_document
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "course_id": course_id
        }

# Initialize exam components
exam_generator = ExamGenerator()
exam_session_manager = ExamSessionManager()

# Helper function for downloading files from storage
def download_file(bucket_name: str, file_path: str) -> bytes:
    """Download file from Supabase storage"""
    try:
        result = supabase.storage.from_(bucket_name).download(file_path)
        return result
    except Exception as e:
        print(f"Download failed: {e}")
        raise HTTPException(404, detail=f"File not found: {file_path}")

@app.post("/api/upload-past-paper")
async def upload_past_paper(
    course_id: str = Form(...),
    file: UploadFile = File(...),
    user_id: str = Form("anonymous")
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
    """Generate a practice exam based on course materials and past paper patterns"""
    try:
        print(f"🎯 Generating practice exam for course: {course_id}")
        
        # Validate inputs
        if question_count < 1 or question_count > 50:
            raise HTTPException(400, detail="Question count must be between 1 and 50")
        
        if time_limit < 5 or time_limit > 300:
            raise HTTPException(400, detail="Time limit must be between 5 and 300 minutes")
        
        # Parse question types
        try:
            question_types_list = json.loads(question_types)
        except json.JSONDecodeError:
            raise HTTPException(400, detail="Invalid question_types format")
        
        # Check if course exists and has content
        course_check = supabase.table("courses").select("*").eq("course_id", course_id).execute()
        if not course_check.data:
            raise HTTPException(400, detail="Course not found")
        
        files_check = supabase.table("files").select("filename").eq("course_id", course_id).execute()
        if not files_check.data:
            raise HTTPException(400, detail="No course materials found. Upload files first.")
        
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
        
        # Generate the exam
        result = exam_generator.generate_practice_exam(course_id, exam_specs)
        
        if result.get("status") == "error":
            raise HTTPException(500, detail=result.get("message", "Exam generation failed"))
        
        exam_data = result["exam"]
        
        print(f"✅ Generated exam with {len(exam_data['questions'])} questions")
        
        return {
            "status": "success",
            "exam": exam_data,
            "message": f"Generated {exam_data['question_count']} question exam"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Exam generation failed: {e}")
        raise HTTPException(500, detail=f"Generation failed: {str(e)}")

@app.post("/api/create-exam-session")
async def create_exam_session(
    exam_data: str = Form(...),
    user_id: str = Form("anonymous"),
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
        raise HTTPException(500, detail=f"Session creation failed: {str(e)}")

@app.post("/api/start-exam-session/{session_id}")
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
        raise HTTPException(500, detail=f"Failed to start session: {str(e)}")

@app.post("/api/pause-exam-session/{session_id}")
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
        raise HTTPException(500, detail=f"Failed to pause session: {str(e)}")

@app.post("/api/save-exam-answer")
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
        raise HTTPException(500, detail=f"Failed to save answer: {str(e)}")

@app.post("/api/navigate-exam-question")
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
        raise HTTPException(500, detail=f"Navigation failed: {str(e)}")

@app.post("/api/submit-exam/{session_id}")
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
        raise HTTPException(500, detail=f"Submission failed: {str(e)}")

@app.get("/api/exam-session/{session_id}")
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
        raise HTTPException(500, detail=f"Failed to get session: {str(e)}")

@app.get("/api/exam-history/{user_id}")
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
        raise HTTPException(500, detail=f"Failed to get exam history: {str(e)}")

@app.delete("/api/exam-session/{session_id}")
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
        raise HTTPException(500, detail=f"Deletion failed: {str(e)}")

@app.get("/api/past-papers/{course_id}")
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
        raise HTTPException(500, detail=f"Failed to get past papers: {str(e)}")

@app.post("/api/solve-exam-question")
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
        raise HTTPException(500, f"Solve failed: {str(e)}")

@app.get("/api/exam-analytics/{course_id}/{user_id}")
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
        raise HTTPException(500, detail=f"Analytics failed: {str(e)}")

def calculate_exam_analytics(exam_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive exam analytics"""
    if not exam_history:
        return {
            "average_score": 0,
            "total_exams": 0,
            "improvement_trend": "no_data",
            "strong_topics": [],
            "weak_topics": [],
            "time_efficiency": 0,
            "grade_distribution": {}
        }
    
    completed_exams = [exam for exam in exam_history if exam["status"] == "completed"]
    
    if not completed_exams:
        return {
            "average_score": 0,
            "total_exams": len(exam_history),
            "improvement_trend": "no_completed_exams",
            "strong_topics": [],
            "weak_topics": [],
            "time_efficiency": 0,
            "grade_distribution": {}
        }
    
    # Calculate average score
    scores = [exam["final_score"]["percentage"] for exam in completed_exams if exam.get("final_score")]
    average_score = sum(scores) / len(scores) if scores else 0
    
    # Calculate improvement trend
    improvement_trend = "stable"
    if len(scores) >= 3:
        recent_avg = sum(scores[-3:]) / 3
        earlier_avg = sum(scores[:-3]) / (len(scores) - 3) if len(scores) > 3 else scores[0]
        if recent_avg > earlier_avg + 5:
            improvement_trend = "improving"
        elif recent_avg < earlier_avg - 5:
            improvement_trend = "declining"
    
    # Topic performance analysis
    topic_stats = {}
    for exam in completed_exams:
        final_score = exam.get("final_score", {})
        topic_performance = final_score.get("topic_performance", {})
        
        for topic, performance in topic_performance.items():
            if topic not in topic_stats:
                topic_stats[topic] = {"correct": 0, "total": 0}
            
            topic_stats[topic]["correct"] += performance["correct"]
            topic_stats[topic]["total"] += performance["total"]
    
    # Identify strong and weak topics
    strong_topics = []
    weak_topics = []
    
    for topic, stats in topic_stats.items():
        if stats["total"] >= 3:  # Only consider topics with sufficient data
            accuracy = stats["correct"] / stats["total"]
            if accuracy >= 0.8:
                strong_topics.append({"topic": topic, "accuracy": round(accuracy * 100, 1)})
            elif accuracy <= 0.6:
                weak_topics.append({"topic": topic, "accuracy": round(accuracy * 100, 1)})
    
    # Time efficiency
    time_efficiencies = []
    for exam in completed_exams:
        final_score = exam.get("final_score", {})
        time_metrics = final_score.get("time_metrics", {})
        if time_metrics.get("time_efficiency"):
            time_efficiencies.append(time_metrics["time_efficiency"])
    
    avg_time_efficiency = sum(time_efficiencies) / len(time_efficiencies) if time_efficiencies else 0
    
    # Grade distribution
    grade_distribution = {}
    for exam in completed_exams:
        final_score = exam.get("final_score", {})
        grade = final_score.get("letter_grade", "F")
        grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
    
    return {
        "average_score": round(average_score, 1),
        "total_exams": len(completed_exams),
        "improvement_trend": improvement_trend,
        "strong_topics": sorted(strong_topics, key=lambda x: x["accuracy"], reverse=True)[:5],
        "weak_topics": sorted(weak_topics, key=lambda x: x["accuracy"])[:5],
        "time_efficiency": round(avg_time_efficiency, 1),
        "grade_distribution": grade_distribution,
        "recent_scores": scores[-5:] if len(scores) >= 5 else scores,
        "score_trend": scores
    }

# Scheduled task to auto-submit expired exams
@app.get("/api/admin/auto-submit-expired-exams")
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
        raise HTTPException(500, detail=f"Auto-submit failed: {str(e)}")