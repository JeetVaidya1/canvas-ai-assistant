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
