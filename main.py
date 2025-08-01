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
from query_engine import ask_question
from storage import upload_file
from ingest import process_file

# NEW: Try to import enhanced modules
try:
    from enhanced_ingest import process_file_enhanced, delete_file_from_course as enhanced_delete_file
    from enhanced_query_engine import enhanced_ask_question
    ENHANCED_MODE = True
    print("✅ Enhanced multimodal system loaded!")
except ImportError as e:
    print(f"⚠️ Enhanced system not available: {e}")
    ENHANCED_MODE = False

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
            "uploaded_at": "now()"
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
                    "uploaded_at": "now()"
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
    question:    str             = Form(...),
    course_id:   str             = Form(...),
    session_id:  str | None      = Form(None),
    user_id:     str             = Form("anonymous")  # TODO: wire in real auth
):
    # 1) Create a new chat_session if none was provided
    if not session_id:
        try:
            resp = supabase.table("chat_sessions").insert({
                "user_id":    user_id,
                "course_id":  course_id,
                "title":      question[:50],
                "created_at": datetime.utcnow().isoformat()
            }).execute()
            session_id = resp.data[0]["id"]
        except Exception as e:
            raise HTTPException(500, detail=f"Couldn't create session: {e}")

    # 2) Record the user's question
    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "role":       "user",
            "content":    question,
            "timestamp":  datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        raise HTTPException(500, detail=f"Couldn't save question: {e}")

    # 3) **ENHANCED: Generate the AI's answer with multimodal support**
    try:
        if ENHANCED_MODE:
            print("🤖 Using enhanced question answering...")
            answer = enhanced_ask_question(question, course_id)
            print("✅ Enhanced answer generated!")
        else:
            print("📝 Using basic question answering...")
            answer = ask_question(question, course_id)
    except Exception as e:
        print(f"❌ Enhanced QA failed, using fallback: {e}")
        answer = ask_question(question, course_id)

    # 4) Record the assistant's answer
    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "role":       "assistant",
            "content":    answer,
            "timestamp":  datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        raise HTTPException(500, detail=f"Couldn't save answer: {e}")

    return {
        "session_id": session_id,
        "question":   question,
        "answer":     answer
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

# **NEW: Enhanced system status endpoint**
@app.get("/system-status")
def get_system_status():
    """
    Get current system capabilities and status
    """
    return {
        "enhanced_mode": ENHANCED_MODE,
        "capabilities": {
            "multimodal_processing": ENHANCED_MODE,
            "image_extraction": ENHANCED_MODE,
            "enhanced_formatting": ENHANCED_MODE,
            "question_classification": ENHANCED_MODE
        },
        "version": "1.1.0" if ENHANCED_MODE else "1.0.0"
    }