from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from ingest import process_file
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

# 1. Load the secret note (.env)
load_dotenv()

# 2. Read the URL & KEY from that note
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# 3. Make a Supabase “client” to talk to the database
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

    # 4) **Ingest & chunk** via your updated ingest.py
    #    This call will extract text, embed, store in pgvector, and return chunk previews
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
    return {"status": "✅ API is running"}

@app.post("/create-course")
def create_course(course_id: str = Form(...), title: str = Form(...)):
    courses = load_courses()
    if course_id in courses:
        return {"status": "exists", "message": "Course already exists"}

    os.makedirs(f"data/{course_id}", exist_ok=True)
    os.makedirs(f"vectorstores/{course_id}", exist_ok=True)

    courses[course_id] = {"title": title, "files": []}
    save_courses(courses)

    return {"status": "ok", "message": f"Created course {title}"}

@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    course_id: str = Form(...)
):
    courses = load_courses()
    if course_id not in courses:
        return {"status": "error", "message": "Invalid course_id"}

    chunks_preview = []
    for file in files:
        contents = await file.read()
        chunks = process_file(file.filename, contents, course_id)
        chunks_preview.extend(chunks[:2])  # Preview

        # Save file metadata
        courses[course_id]["files"].append(file.filename)
        file_path = f"data/{course_id}/{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(contents)

    save_courses(courses)
    return {"status": "ok", "chunks": chunks_preview}

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
            raise HTTPException(500, detail=f"Couldn’t create session: {e}")

    # 2) Record the user’s question
    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "role":       "user",
            "content":    question,
            "timestamp":  datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        raise HTTPException(500, detail=f"Couldn’t save question: {e}")

    # 3) Generate the AI’s answer
    answer = ask_question(question, course_id)

    # 4) Record the assistant’s answer
    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "role":       "assistant",
            "content":    answer,
            "timestamp":  datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        raise HTTPException(500, detail=f"Couldn’t save answer: {e}")

    return {
        "session_id": session_id,
        "question":   question,
        "answer":     answer
    }



@app.get("/list-courses")
def list_courses():
    courses_file = "courses.json"
    if not os.path.exists(courses_file):
        return {"courses": []}
    
    with open(courses_file, "r") as f:
        courses = json.load(f)
    
    return {"courses": [{"course_id": cid, "title": data["title"]} for cid, data in courses.items()]}

@app.get("/list-files")
def list_files(course_id: str):
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
        # First, delete from vector database using ingest module
        deleted = delete_file_from_course(course_id, filename)
        if deleted:
            # Remove from courses.json
            courses = load_courses()
            if course_id in courses and filename in courses[course_id]["files"]:
                courses[course_id]["files"].remove(filename)
                save_courses(courses)
            
            # Also remove from vectorstores files.json (the one that list-files reads)
            files_json_path = f"vectorstores/{course_id}/files.json"
            if os.path.exists(files_json_path):
                with open(files_json_path, "r") as f:
                    file_list = json.load(f)
                
                if filename in file_list:
                    file_list.remove(filename)
                    with open(files_json_path, "w") as f:
                        json.dump(file_list, f, indent=2)
            
            # Also delete the actual file from the data directory
            file_path = f"data/{course_id}/{filename}"
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return {"status": "ok", "message": f"Deleted {filename} from {course_id}"}
        else:
            return {"status": "error", "message": "File not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/delete-course")
async def delete_entire_course(course_id: str = Form(...)):
    try:
        # First, delete from vector database using ingest module
        success = delete_course(course_id)
        
        # Then, remove from courses.json
        courses = load_courses()
        if course_id in courses:
            del courses[course_id]
            save_courses(courses)
            
            # Also clean up the data directory
            data_path = f"data/{course_id}"
            if os.path.exists(data_path):
                shutil.rmtree(data_path)
            
            return {"status": "ok", "message": f"Deleted course {course_id}"}
        else:
            return {"status": "error", "message": "Course not found"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

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
        raise HTTPException(500, detail=f"Couldn’t fetch sessions: {e}")
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
        raise HTTPException(500, detail=f"Couldn’t fetch messages: {e}")
    return {"messages": messages}