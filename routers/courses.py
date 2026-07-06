import json
import logging
import os
import shutil
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from auth import current_user_id, get_current_user, require_course_access
from core import courses_store
from core.courses_store import CourseStoreError
from deps import ENHANCED_MODE, enhanced_delete_file, process_file_enhanced, supabase
from ingest import delete_course, delete_file_from_course, process_file
from storage import upload_file

logger = logging.getLogger(__name__)

router = APIRouter()


# NOTE: the legacy single-file POST /upload/{course_id} route was DELETED.
# It had no authentication and no course-access check, and the frontend never
# calls it (src/lib/api/courses.ts only uses the authenticated bulk POST
# /upload below). Removing it closes the hole outright instead of securing
# dead code.


@router.get("/")
def health_check():
    status_message = "✅ API is running"
    if ENHANCED_MODE:
        status_message += " (Enhanced Mode Active 🚀)"
    return {"status": status_message}


@router.post("/create-course")
def create_course(course_id: str = Form(...), title: str = Form(...),
                  user_id: str = Depends(current_user_id)):
    # Supabase is the single source of truth for course records (the legacy
    # local courses.json fallback is gone — it was lost on every redeploy).
    try:
        if courses_store.course_exists(course_id):
            raise HTTPException(400, detail="Course already exists")
        courses_store.create_course(course_id, title, owner_id=user_id)
    except CourseStoreError:
        logger.exception("Course creation failed for %s", course_id)
        raise HTTPException(500, detail="Failed to create course")

    # Local scratch directories for uploads / vector artifacts.
    os.makedirs(f"data/{course_id}", exist_ok=True)
    os.makedirs(f"vectorstores/{course_id}", exist_ok=True)

    return {"status": "ok", "message": f"Created course {title}"}


@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    course_id: str = Form(...),
    user=Depends(get_current_user)
):
    require_course_access(course_id, user)
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

            # 6) Keep a local scratch copy of the raw file. The file list itself
            # lives in the Supabase `files` table (already written above); the
            # legacy courses.json bookkeeping is gone.
            print("💿 Saving to local storage...")
            try:
                file_path = f"data/{course_id}/{file.filename}"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(content)
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


@router.get("/list-courses")
def list_courses(user_id: str = Depends(current_user_id)):
    """List ONLY the courses the token user owns or has joined."""
    try:
        return {"courses": courses_store.list_courses_for_user(user_id)}
    except CourseStoreError:
        logger.exception("Failed to list courses")
        raise HTTPException(500, detail="Failed to list courses")


@router.get("/list-files")
def list_files(course_id: str, user=Depends(get_current_user)):
    require_course_access(course_id, user)
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


@router.post("/delete-file")
async def delete_file(course_id: str = Form(...), filename: str = Form(...), user=Depends(get_current_user)):
    require_course_access(course_id, user)
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

        return {"status": "ok", "message": f"Deleted {filename} from {course_id}"}
        
    except Exception as e:
        logger.exception("Failed to delete file")
        raise HTTPException(500, detail="Failed to delete file")


@router.post("/delete-course")
async def delete_entire_course(course_id: str = Form(...), user=Depends(get_current_user)):
    require_course_access(course_id, user)
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
        
        # Delete course record (files/embeddings cascade via FK)
        courses_store.delete_course(course_id)

        # Delete from vector store
        success = delete_course(course_id)

        # Clean up local scratch directories
        data_path = f"data/{course_id}"
        if os.path.exists(data_path):
            shutil.rmtree(data_path)

        vectorstore_path = f"vectorstores/{course_id}"
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path)

        return {"status": "ok", "message": f"Deleted course {course_id}"}
        
    except Exception as e:
        logger.exception("Failed to delete course")
        raise HTTPException(500, detail="Failed to delete course")

