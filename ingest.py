import os
import io
import tempfile
import json
from dotenv import load_dotenv
import pdfplumber
from pptx import Presentation
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from vector_store import VectorStore
from supabase import create_client

# Load environment variables for OpenAI and Supabase
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize vector store and Supabase client
vector_store = VectorStore()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def extract_text_from_pptx(file_bytes: bytes) -> str:
    """Extract text from PPTX bytes."""
    prs = Presentation(io.BytesIO(file_bytes))
    text_runs = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text_runs.append(shape.text)
    return "\n".join(text_runs)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        doc = Document(tmp.name)
    os.remove(tmp.name)
    return "\n".join(p.text for p in doc.paragraphs)


def process_file(filename: str, file_bytes: bytes, course_id: str) -> list[dict]:
    """
    Ingest a file: extract text, split into chunks, embed, and store in pgvector.
    Returns a preview list of chunks.
    """
    # 1) Extract raw text based on extension
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        raw_text = extract_text_from_pdf(file_bytes)
    elif ext == ".pptx":
        raw_text = extract_text_from_pptx(file_bytes)
    elif ext == ".docx":
        raw_text = extract_text_from_docx(file_bytes)
    else:
        return [{"chunk": f"Unsupported file type: {ext}"}]

    if not raw_text or len(raw_text.strip()) < 10:
        return [{"chunk": f"No usable text found in {filename}"}]

    # 2) Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(raw_text)

    # 3) Embed all chunks in one batch
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=chunks
    )
    embeddings = [r.embedding for r in response.data]

    # 4) Store each chunk + vector
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vector_store.add(
            course_id=course_id,
            doc_name=filename,
            chunk_id=idx,
            embedding=emb,
            content=chunk
        )

    # 5) Return preview of first 5 chunks
    return [{"chunk": c} for c in chunks[:5]]


def delete_file_from_course(course_id: str, filename: str) -> bool:
    """Delete all embeddings for a specific file from a course."""
    try:
        # Delete from embeddings table
        supabase.table("embeddings").delete().eq("course_id", course_id).eq("doc_name", filename).execute()
        
        # Delete from files table
        supabase.table("files").delete().eq("course_id", course_id).eq("filename", filename).execute()
        
        # Delete from Supabase storage
        storage_path = f"{course_id}/{filename}"
        try:
            supabase.storage.from_("course-files").remove([storage_path])
        except Exception as e:
            print(f"Warning: Storage deletion failed: {e}")
        
        return True
    except Exception as e:
        print(f"Error deleting file {filename} from course {course_id}: {e}")
        return False


def delete_course(course_id: str) -> bool:
    """Delete an entire course and all its data."""
    try:
        # Delete all embeddings for this course
        supabase.table("embeddings").delete().eq("course_id", course_id).execute()
        
        # Delete all files metadata for this course
        supabase.table("files").delete().eq("course_id", course_id).execute()
        
        # Delete all chat sessions and messages for this course
        sessions_resp = supabase.table("chat_sessions").select("id").eq("course_id", course_id).execute()
        session_ids = [s["id"] for s in sessions_resp.data]
        
        for session_id in session_ids:
            supabase.table("messages").delete().eq("session_id", session_id).execute()
        
        supabase.table("chat_sessions").delete().eq("course_id", course_id).execute()
        
        # Delete the course itself
        supabase.table("courses").delete().eq("course_id", course_id).execute()
        
        # Delete all files from storage for this course
        try:
            # List all files in the course folder
            files_list = supabase.storage.from_("course-files").list(course_id)
            if files_list:
                file_paths = [f"{course_id}/{f['name']}" for f in files_list]
                supabase.storage.from_("course-files").remove(file_paths)
        except Exception as e:
            print(f"Warning: Storage cleanup failed: {e}")
        
        return True
    except Exception as e:
        print(f"Error deleting course {course_id}: {e}")
        return False