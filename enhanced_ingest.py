# enhanced_ingest.py - Phase 1: Basic multimodal processing

import os
import io
import tempfile
import json
import base64
import fitz  # PyMuPDF
from dotenv import load_dotenv
import pdfplumber
from pptx import Presentation
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from vector_store import VectorStore

# Load environment variables
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
vector_store = VectorStore()

def extract_text_from_pdf_basic(file_bytes: bytes) -> str:
    """Basic text extraction (your existing method)"""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def extract_images_from_pdf(file_bytes: bytes) -> list:
    """Extract images from PDF using PyMuPDF"""
    images = []
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_document, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        img_b64 = base64.b64encode(img_data).decode()
                        
                        images.append({
                            'data': img_b64,
                            'page': page_num + 1,
                            'index': img_index
                        })
                    
                    pix = None
                except Exception as e:
                    print(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
        
        pdf_document.close()
    except Exception as e:
        print(f"Error processing PDF for images: {e}")
    
    return images

def describe_image_with_gpt4v(image_data: str, context: str = "") -> str:
    """Use GPT-4V to describe images"""
    try:
        prompt = f"""
        Describe this image in detail for educational purposes. Focus on:
        1. What the image shows
        2. Key educational concepts it illustrates
        3. Important details students should notice
        
        Context: {context}
        
        Make it educational and clear for students.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # Use GPT-4V
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"}
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.2
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error describing image: {e}")
        return "Image description unavailable."

def process_file_enhanced(filename: str, file_bytes: bytes, course_id: str) -> list[dict]:
    """
    Enhanced file processing that handles both text and images
    """
    print(f"🚀 Enhanced processing: {filename}")
    
    # Step 1: Extract text (existing method)
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        raw_text = extract_text_from_pdf_basic(file_bytes)
        # Also extract images from PDF
        images = extract_images_from_pdf(file_bytes)
    elif ext == ".pptx":
        raw_text = extract_text_from_pptx(file_bytes)
        images = []  # PowerPoint image extraction can be added later
    elif ext == ".docx":
        raw_text = extract_text_from_docx(file_bytes)
        images = []  # Word image extraction can be added later
    else:
        return [{"chunk": f"Unsupported file type: {ext}"}]

    all_chunks = []
    
    # Step 2: Process text chunks (existing logic)
    if raw_text and len(raw_text.strip()) >= 10:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_chunks = splitter.split_text(raw_text)
        
        # Embed text chunks
        if text_chunks:
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_chunks
            )
            embeddings = [r.embedding for r in response.data]
            
            # Store text chunks
            for idx, (chunk, emb) in enumerate(zip(text_chunks, embeddings)):
                vector_store.add(
                    course_id=course_id,
                    doc_name=filename,
                    chunk_id=f"text_{idx}",
                    embedding=emb,
                    content=chunk
                )
                
                all_chunks.append({
                    "type": "text",
                    "chunk": chunk[:200] + "..." if len(chunk) > 200 else chunk
                })
    
    # Step 3: Process images (NEW!)
    print(f"📸 Found {len(images)} images to process")
    
    for i, image_info in enumerate(images[:5]):  # Limit to 5 images for now
        try:
            # Describe the image using GPT-4V
            description = describe_image_with_gpt4v(
                image_info['data'],
                context=f"This image is from {filename}, page {image_info['page']}"
            )
            
            # Create enhanced content with image description
            image_content = f"""[IMAGE CONTENT - Page {image_info['page']}]
{description}
[Source: {filename}, Page {image_info['page']}]"""
            
            # Create embedding for the image description
            emb_resp = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=[image_content]
            )
            embedding = emb_resp.data[0].embedding
            
            # Store image description
            vector_store.add(
                course_id=course_id,
                doc_name=filename,
                chunk_id=f"image_{i}",
                embedding=embedding,
                content=image_content
            )
            
            all_chunks.append({
                "type": "image",
                "chunk": description[:200] + "..." if len(description) > 200 else description,
                "page": image_info['page']
            })
            
            print(f"✅ Processed image {i+1} from page {image_info['page']}")
            
        except Exception as e:
            print(f"❌ Error processing image {i}: {e}")
    
    print(f"✅ Enhanced processing complete: {len(all_chunks)} chunks")
    return all_chunks[:5]  # Return preview

# Keep your existing functions for compatibility
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

# Your existing delete functions
def delete_file_from_course(course_id: str, filename: str) -> bool:
    """Delete all embeddings for a specific file from a course."""
    try:
        from supabase import create_client
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
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
        from supabase import create_client
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
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