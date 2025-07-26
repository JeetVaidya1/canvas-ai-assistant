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

# Load environment variables for OpenAI
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize vector store
vector_store = VectorStore()


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
    # TODO: implement removal from pgvector and Supabase storage
    return False


def delete_course(course_id: str) -> bool:
    # TODO: implement dropping all chunks and cleaning storage
    return False
