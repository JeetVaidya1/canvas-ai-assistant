import os
import io
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import pdfplumber
from pptx import Presentation
from docx import Document
import shutil
import json

def extract_text_from_pdf(file_bytes):
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

def extract_text_from_pptx(file_bytes):
    prs = Presentation(io.BytesIO(file_bytes))
    text = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text.append(shape.text)
    return "\n".join(text)

def extract_text_from_docx(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        doc = Document(tmp.name)
    os.remove(tmp.name)
    return "\n".join([p.text for p in doc.paragraphs])

def process_file(filename, file_bytes, course_id):
    # Step 1: Detect file type
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

    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(raw_text)

    # Step 3: Embed the chunks
    embeddings = OpenAIEmbeddings()
    db_dir = os.path.join("vectorstores", course_id)
    os.makedirs(db_dir, exist_ok=True)

    index_file = os.path.join(db_dir, "index.faiss")
    if os.path.exists(index_file):
        vectorstore = FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_texts(chunks)
    else:
        vectorstore = FAISS.from_texts(chunks, embeddings)

    # Track uploaded filenames for frontend
    files_json_path = os.path.join(db_dir, "files.json")
    if os.path.exists(files_json_path):
        with open(files_json_path, "r") as f:
            uploaded_files = json.load(f)
    else:
        uploaded_files = []

    if filename not in uploaded_files:
        uploaded_files.append(filename)
        with open(files_json_path, "w") as f:
            json.dump(uploaded_files, f, indent=2)

    # Also track metadata for deletion purposes
    metadata_path = os.path.join(db_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata[filename] = chunks
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    vectorstore.save_local(db_dir)
    return [{"chunk": chunk} for chunk in chunks[:5]]

def delete_file_from_course(course_id, filename):
    db_dir = os.path.join("vectorstores", course_id)
    
    # Remove from files.json
    files_json_path = os.path.join(db_dir, "files.json")
    files_updated = False
    
    if os.path.exists(files_json_path):
        try:
            with open(files_json_path, "r") as f:
                uploaded_files = json.load(f)
            
            if filename in uploaded_files:
                uploaded_files.remove(filename)
                with open(files_json_path, "w") as f:
                    json.dump(uploaded_files, f, indent=2)
                files_updated = True
        except:
            pass
    
    # Try to rebuild vectorstore if metadata exists
    metadata_path = os.path.join(db_dir, "metadata.json")
    metadata_updated = False
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            if filename in metadata:
                # Remove chunks for this file
                del metadata[filename]
                metadata_updated = True

                # Rebuild vectorstore from remaining chunks
                all_texts = []
                for chunks in metadata.values():
                    all_texts.extend(chunks)

                embeddings = OpenAIEmbeddings()
                if all_texts:
                    new_vs = FAISS.from_texts(all_texts, embeddings)
                    new_vs.save_local(db_dir)
                else:
                    # No texts left, remove the vector store files
                    for file in ["index.faiss", "index.pkl"]:
                        file_path = os.path.join(db_dir, file)
                        if os.path.exists(file_path):
                            os.remove(file_path)

                # Save updated metadata
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
        except:
            pass

    return files_updated or metadata_updated

def delete_course(course_id):
    db_dir = os.path.join("vectorstores", course_id)
    if not os.path.exists(db_dir):
        return False

    # Remove from index
    index_path = os.path.join("vectorstores", "course_index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path, "r") as f:
                index = json.load(f)
            if course_id in index:
                del index[course_id]
                with open(index_path, "w") as f:
                    json.dump(index, f)
        except:
            pass

    # Delete folder
    try:
        shutil.rmtree(db_dir)
        return True
    except:
        return False