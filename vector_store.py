import os
import pickle
from typing import List

from openai import OpenAI
from dotenv import load_dotenv
from pdfplumber import open as pdf_open
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Embedding setup
embeddings = OpenAIEmbeddings(api_key=api_key)

def load_pdf(file_path: str) -> List[Document]:
    """Load and split a PDF into smaller chunks for embedding."""
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return pages

def create_vector_store(file_path: str, save_path: str = "vector_store"):
    """Create and save a FAISS vector store from a PDF file."""
    print(f"📄 Loading PDF: {file_path}")
    docs = load_pdf(file_path)

    print("🔢 Splitting and embedding content...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    db = FAISS.from_documents(split_docs, embeddings)

    print(f"💾 Saving vector store to: {save_path}")
    db.save_local(save_path)
    print("✅ Done.")
