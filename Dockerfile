# Vindexa backend — FastAPI + local BGE embeddings, for Fly.io (always-on container).
# Python 3.12 to match the dev .venv.
FROM python:3.12-slim AS base

# System libraries:
#  - poppler-utils + tesseract-ocr: required by pdf2image / pytesseract (document ingest)
#  - libgl1 + libglib2.0-0: required by opencv-python at import time
#  - curl: used by Fly's healthcheck tooling / debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Keep Python lean and unbuffered (logs flush immediately for Fly).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Bake + load the embedding model from this path (set before the model download below).
    HF_HOME=/opt/hf-cache \
    LOCAL_EMBED_MODEL=BAAI/bge-large-en-v1.5

WORKDIR /app

# Install CPU-only torch FIRST so sentence-transformers doesn't pull the ~2GB CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Then the rest of the dependencies (torch requirement is already satisfied).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the BGE model into the image so cold starts don't download 1.3GB at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${LOCAL_EMBED_MODEL}')"

# Application code (see .dockerignore for what is excluded — frontend, data/, vectorstores/, etc.)
COPY . .

EXPOSE 8080

# Bind to 0.0.0.0:8080 to match fly.toml internal_port.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
