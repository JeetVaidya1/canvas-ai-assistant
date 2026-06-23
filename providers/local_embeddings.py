# providers/local_embeddings.py
"""
Local embedding backend that mimics the OpenAI embeddings response shape.

Replaces `client.embeddings.create(model=..., input=...)` with a local
sentence-transformers model (no API key, runs on-device). The returned object
exposes `.data[i].embedding` exactly like the OpenAI SDK so call sites do not
need to change.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import List, Sequence, Union

# BGE-large produces 1024-dim, L2-normalized vectors. This MUST match the
# pgvector column dimension in schema.sql and the FAISS index dimension.
DEFAULT_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
EMBED_DIM = 1024

# BGE retrieval is asymmetric: queries get an instruction prefix, documents do
# not. Applying this to queries only (documents were ingested plain) improves
# ranking without re-embedding the corpus.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None
_model_lock = threading.Lock()


def _get_model():
    """Lazily load the sentence-transformers model exactly once (it is heavy)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                # Imported lazily so importing this module stays cheap.
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(DEFAULT_EMBED_MODEL)
    return _model


@dataclass(frozen=True)
class _EmbeddingItem:
    embedding: List[float]
    index: int


@dataclass(frozen=True)
class _EmbeddingResponse:
    data: List[_EmbeddingItem]
    model: str


def _normalize_input(value: Union[str, Sequence[str]]) -> List[str]:
    """OpenAI accepts a string or a list of strings; normalize to a list."""
    if value is None:
        raise ValueError("embeddings input must not be None")
    if isinstance(value, str):
        return [value]
    items = list(value)
    if not items:
        raise ValueError("embeddings input must not be empty")
    if not all(isinstance(x, str) for x in items):
        raise ValueError("embeddings input must be a string or list of strings")
    return items


def embed_query(text: str) -> List[float]:
    """Embed a search query with the BGE instruction prefix (asymmetric retrieval)."""
    model = _get_model()
    vec = model.encode(
        BGE_QUERY_PREFIX + text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [float(x) for x in vec]


class EmbeddingsNamespace:
    """Mirrors `client.embeddings` with a `.create(...)` method."""

    def create(self, *, model: str = DEFAULT_EMBED_MODEL,
               input: Union[str, Sequence[str]], **_ignored) -> _EmbeddingResponse:
        texts = _normalize_input(input)
        st_model = _get_model()
        # normalize_embeddings=True yields unit vectors so cosine == dot product,
        # which is what the pgvector RPCs and FAISS assume.
        vectors = st_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        items = [
            _EmbeddingItem(embedding=[float(x) for x in vec], index=i)
            for i, vec in enumerate(vectors)
        ]
        return _EmbeddingResponse(data=items, model=DEFAULT_EMBED_MODEL)
