# providers/__init__.py
"""Provider compatibility layer: Claude for chat/vision, local model for embeddings."""
from .client import CompatClient, make_client
from .local_embeddings import EMBED_DIM
from .anthropic_chat import stream_text

__all__ = ["CompatClient", "make_client", "EMBED_DIM", "stream_text"]
