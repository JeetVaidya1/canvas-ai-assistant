# providers/client.py
"""
Unified, OpenAI-SDK-shaped client backed by Claude (chat/vision) + a local
sentence-transformers model (embeddings).

Usage (drop-in for `OpenAI(api_key=...)`):

    from providers import make_client
    client = make_client()
    client.chat.completions.create(model="gpt-5", messages=[...])
    client.embeddings.create(model="...", input=["text"])
"""
from __future__ import annotations

from .anthropic_chat import ChatNamespace
from .local_embeddings import EmbeddingsNamespace


class CompatClient:
    """Exposes the `.chat.completions.create` and `.embeddings.create` surface."""

    def __init__(self):
        self.chat = ChatNamespace()
        self.embeddings = EmbeddingsNamespace()


def make_client(*_args, **_kwargs) -> CompatClient:
    """Factory mirroring `OpenAI(api_key=...)`; extra args are ignored."""
    return CompatClient()
