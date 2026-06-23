# providers/model_map.py
"""
Map whatever model string the legacy code passes (gpt-5, gpt-5-mini, gpt-4o,
gpt-5-vision, or an already-correct claude-* id) onto a concrete Claude model.

Most engines read model names from env (MODEL_DEFAULT / MODEL_COMPLEX / etc.),
so setting those in .env routes everything here. A few call sites hardcode
"gpt-*" strings; this resolver catches those too.
"""
from __future__ import annotations

import os

# Concrete Claude model ids.
CLAUDE_FAST = os.getenv("CLAUDE_FAST_MODEL", "claude-haiku-4-5-20251001")
CLAUDE_SMART = os.getenv("CLAUDE_SMART_MODEL", "claude-sonnet-4-6")


def resolve_chat_model(name: str | None) -> str:
    """Return a concrete Claude model id for a requested model name."""
    if not name:
        return CLAUDE_SMART
    lowered = name.lower()
    # Already a Claude id — respect it verbatim.
    if "claude" in lowered:
        return name
    # Cheap/fast tier.
    if "mini" in lowered or "haiku" in lowered or "fast" in lowered:
        return CLAUDE_FAST
    # Everything else (gpt-5, gpt-4o, gpt-5-vision, ...) -> the smart tier.
    # Sonnet is vision-capable, so vision calls resolve here correctly.
    return CLAUDE_SMART
