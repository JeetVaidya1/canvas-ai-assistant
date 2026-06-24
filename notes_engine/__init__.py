# notes_engine — Conversational, polished notes (RAG + QA polish)
#
# Facade package: re-exports the exact public API that previously lived in the
# single-file module ``notes_engine.py`` so that every existing
# ``import notes_engine`` / ``from notes_engine import X`` keeps working
# identically. Import-time side effects (client + vector store construction)
# happen via ``.config`` below, preserving original load-time behavior.

# Module-level singletons + config (constructed at import time, as before)
from .config import (
    openai_client,
    vector_store,
    MODEL_DEFAULT,
    MODEL_COMPLEX,
    EMBED_MODEL,
    RAG_TOP_K_FILE,
    MAX_TOK_DEFAULT,
    MAX_TOK_COMPLEX,
    ALLOW_GENERAL_FILL,
    INCLUDE_FLASHCARDS,
    SYSTEM_STYLE,
)

# Utilities
from .helpers import (
    _is_dict,
    _safe_get,
    _safe_json_obj,
    _prettify_notes,
    _nice_fallback_title,
)

# Retrieval helpers
from .retrieval import (
    _sanitize_chunks,
    _get_file_chunks_by_meta,
    extract_content_from_files,
)

# Prompt builders + schema
from .prompts import (
    _build_combined_context,
    _route,
    _notes_instruction,
    _FLASHCARD_SCHEMA,
)

# Core generation + extraction
from .generation import (
    generate_detailed_notes,
    extract_topics_from_content,
    _generate_flashcards,
    generate_notes_from_files,
)

# DB helpers
from .persistence import (
    save_notes_to_db,
    get_notes_from_db,
    delete_note_from_db,
)
