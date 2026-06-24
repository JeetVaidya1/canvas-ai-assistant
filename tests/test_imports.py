"""Import-smoke safety net.

The single highest-value protection for the upcoming refactor: every module must
import cleanly. A moved symbol, a broken relative import, or a circular import
fails here immediately — before it ever reaches a route handler in production.

Dead scripts (debug_live_api, test_gpt4v, check_endpoints, reingest, cost_analysis)
are intentionally excluded: they're slated for deletion in Phase 1.
"""
import importlib

import pytest

PROVIDERS = [
    "providers.model_map",
    "providers.client",
    "providers.anthropic_chat",
    "providers.structured",
    "providers.local_embeddings",
    "providers.claude_auth",
    "providers.pricing",
]

RAG = [
    "rag.retrieval",
    "rag.rerank",
]

ENGINES = [
    "response_formatter",
    "storage",
    "vector_store",
    "ingest",
    "enhanced_ingest",
    "query_engine",
    "enhanced_query_engine",
    "conversational_rag_engine",
    "quiz_assistant_engine",
    "quiz_engine",
    "exam_generator",
    "exam_session_manager",
    "practice_generator",
    "notes_engine",
    "planner_engine",
    "flashcard_engine",
    "learning_analytics",
    "readiness_engine",
    "review_engine",
    "mistake_engine",
    "concept_graph",
    "context_pack",
    "github_engine",
    "canvas_engine",
    "sharing_engine",
    "socratic_engine",
    "feynman_engine",
    "exports",
    "auth",
    "errors",
    "rate_limit",
    "usage_tracker",
    "deps",
]

ROUTERS = [
    f"routers.{name}"
    for name in (
        "ai_export", "analytics", "auth_api", "canvas_lms", "chat", "concepts",
        "courses", "exams", "exports_api", "flashcards", "github_io", "notes",
        "planner", "practice", "quiz", "reviews", "sharing", "system", "tutor",
    )
]

ALL_MODULES = PROVIDERS + RAG + ENGINES + ROUTERS + ["main"]


@pytest.mark.smoke
@pytest.mark.parametrize("module_name", ALL_MODULES)
def test_module_imports_cleanly(module_name):
    """Each app module imports without raising (no network/DB at import time)."""
    importlib.import_module(module_name)


@pytest.mark.smoke
def test_fastapi_app_builds():
    """The FastAPI app exists with all routers wired.

    This version of FastAPI keeps included routers as ``_IncludedRouter`` wrappers
    in ``app.routes`` rather than flattening their paths, so the OpenAPI schema is
    the reliable surface to assert the full route table against.
    """
    import main

    paths = main.app.openapi()["paths"]
    assert "/system-status" in paths
    # Representative sample proving include_router ran across multiple routers.
    assert "/quiz/generate" in paths
    # A real app surface, not just the FastAPI defaults.
    assert len(paths) > 50
