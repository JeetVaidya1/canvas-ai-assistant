"""System endpoints: /system-status, the ENABLE_DEBUG_ENDPOINTS gate over real
routes, and /health/rag failure hygiene.
"""
import pytest

import routers.system as system_router
from fakes_endpoints import LEAK_MARKER, FakeSupabase

pytestmark = pytest.mark.unit

DEBUG_ROUTES = [
    "/admin/usage",
    "/debug-course-content/cs101",
    "/course-subject-detection/cs101",
    "/debug-vector-content/cs101",
]


def test_system_status_200_with_request_id(client):
    resp = client.get("/system-status")
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID")
    body = resp.json()
    assert isinstance(body["enhanced_mode"], bool)
    assert isinstance(body["conversational_mode"], bool)
    assert body["capabilities"]["quiz_assistance"] is True
    assert body["version"] in ("2.0.0", "3.0.0")


@pytest.mark.parametrize("path", DEBUG_ROUTES)
def test_debug_routes_404_when_flag_unset(client, monkeypatch, path):
    monkeypatch.delenv("ENABLE_DEBUG_ENDPOINTS", raising=False)
    resp = client.get(path)
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Not Found"}


def test_debug_course_content_200_when_enabled(client, monkeypatch):
    monkeypatch.setenv("ENABLE_DEBUG_ENDPOINTS", "1")
    monkeypatch.setattr(
        system_router,
        "supabase",
        FakeSupabase({
            "courses": [{"course_id": "cs101", "title": "Algorithms"}],
            "files": [],
            "embeddings": [],
        }),
    )
    resp = client.get("/debug-course-content/cs101")
    assert resp.status_code == 200
    body = resp.json()
    assert body["course_id"] == "cs101"
    assert body["course_info"]["title"] == "Algorithms"
    assert body["vector_store_status"]["populated"] is False


def test_admin_usage_200_when_enabled(client, monkeypatch):
    monkeypatch.setenv("ENABLE_DEBUG_ENDPOINTS", "1")
    resp = client.get("/admin/usage")
    assert resp.status_code == 200
    assert isinstance(resp.json(), dict)


def test_health_rag_reports_no_embeddings(client, monkeypatch):
    monkeypatch.setattr(system_router, "supabase", FakeSupabase({"embeddings": []}))
    resp = client.get("/health/rag")
    assert resp.status_code == 200
    assert resp.json() == {"ok": False, "reason": "no embeddings yet"}


def test_health_rag_failure_does_not_leak_internals(client, monkeypatch):
    monkeypatch.setattr(system_router, "supabase", FakeSupabase(fail=True))
    resp = client.get("/health/rag")
    assert resp.status_code == 200
    assert resp.json() == {"ok": False, "error": "rag health check failed"}
    assert LEAK_MARKER not in resp.text
