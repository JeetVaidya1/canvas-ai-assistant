"""Error envelope hygiene.

Truly unhandled exceptions must come back as the clean
``{"error": "internal_server_error", "message": ...}`` envelope from
errors.install_error_handlers — never a stack trace or exception text.
Deliberate HTTPExceptions keep FastAPI's ``{"detail": ...}`` shape and must
not echo internals either.
"""
import pytest

import routers.chat as chat_router
from core import courses_store
from fakes_endpoints import LEAK_MARKER, FakeSupabase

pytestmark = pytest.mark.unit

SECRET = "secret-internal-token=abc123"


@pytest.fixture
def exploding_list_courses(monkeypatch):
    """Make courses_store.list_courses_for_user raise a NON-CourseStoreError.

    The router only catches CourseStoreError, so this escapes the handler and
    exercises the app-wide unhandled-exception envelope. (/list-courses now
    requires auth, so these tests run as an authenticated user.)
    """
    def _boom(_user_id):
        raise ValueError(f"connection string leaked: {SECRET}")

    monkeypatch.setattr(courses_store, "list_courses_for_user", _boom)


def test_unhandled_exception_returns_clean_envelope(client, as_user, exploding_list_courses):
    as_user("u-err")
    resp = client.get("/list-courses")
    assert resp.status_code == 500
    assert resp.json() == {
        "error": "internal_server_error",
        "message": "Something went wrong on our end. Please try again.",
    }


def test_unhandled_exception_leaks_no_internals(client, as_user, exploding_list_courses):
    as_user("u-err")
    resp = client.get("/list-courses")
    assert resp.status_code == 500
    assert SECRET not in resp.text
    assert "ValueError" not in resp.text
    assert "Traceback" not in resp.text
    assert "connection string" not in resp.text


def test_envelope_has_exactly_error_and_message_keys(client, as_user, exploding_list_courses):
    as_user("u-err")
    body = client.get("/list-courses").json()
    assert set(body.keys()) == {"error", "message"}


def test_deliberate_500_keeps_detail_shape_without_internals(client, as_user, monkeypatch):
    as_user("u-err")
    monkeypatch.setattr(chat_router, "supabase", FakeSupabase(fail=True))
    resp = client.get("/sessions")
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Couldn't fetch sessions"}
    assert LEAK_MARKER not in resp.text
    assert "RuntimeError" not in resp.text


def test_unknown_route_is_plain_404(client):
    resp = client.get("/definitely-not-a-route")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Not Found"}


def test_validation_error_is_structured_422(client, as_user):
    as_user("u-err")
    resp = client.post("/ask", data={})
    assert resp.status_code == 422
    body = resp.json()
    assert isinstance(body["detail"], list)
    missing = {tuple(err["loc"]) for err in body["detail"]}
    assert ("body", "question") in missing
    assert ("body", "course_id") in missing
