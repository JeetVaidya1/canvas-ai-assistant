"""Auth boundary at the HTTP layer.

Protected endpoints must reject requests with no/malformed/invalid credentials
(401) and serve authenticated ones. Token verification is exercised against a
fake GoTrue client (``auth._auth_client``) so nothing touches the network.
"""
from collections import OrderedDict
from types import SimpleNamespace

import pytest

import auth
import routers.chat as chat_router
from fakes_endpoints import FakeSupabase

pytestmark = pytest.mark.unit

PROTECTED_ENDPOINTS = [
    ("get", "/sessions"),
    ("get", "/sessions/some-session/messages"),
    ("delete", "/sessions/some-session"),
    ("post", "/create-course"),
    ("post", "/ask"),
    ("post", "/ask/stream"),
    ("post", "/api/create-exam-session"),
    ("post", "/api/start-exam-session/some-session"),
    ("post", "/api/pause-exam-session/some-session"),
    ("post", "/api/save-exam-answer"),
    ("post", "/api/navigate-exam-question"),
    ("post", "/api/submit-exam/some-session"),
    ("get", "/api/exam-session/some-session"),
    ("delete", "/api/exam-session/some-session"),
    ("get", "/api/exam-history/any-user"),
    ("get", "/list-courses"),
    ("get", "/list-files?course_id=cs101"),
    ("post", "/delete-course"),
]


@pytest.mark.parametrize("method,path", PROTECTED_ENDPOINTS)
def test_no_auth_header_is_401(client, method, path):
    resp = getattr(client, method)(path)
    assert resp.status_code == 401
    assert resp.json() == {"detail": "Missing or malformed Authorization header"}


@pytest.mark.parametrize("header", ["Basic dXNlcjpwdw==", "bearer", "Token abc", ""])
def test_malformed_auth_header_is_401(client, header):
    resp = client.get("/sessions", headers={"Authorization": header})
    assert resp.status_code == 401


def test_invalid_bearer_token_is_401(client, monkeypatch):
    class _RejectingGoTrue:
        def get_user(self, _token):
            raise RuntimeError("gotrue says no")

    monkeypatch.setattr(auth, "_auth_client", SimpleNamespace(auth=_RejectingGoTrue()))
    monkeypatch.setattr(auth, "_cache", OrderedDict())
    resp = client.get("/sessions", headers={"Authorization": "Bearer bogus-token"})
    assert resp.status_code == 401
    assert resp.json() == {"detail": "Invalid or expired session"}
    # Internal failure text must not leak.
    assert "gotrue" not in resp.text


def test_valid_bearer_token_reaches_handler(client, monkeypatch):
    """Full path: Bearer token -> fake GoTrue verify -> handler runs as that user."""
    user = SimpleNamespace(id="u-gotrue", email="real@user.dev")

    class _AcceptingGoTrue:
        def get_user(self, _token):
            return SimpleNamespace(user=user)

    monkeypatch.setattr(auth, "_auth_client", SimpleNamespace(auth=_AcceptingGoTrue()))
    monkeypatch.setattr(auth, "_cache", OrderedDict())
    fake = FakeSupabase({"chat_sessions": [
        {"id": "s1", "user_id": "u-gotrue", "title": "mine"},
        {"id": "s2", "user_id": "someone-else", "title": "not mine"},
    ]})
    monkeypatch.setattr(chat_router, "supabase", fake)

    resp = client.get("/sessions", headers={"Authorization": "Bearer good-token"})
    assert resp.status_code == 200
    sessions = resp.json()["sessions"]
    assert [s["id"] for s in sessions] == ["s1"]


def test_dependency_override_gives_200(client, as_user, monkeypatch):
    as_user("u-override")
    monkeypatch.setattr(chat_router, "supabase", FakeSupabase({"chat_sessions": []}))
    resp = client.get("/sessions")
    assert resp.status_code == 200
    assert resp.json() == {"sessions": []}


def test_401_response_still_carries_request_id(client):
    resp = client.get("/sessions")
    assert resp.status_code == 401
    assert resp.headers.get("X-Request-ID")


def test_401_body_is_detail_shape_only(client):
    resp = client.get("/sessions")
    body = resp.json()
    assert set(body.keys()) == {"detail"}
    assert "Traceback" not in resp.text
