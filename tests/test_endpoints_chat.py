"""Chat endpoints: /sessions listing is bound to the token user, /ask validates
input, enforces course access, and persists the Q/A exchange; per-session
routes enforce ownership (foreign/unknown sessions 404 identically, so ids
can't be enumerated).

Hermetic: routers.chat's supabase reference is a FakeSupabase; auth._db is a
FakeSupabase so require_course_access runs for real; the QA engines are forced
onto the basic path with a stubbed ask_question.
"""
import pytest

import auth
import routers.chat as chat_router
from fakes_endpoints import LEAK_MARKER, FakeSupabase

pytestmark = pytest.mark.unit

USER = "u-chat"
OTHER = "u-other"
COURSE = "cs101"
STUB_ANSWER = "stubbed answer from course materials"


@pytest.fixture
def chat_env(monkeypatch):
    """Force the basic QA path with a stub engine; returns a fake-db installer.

    Also installs a fake auth DB so require_course_access sees COURSE as owned
    by USER (override with ``courses=`` / ``memberships=``).
    """
    monkeypatch.setattr(chat_router, "CONVERSATIONAL_MODE", False)
    monkeypatch.setattr(chat_router, "ENHANCED_MODE", False)
    monkeypatch.setattr(chat_router, "ask_question", lambda _q, _cid: STUB_ANSWER)

    def _install(tables=None, fail=False, courses=None, memberships=None):
        db = FakeSupabase(tables or {"chat_sessions": [], "messages": []}, fail=fail)
        monkeypatch.setattr(chat_router, "supabase", db)
        monkeypatch.setattr(auth, "_db", FakeSupabase({
            "courses": courses if courses is not None
            else [{"course_id": COURSE, "owner_id": USER}],
            "course_memberships": memberships or [],
        }))
        return db

    return _install


# ---- GET /sessions ------------------------------------------------------

def test_sessions_returns_only_token_users_sessions(client, as_user, chat_env):
    as_user(USER)
    chat_env({"chat_sessions": [
        {"id": "s1", "user_id": USER, "title": "a"},
        {"id": "s2", "user_id": OTHER, "title": "b"},
        {"id": "s3", "user_id": USER, "title": "c"},
    ]})
    resp = client.get("/sessions")
    assert resp.status_code == 200
    sessions = resp.json()["sessions"]
    assert {s["id"] for s in sessions} == {"s1", "s3"}
    assert all(s["user_id"] == USER for s in sessions)


def test_sessions_ignores_spoofed_user_id_query_param(client, as_user, chat_env):
    """?user_id=<victim> must not switch identity — the token wins."""
    as_user(USER)
    chat_env({"chat_sessions": [
        {"id": "s1", "user_id": USER},
        {"id": "s2", "user_id": OTHER},
    ]})
    resp = client.get("/sessions", params={"user_id": OTHER})
    assert resp.status_code == 200
    assert [s["id"] for s in resp.json()["sessions"]] == ["s1"]


def test_sessions_db_failure_is_honest_500(client, as_user, chat_env):
    as_user(USER)
    chat_env(fail=True)
    resp = client.get("/sessions")
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Couldn't fetch sessions"}
    assert LEAK_MARKER not in resp.text


# ---- POST /ask -------------------------------------------------------------

@pytest.mark.parametrize("data", [
    {},
    {"question": "What is a heap?"},
    {"course_id": "cs101"},
])
def test_ask_missing_fields_is_422(client, as_user, chat_env, data):
    as_user(USER)
    chat_env()
    resp = client.post("/ask", data=data)
    assert resp.status_code == 422
    body = resp.json()
    assert "detail" in body and isinstance(body["detail"], list)


def test_ask_creates_session_and_persists_exchange(client, as_user, chat_env):
    as_user(USER)
    db = chat_env()
    resp = client.post("/ask", data={"question": "What is a heap?", "course_id": "cs101"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == STUB_ANSWER
    assert body["question"] == "What is a heap?"
    assert body["session_id"]

    [session] = db.tables["chat_sessions"]
    assert session["user_id"] == USER  # session belongs to the token user
    assert session["course_id"] == "cs101"

    roles = [m["role"] for m in db.tables["messages"]]
    assert roles == ["user", "assistant"]
    assert db.tables["messages"][1]["content"] == STUB_ANSWER


def test_ask_reuses_provided_session_id(client, as_user, chat_env):
    as_user(USER)
    db = chat_env({"chat_sessions": [{"id": "sess-9", "user_id": USER}], "messages": []})
    resp = client.post(
        "/ask",
        data={"question": "Follow-up?", "course_id": "cs101", "session_id": "sess-9"},
    )
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "sess-9"
    assert len(db.tables["chat_sessions"]) == 1  # no new session created


def test_ask_session_create_failure_is_honest_500(client, as_user, chat_env):
    as_user(USER)
    chat_env(fail=True)
    resp = client.post("/ask", data={"question": "Q?", "course_id": "cs101"})
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Couldn't create session"}
    assert LEAK_MARKER not in resp.text


def test_ask_fires_chat_tracking_in_background(client, as_user, chat_env, monkeypatch):
    """A successful /ask counts toward mastery/streak: track_interaction fires
    (on BackgroundTasks — TestClient runs it after the response is built)."""
    import deps

    calls = []

    class RecorderAnalytics:
        def track_interaction(self, **kwargs):
            calls.append(kwargs)
            return True

    monkeypatch.setattr(deps, "analytics_engine", RecorderAnalytics())
    as_user(USER)
    chat_env()
    resp = client.post("/ask", data={"question": "What is a heap?", "course_id": COURSE})
    assert resp.status_code == 200

    [tracked] = calls
    assert tracked["user_id"] == USER
    assert tracked["course_id"] == COURSE
    assert tracked["question"] == "What is a heap?"
    assert tracked["question_type"] == "chat"
    assert tracked["answer"] == STUB_ANSWER


def test_ask_tracking_failure_never_affects_the_response(client, as_user, chat_env, monkeypatch):
    import deps

    class ExplodingAnalytics:
        def track_interaction(self, **kwargs):
            raise RuntimeError("analytics down")

    monkeypatch.setattr(deps, "analytics_engine", ExplodingAnalytics())
    as_user(USER)
    chat_env()
    resp = client.post("/ask", data={"question": "Q?", "course_id": COURSE})
    assert resp.status_code == 200
    assert resp.json()["answer"] == STUB_ANSWER


def test_ask_foreign_course_is_403(client, as_user, chat_env):
    """/ask must not answer from a course the token user can't access."""
    as_user(USER)
    db = chat_env(courses=[{"course_id": "bio200", "owner_id": OTHER}])
    resp = client.post("/ask", data={"question": "Q?", "course_id": "bio200"})
    assert resp.status_code == 403
    assert resp.json() == {"detail": "You don't have access to this course"}
    assert db.tables["chat_sessions"] == []  # nothing persisted
    assert db.tables["messages"] == []


def test_ask_member_course_is_allowed(client, as_user, chat_env):
    as_user(USER)
    chat_env(
        courses=[{"course_id": "bio200", "owner_id": OTHER}],
        memberships=[{"id": 1, "course_id": "bio200", "user_id": USER}],
    )
    resp = client.post("/ask", data={"question": "Q?", "course_id": "bio200"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == STUB_ANSWER


def test_ask_foreign_session_id_is_404(client, as_user, chat_env):
    """Writing into someone else's chat session must 404 (non-enumeration)."""
    as_user(USER)
    db = chat_env({"chat_sessions": [{"id": "sess-x", "user_id": OTHER}], "messages": []})
    resp = client.post(
        "/ask",
        data={"question": "Q?", "course_id": COURSE, "session_id": "sess-x"},
    )
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Session not found"}
    assert db.tables["messages"] == []


def test_ask_stream_foreign_course_is_403(client, as_user, chat_env):
    as_user(USER)
    chat_env(courses=[{"course_id": "bio200", "owner_id": OTHER}])
    resp = client.post("/ask/stream", data={"question": "Q?", "course_id": "bio200"})
    assert resp.status_code == 403
    assert resp.json() == {"detail": "You don't have access to this course"}


def test_ask_stream_foreign_session_id_is_404(client, as_user, chat_env):
    as_user(USER)
    db = chat_env({"chat_sessions": [{"id": "sess-x", "user_id": OTHER}], "messages": []})
    resp = client.post(
        "/ask/stream",
        data={"question": "Q?", "course_id": COURSE, "session_id": "sess-x"},
    )
    assert resp.status_code == 404
    assert db.tables["messages"] == []


# ---- session messages / delete ------------------------------------------
# Ownership rule: foreign sessions and unknown sessions both 404 identically
# (non-enumeration — a stranger can't learn that a session id exists).

def test_get_messages_returns_owned_session_messages(client, as_user, chat_env):
    as_user(USER)
    chat_env({
        "chat_sessions": [{"id": "s1", "user_id": USER}],
        "messages": [
            {"id": "m1", "session_id": "s1", "role": "user", "content": "hi"},
            {"id": "m2", "session_id": "s1", "role": "assistant", "content": "hello"},
            {"id": "m3", "session_id": "s2", "role": "user", "content": "other"},
        ],
    })
    resp = client.get("/sessions/s1/messages")
    assert resp.status_code == 200
    assert [m["id"] for m in resp.json()["messages"]] == ["m1", "m2"]


def test_get_messages_requires_auth(client, chat_env):
    chat_env({"chat_sessions": [{"id": "s1", "user_id": USER}], "messages": []})
    resp = client.get("/sessions/s1/messages")
    assert resp.status_code == 401


def test_get_messages_foreign_session_is_404(client, as_user, chat_env):
    as_user(USER)
    chat_env({
        "chat_sessions": [{"id": "s2", "user_id": OTHER}],
        "messages": [{"id": "m1", "session_id": "s2", "role": "user", "content": "secret"}],
    })
    resp = client.get("/sessions/s2/messages")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Session not found"}
    assert "secret" not in resp.text


def test_get_messages_unknown_session_is_same_404(client, as_user, chat_env):
    as_user(USER)
    chat_env()
    resp = client.get("/sessions/does-not-exist/messages")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Session not found"}


def test_delete_session_removes_owned_session_and_messages(client, as_user, chat_env):
    as_user(USER)
    db = chat_env({
        "chat_sessions": [{"id": "s1", "user_id": USER}],
        "messages": [{"id": "m1", "session_id": "s1", "role": "user", "content": "x"}],
    })
    resp = client.delete("/sessions/s1")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert db.tables["chat_sessions"] == []
    assert db.tables["messages"] == []


def test_delete_session_requires_auth(client, chat_env):
    db = chat_env({"chat_sessions": [{"id": "s1", "user_id": USER}], "messages": []})
    resp = client.delete("/sessions/s1")
    assert resp.status_code == 401
    assert len(db.tables["chat_sessions"]) == 1  # untouched


def test_delete_foreign_session_is_404_and_untouched(client, as_user, chat_env):
    as_user(USER)
    db = chat_env({
        "chat_sessions": [{"id": "s2", "user_id": OTHER}],
        "messages": [{"id": "m1", "session_id": "s2", "role": "user", "content": "x"}],
    })
    resp = client.delete("/sessions/s2")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Session not found"}
    assert len(db.tables["chat_sessions"]) == 1
    assert len(db.tables["messages"]) == 1
