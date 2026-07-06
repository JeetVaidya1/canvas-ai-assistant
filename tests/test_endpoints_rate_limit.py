"""AI rate limiting at the HTTP layer, with the REAL ai_rate_limit dependency.

Only auth (current_user_id) is overridden; rate_limit._ai_limiter is swapped
for a tiny SlidingWindowLimiter so the window is exceeded in a few requests.
"""
import pytest

import auth
import main
import rate_limit
import routers.chat as chat_router
from fakes_endpoints import FakeSupabase
from rate_limit import SlidingWindowLimiter

pytestmark = pytest.mark.unit

ASK_FORM = {"question": "What is a heap?", "course_id": "cs101"}


@pytest.fixture
def tiny_limiter(monkeypatch):
    """Real ai_rate_limit dependency, 2-requests-per-minute limiter."""
    limiter = SlidingWindowLimiter(max_requests=2, window_seconds=60.0)
    monkeypatch.setattr(rate_limit, "_ai_limiter", limiter)
    # Hermetic /ask happy path: stub engine + fake DB, basic QA mode.
    monkeypatch.setattr(chat_router, "CONVERSATIONAL_MODE", False)
    monkeypatch.setattr(chat_router, "ENHANCED_MODE", False)
    monkeypatch.setattr(chat_router, "ask_question", lambda _q, _cid: "ok")
    monkeypatch.setattr(chat_router, "supabase", FakeSupabase())
    # Course authz is covered by test_endpoints_chat; here we only exercise
    # the limiter, so grant access unconditionally.
    monkeypatch.setattr(chat_router, "require_course_access", lambda _cid, _user: None)
    return limiter


def _as(user_id):
    main.app.dependency_overrides[auth.current_user_id] = lambda: user_id
    main.app.dependency_overrides[auth.get_current_user] = lambda: {
        "id": user_id,
        "email": f"{user_id}@test.dev",
    }


def test_requests_within_limit_succeed(client, tiny_limiter):
    _as("rl-user-a")
    assert client.post("/ask", data=ASK_FORM).status_code == 200
    assert client.post("/ask", data=ASK_FORM).status_code == 200


def test_exceeding_window_returns_429_with_retry_after(client, tiny_limiter):
    _as("rl-user-b")
    client.post("/ask", data=ASK_FORM)
    client.post("/ask", data=ASK_FORM)
    resp = client.post("/ask", data=ASK_FORM)
    assert resp.status_code == 429
    retry_after = resp.headers.get("Retry-After")
    assert retry_after is not None and int(retry_after) >= 1
    body = resp.json()
    assert set(body.keys()) == {"detail"}
    assert body["detail"].startswith("Rate limit exceeded")


def test_limit_is_per_user_not_global(client, tiny_limiter):
    _as("rl-user-c")
    client.post("/ask", data=ASK_FORM)
    client.post("/ask", data=ASK_FORM)
    assert client.post("/ask", data=ASK_FORM).status_code == 429
    # A different authenticated user still has full capacity.
    _as("rl-user-d")
    assert client.post("/ask", data=ASK_FORM).status_code == 200


def test_429_applies_before_any_work_happens(client, tiny_limiter, monkeypatch):
    """Once limited, the QA engine must not be invoked at all."""
    calls = []
    monkeypatch.setattr(
        chat_router, "ask_question", lambda q, cid: calls.append(q) or "ok"
    )
    _as("rl-user-e")
    client.post("/ask", data=ASK_FORM)
    client.post("/ask", data=ASK_FORM)
    assert len(calls) == 2
    assert client.post("/ask", data=ASK_FORM).status_code == 429
    assert len(calls) == 2  # engine untouched on the limited request
