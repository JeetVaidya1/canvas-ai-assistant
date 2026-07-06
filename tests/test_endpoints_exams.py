"""Exam-session lifecycle at the HTTP layer with a mocked exam engine.

routers.exams' module-level ``exam_session_manager`` reference is swapped for
a scripted fake, so the full create -> start -> get -> answer -> submit flow
runs with zero network/DB access.
"""
import json

import pytest

import routers.exams as exams_router

pytestmark = pytest.mark.unit

USER = "u-exam"
OTHER = "u-intruder"
EXAM = {"name": "Midterm Practice", "time_limit": 60, "questions": [{"id": "q1"}]}


class FakeExamManager:
    """In-memory stand-in for deps.exam_session_manager."""

    def __init__(self):
        self.sessions = {}
        self.created_with = []
        self.history_calls = []

    def create_exam_session(self, user_id, course_id, exam_data):
        self.created_with.append((user_id, course_id))
        session = {
            "id": f"sess-{len(self.sessions) + 1}",
            "user_id": user_id,
            "course_id": course_id,
            "exam_data": exam_data,
            "status": "created",
            "user_answers": {},
        }
        self.sessions[session["id"]] = session
        return {"status": "success", "session": dict(session)}

    def start_exam_session(self, session_id):
        session = self.sessions.get(session_id)
        if session is None:
            return {"status": "error", "message": "Session not found"}
        if session["status"] != "created":
            return {"status": "error", "message": "Session already started or completed"}
        self.sessions[session_id] = {**session, "status": "active"}
        return {"status": "success", "session": dict(self.sessions[session_id])}

    def get_session(self, session_id):
        session = self.sessions.get(session_id)
        if session is None:
            return {"status": "error", "message": "Session not found"}
        return {"status": "success", "session": dict(session)}

    def save_answer(self, session_id, question_id, answer):
        session = self.sessions.get(session_id)
        if session is None or session["status"] != "active":
            return {"status": "error", "message": "Session not active"}
        answers = {**session["user_answers"], question_id: answer}
        self.sessions[session_id] = {**session, "user_answers": answers}
        return {"status": "success", "answers_saved": len(answers)}

    def submit_exam(self, session_id):
        session = self.sessions.get(session_id)
        if session is None or session["status"] != "active":
            return {"status": "error", "message": "Session not active"}
        self.sessions[session_id] = {**session, "status": "completed"}
        return {"status": "success", "final_score": {"percentage": 80.0, "letter_grade": "B"}}

    def pause_exam_session(self, session_id):
        session = self.sessions.get(session_id)
        if session is None or session["status"] != "active":
            return {"status": "error", "message": "Session not active"}
        paused = not session.get("is_paused", False)
        self.sessions[session_id] = {**session, "is_paused": paused}
        return {"status": "success", "session": dict(self.sessions[session_id])}

    def navigate_to_question(self, session_id, question_index):
        session = self.sessions.get(session_id)
        if session is None:
            return {"status": "error", "message": "Session not found"}
        if question_index < 0 or question_index >= len(session["exam_data"]["questions"]):
            return {"status": "error", "message": "Invalid question index"}
        self.sessions[session_id] = {**session, "current_question": question_index}
        return {"status": "success", "current_question": question_index}

    def delete_session(self, session_id):
        return self.sessions.pop(session_id, None) is not None

    def get_user_exam_history(self, user_id, course_id=None):
        self.history_calls.append((user_id, course_id))
        return [dict(s) for s in self.sessions.values()
                if s["user_id"] == user_id
                and (course_id is None or s["course_id"] == course_id)]

    def auto_submit_expired_exams(self):
        return 0


@pytest.fixture
def exam_manager(monkeypatch):
    fake = FakeExamManager()
    monkeypatch.setattr(exams_router, "exam_session_manager", fake)
    return fake


def _create_session(client):
    return client.post(
        "/api/create-exam-session",
        data={"exam_data": json.dumps(EXAM), "course_id": "cs101"},
    )


def test_create_exam_session_requires_auth(client, exam_manager):
    resp = _create_session(client)
    assert resp.status_code == 401


def test_create_exam_session_uses_token_user(client, as_user, exam_manager):
    as_user(USER)
    resp = _create_session(client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["session"]["user_id"] == USER
    assert exam_manager.created_with == [(USER, "cs101")]


def test_create_exam_session_rejects_bad_json(client, as_user, exam_manager):
    as_user(USER)
    resp = client.post(
        "/api/create-exam-session",
        data={"exam_data": "not-json{", "course_id": "cs101"},
    )
    assert resp.status_code == 400
    assert resp.json() == {"detail": "Invalid exam data format"}


def test_full_exam_lifecycle_happy_path(client, as_user, exam_manager):
    as_user(USER)
    session_id = _create_session(client).json()["session"]["id"]

    start = client.post(f"/api/start-exam-session/{session_id}")
    assert start.status_code == 200
    assert start.json()["session"]["status"] == "active"

    state = client.get(f"/api/exam-session/{session_id}")
    assert state.status_code == 200
    assert state.json()["session"]["status"] == "active"

    answer = client.post(
        "/api/save-exam-answer",
        data={"session_id": session_id, "question_id": "q1", "answer": "42"},
    )
    assert answer.status_code == 200
    assert answer.json()["answers_saved"] == 1

    submit = client.post(f"/api/submit-exam/{session_id}")
    assert submit.status_code == 200
    assert submit.json()["final_score"]["percentage"] == 80.0
    assert exam_manager.sessions[session_id]["status"] == "completed"


def test_start_unknown_session_is_404(client, as_user, exam_manager):
    """Unknown session ids 404 — identical to foreign ones (non-enumeration)."""
    as_user(USER)
    resp = client.post("/api/start-exam-session/nope")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Session not found"}


def test_get_unknown_session_is_404(client, as_user, exam_manager):
    as_user(USER)
    resp = client.get("/api/exam-session/nope")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Session not found"}


def test_double_start_is_rejected(client, as_user, exam_manager):
    as_user(USER)
    session_id = _create_session(client).json()["session"]["id"]
    assert client.post(f"/api/start-exam-session/{session_id}").status_code == 200
    resp = client.post(f"/api/start-exam-session/{session_id}")
    assert resp.status_code == 400
    assert resp.json() == {"detail": "Session already started or completed"}


def test_submit_before_start_is_400(client, as_user, exam_manager):
    as_user(USER)
    session_id = _create_session(client).json()["session"]["id"]
    resp = client.post(f"/api/submit-exam/{session_id}")
    assert resp.status_code == 400
    assert resp.json() == {"detail": "Session not active"}


# ---- session ownership: every lifecycle endpoint is bound to its creator ----

LIFECYCLE_CALLS = [
    ("post", "/api/start-exam-session/{sid}", None),
    ("post", "/api/pause-exam-session/{sid}", None),
    ("get", "/api/exam-session/{sid}", None),
    ("post", "/api/submit-exam/{sid}", None),
    ("delete", "/api/exam-session/{sid}", None),
    ("post", "/api/save-exam-answer",
     {"session_id": "{sid}", "question_id": "q1", "answer": "42"}),
    ("post", "/api/navigate-exam-question",
     {"session_id": "{sid}", "question_index": "0"}),
]


def _call(client, method, path, data, sid):
    url = path.format(sid=sid)
    if data is None:
        return getattr(client, method)(url)
    form = {k: v.format(sid=sid) for k, v in data.items()}
    return getattr(client, method)(url, data=form)


@pytest.mark.parametrize("method,path,data", LIFECYCLE_CALLS)
def test_lifecycle_endpoints_require_auth(client, exam_manager, method, path, data):
    exam_manager.create_exam_session(USER, "cs101", EXAM)
    resp = _call(client, method, path, data, "sess-1")
    assert resp.status_code == 401


@pytest.mark.parametrize("method,path,data", LIFECYCLE_CALLS)
def test_foreign_session_is_404_and_untouched(client, as_user, exam_manager,
                                              method, path, data):
    """A stranger acting on someone else's session gets the not-found 404 and
    the session state is never modified."""
    victim_sid = exam_manager.create_exam_session(OTHER, "cs101", EXAM)["session"]["id"]
    exam_manager.start_exam_session(victim_sid)  # active: writes WOULD succeed
    as_user(USER)
    resp = _call(client, method, path, data, victim_sid)
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Session not found"}
    victim = exam_manager.sessions[victim_sid]  # still there, still untouched
    assert victim["status"] == "active"
    assert victim["user_answers"] == {}
    assert victim.get("current_question", 0) == 0
    assert victim.get("is_paused", False) is False


# ---- GET /api/exam-history/{user_id}: identity comes from the token --------

def test_exam_history_requires_auth(client, exam_manager):
    resp = client.get(f"/api/exam-history/{USER}")
    assert resp.status_code == 401


def test_exam_history_ignores_spoofed_path_user_id(client, as_user, exam_manager):
    """Putting a victim's id in the URL must return the TOKEN user's history."""
    exam_manager.create_exam_session(OTHER, "cs101", EXAM)  # victim's exam
    as_user(USER)
    resp = client.get(f"/api/exam-history/{OTHER}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["exams"] == []  # token user has no exams; victim's stay hidden
    assert exam_manager.history_calls == [(USER, None)]


def test_exam_history_keeps_course_filter_for_token_user(client, as_user, exam_manager):
    exam_manager.create_exam_session(USER, "cs101", EXAM)
    as_user(USER)
    resp = client.get(f"/api/exam-history/{USER}", params={"course_id": "cs101"})
    assert resp.status_code == 200
    assert resp.json()["total_exams"] == 1
    assert exam_manager.history_calls == [(USER, "cs101")]


# ---- /api/admin/auto-submit-expired-exams: debug-gated ----------------------

def test_admin_auto_submit_is_hidden_404_by_default(client, exam_manager, monkeypatch):
    monkeypatch.delenv("ENABLE_DEBUG_ENDPOINTS", raising=False)
    resp = client.get("/api/admin/auto-submit-expired-exams")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Not Found"}


def test_admin_auto_submit_runs_when_debug_enabled(client, exam_manager, monkeypatch):
    monkeypatch.setenv("ENABLE_DEBUG_ENDPOINTS", "1")
    resp = client.get("/api/admin/auto-submit-expired-exams")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
