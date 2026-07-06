"""Quiz runner endpoints — V3 fast-start drills + confidence calibration.

Covers:
- POST /quiz/generate: two-phase generation (instant first batch + background
  remainder via BackgroundTasks), user_id stamped from the token, dedupe
  prompting, large-quiz batch splitting, and 'partial' on background failure.
- GET /quiz/{id}/questions: auth, sanitized shape, ownership (404 identical for
  foreign and unknown ids), legacy NULL-user sessions admitted.
- POST /quiz/{id}/answer: confidence stored, invalid confidence rejected, and
  wrong answers graded WITHOUT inline mistake-engine work (deferred to
  BackgroundTasks).
- POST /quiz/{id}/submit: calibration read-out math.

Hermetic: quiz_engine's supabase is a FakeSupabase; retrieve/structured_call
are stubs; deps.analytics_engine, mistake_engine and review_engine are
recorders. TestClient runs BackgroundTasks synchronously AFTER the response is
serialized, so tests can observe both the instant payload and the final DB.
"""
import re

import pytest

import deps
import mistake_engine
import quiz_engine
import review_engine
import routers.quiz as quiz_router
from fakes_endpoints import FakeSupabase

pytestmark = pytest.mark.unit

USER = "u-quiz"
OTHER = "u-other"
COURSE = "cs101"


def _raw_question(i: int) -> dict:
    return {
        "question": f"Question {i}?",
        "options": [f"A) opt-a-{i}", f"B) opt-b-{i}", f"C) opt-c-{i}", f"D) opt-d-{i}"],
        "correct_answer": "B",
        "explanation": f"stored explanation {i}",
        "concept": f"concept-{i % 3}",
        "source_doc": "doc.pdf",
        "source_page": i,
    }


class Recorder:
    """Callable stub that records (args, kwargs) and returns a fixed value."""

    def __init__(self, result=None, raises: Exception | None = None):
        self.calls: list[tuple] = []
        self.result = result
        self.raises = raises

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.raises:
            raise self.raises
        return self.result


class GenerationStub:
    """structured_call stand-in: mints uniquely-numbered questions per call.

    Parses the requested count out of the prompt ("Create N ..."), so phase-1
    and background batches produce the sizes the engine asked for. Set
    ``fail_after`` to make later calls raise (background-failure tests).
    """

    def __init__(self, fail_after: int | None = None):
        self.prompts: list[str] = []
        self.next_number = 1
        self.fail_after = fail_after

    def __call__(self, messages, schema=None, tool_name=None, model=None, max_tokens=None):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        if self.fail_after is not None and len(self.prompts) > self.fail_after:
            raise RuntimeError("LLM exploded mid-background")
        count = int(re.search(r"Create (\d+) ", prompt).group(1))
        questions = [_raw_question(self.next_number + j) for j in range(count)]
        self.next_number += count
        return {"questions": questions}


@pytest.fixture
def quiz_env(monkeypatch):
    """Install all fakes; returns (db, generation_stub, recorders_dict)."""

    async def _valid_course(_course_id):
        return {"status": "valid", "error": None}

    def _install(fail_after: int | None = None, tables: dict | None = None):
        db = FakeSupabase(tables or {
            "quiz_sessions": [], "quiz_questions": [], "quiz_responses": [],
        })
        gen = GenerationStub(fail_after=fail_after)
        recorders = {
            "retrieve": Recorder(result=[{"content": "chunk", "doc_name": "doc.pdf", "page": 1}]),
            "track_quiz_answer": Recorder(result=True),
            "explain_mistake": Recorder(result={
                "explanation": "grounded mistake explanation",
                "source": {"doc_name": "doc.pdf", "page": 2},
            }),
            "seed_from_mistake": Recorder(result="review-item-1"),
        }
        monkeypatch.setattr(quiz_engine, "_supabase", db)
        monkeypatch.setattr(quiz_engine, "retrieve", recorders["retrieve"])
        monkeypatch.setattr(quiz_engine, "structured_call", gen)
        monkeypatch.setattr(quiz_router, "validate_course_for_practice", _valid_course)

        class _Analytics:
            track_quiz_answer = staticmethod(recorders["track_quiz_answer"])

        monkeypatch.setattr(deps, "analytics_engine", _Analytics())
        monkeypatch.setattr(mistake_engine, "explain_mistake", recorders["explain_mistake"])
        monkeypatch.setattr(review_engine, "seed_from_mistake", recorders["seed_from_mistake"])
        return db, gen, recorders

    return _install


def _seed_session(db: FakeSupabase, quiz_id: str = "quiz-1", user_id: str | None = USER,
                  num_questions: int = 2, generation_status: str = "ready",
                  status: str = "active", course_id: str = COURSE,
                  created_at: str = "2026-07-01T00:00:00") -> None:
    db.tables["quiz_sessions"].append({
        "id": quiz_id, "course_id": course_id, "user_id": user_id, "topic": "hashing",
        "difficulty": "medium", "num_questions": num_questions,
        "num_requested": num_questions, "generation_status": generation_status,
        "status": status, "created_at": created_at,
    })
    for i in range(1, num_questions + 1):
        raw = _raw_question(i)
        db.tables["quiz_questions"].append({
            "quiz_id": quiz_id, "question_id": f"q{i}", "question": raw["question"],
            "options": raw["options"], "correct_answer": raw["correct_answer"],
            "explanation": raw["explanation"], "concept": raw["concept"],
            "difficulty": "medium", "source_doc": raw["source_doc"],
            "source_page": raw["source_page"],
        })


# ---- POST /quiz/generate: two-phase fast start -----------------------------

def test_generate_returns_first_batch_instantly_and_finishes_in_background(
        client, as_user, quiz_env):
    as_user(USER)
    db, gen, recorders = quiz_env()

    resp = client.post("/quiz/generate", data={
        "course_id": COURSE, "topic": "hashing",
        "num_questions": 10, "difficulty": "medium",
    })
    assert resp.status_code == 200
    body = resp.json()

    # Instant payload: only the first 3 questions, honestly marked 'generating'.
    assert body["generation_status"] == "generating"
    assert body["num_requested"] == 10
    assert body["num_questions"] == 3
    assert len(body["questions"]) == 3
    assert "_background" not in body  # internal plumbing never leaks
    for q in body["questions"]:
        assert "correct_answer" not in q and "explanation" not in q
        assert set(q) == {"id", "question", "options", "concept", "difficulty", "source"}

    # Background task (run synchronously by TestClient) appended the rest.
    rows = db.tables["quiz_questions"]
    assert len(rows) == 10
    assert [r["question_id"] for r in rows] == [f"q{i}" for i in range(1, 11)]

    [session] = db.tables["quiz_sessions"]
    assert session["user_id"] == USER            # NULL-user_id bug is fixed
    assert session["generation_status"] == "ready"
    assert session["num_questions"] == 10
    assert session["num_requested"] == 10

    # One retrieval reused across phases; 1 sync + 1 background LLM call (<15).
    assert len(recorders["retrieve"].calls) == 1
    assert len(gen.prompts) == 2
    # The background prompt lists phase-1 stems so it can't duplicate them.
    assert "Question 1?" in gen.prompts[1]
    assert "Question 3?" in gen.prompts[1]


def test_generate_small_quiz_is_single_phase_ready(client, as_user, quiz_env):
    as_user(USER)
    db, gen, _ = quiz_env()
    resp = client.post("/quiz/generate", data={"course_id": COURSE, "num_questions": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["generation_status"] == "ready"
    assert body["num_requested"] == 3
    assert len(body["questions"]) == 3
    assert len(gen.prompts) == 1                  # no background call at all
    assert len(db.tables["quiz_questions"]) == 3
    assert db.tables["quiz_sessions"][0]["generation_status"] == "ready"


def test_generate_large_quiz_splits_background_into_two_batches(client, as_user, quiz_env):
    as_user(USER)
    db, gen, _ = quiz_env()
    resp = client.post("/quiz/generate", data={"course_id": COURSE, "num_questions": 20})
    assert resp.status_code == 200
    assert resp.json()["num_requested"] == 20
    assert len(gen.prompts) == 3                  # 1 sync + 2 background batches
    assert len(db.tables["quiz_questions"]) == 20
    assert db.tables["quiz_sessions"][0]["generation_status"] == "ready"
    # Second background batch must also be told about first-batch stems.
    assert "Question 4?" in gen.prompts[2]


def test_generate_background_failure_marks_partial_but_session_playable(
        client, as_user, quiz_env):
    as_user(USER)
    db, gen, _ = quiz_env(fail_after=1)           # phase 1 ok, background dies
    resp = client.post("/quiz/generate", data={"course_id": COURSE, "num_questions": 10})
    assert resp.status_code == 200                # user got their instant start
    assert len(resp.json()["questions"]) == 3
    [session] = db.tables["quiz_sessions"]
    assert session["generation_status"] == "partial"
    assert len(db.tables["quiz_questions"]) == 3  # playable with what exists


def test_generate_requires_auth(client, quiz_env):
    quiz_env()
    resp = client.post("/quiz/generate", data={"course_id": COURSE, "num_questions": 5})
    assert resp.status_code == 401


# ---- GET /quiz/{quiz_id}/questions: polling endpoint ------------------------

def test_questions_poll_returns_sanitized_questions_and_progress(client, as_user, quiz_env):
    as_user(USER)
    db, _, _ = quiz_env()
    _seed_session(db, "quiz-1", USER, num_questions=2, generation_status="generating")

    resp = client.get("/quiz/quiz-1/questions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["quiz_id"] == "quiz-1"
    assert body["generation_status"] == "generating"
    assert body["num_requested"] == 2
    assert body["num_questions"] == 2
    assert [q["id"] for q in body["questions"]] == ["q1", "q2"]
    for q in body["questions"]:
        assert "correct_answer" not in q and "explanation" not in q


def test_questions_poll_requires_auth(client, quiz_env):
    db, _, _ = quiz_env()
    _seed_session(db, "quiz-1", USER)
    assert client.get("/quiz/quiz-1/questions").status_code == 401


def test_questions_poll_foreign_and_unknown_quiz_404_identically(client, as_user, quiz_env):
    as_user(USER)
    db, _, _ = quiz_env()
    _seed_session(db, "quiz-foreign", OTHER)
    foreign = client.get("/quiz/quiz-foreign/questions")
    unknown = client.get("/quiz/does-not-exist/questions")
    assert foreign.status_code == unknown.status_code == 404
    assert foreign.json() == unknown.json() == {"detail": "Quiz not found"}


def test_questions_poll_admits_legacy_null_user_sessions(client, as_user, quiz_env):
    as_user(USER)
    db, _, _ = quiz_env()
    _seed_session(db, "quiz-legacy", user_id=None)
    assert client.get("/quiz/quiz-legacy/questions").status_code == 200


# ---- POST /quiz/{quiz_id}/answer: instant grading + deferred mistake work ---

def test_correct_answer_stores_confidence_and_skips_mistake_work(client, as_user, quiz_env):
    as_user(USER)
    db, _, recorders = quiz_env()
    _seed_session(db)

    resp = client.post("/quiz/quiz-1/answer", data={
        "question_id": "q1", "selected": "B", "confidence": "sure",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_correct"] is True
    assert body["explanation"] == "stored explanation 1"

    [response_row] = db.tables["quiz_responses"]
    assert response_row["confidence"] == "sure"
    assert response_row["is_correct"] is True
    assert response_row["user_id"] == USER
    assert len(recorders["track_quiz_answer"].calls) == 1
    assert recorders["explain_mistake"].calls == []
    assert recorders["seed_from_mistake"].calls == []


def test_confidence_is_optional_and_stored_null(client, as_user, quiz_env):
    as_user(USER)
    db, _, _ = quiz_env()
    _seed_session(db)
    resp = client.post("/quiz/quiz-1/answer", data={"question_id": "q1", "selected": "B"})
    assert resp.status_code == 200
    assert db.tables["quiz_responses"][0]["confidence"] is None


def test_invalid_confidence_is_rejected(client, as_user, quiz_env):
    as_user(USER)
    db, _, _ = quiz_env()
    _seed_session(db)
    resp = client.post("/quiz/quiz-1/answer", data={
        "question_id": "q1", "selected": "B", "confidence": "yolo",
    })
    assert resp.status_code == 400
    assert "confidence" in resp.json()["detail"]
    assert db.tables["quiz_responses"] == []      # nothing persisted


def test_wrong_answer_returns_stored_explanation_and_defers_mistake_work(
        client, as_user, quiz_env):
    as_user(USER)
    db, _, recorders = quiz_env()
    _seed_session(db)

    resp = client.post("/quiz/quiz-1/answer", data={
        "question_id": "q1", "selected": "A", "confidence": "guessing",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_correct"] is False
    assert body["correct_answer"] == "B"
    # Instant feedback comes from the STORED explanation; the grounded mistake
    # explanation is produced later in the background, not in this payload.
    assert body["explanation"] == "stored explanation 1"
    assert body["mistake_explanation"] == ""

    # The background task DID run (TestClient runs it post-response): the
    # mistake was explained and a review item seeded with the grounded text.
    assert len(recorders["explain_mistake"].calls) == 1
    assert len(recorders["seed_from_mistake"].calls) == 1
    _, seed_kwargs = recorders["seed_from_mistake"].calls[0]
    assert seed_kwargs["user_id"] == USER
    assert seed_kwargs["explanation"] == "grounded mistake explanation"
    assert seed_kwargs["answer"] == "B) opt-b-1"  # full text of the correct option


def test_grade_answer_engine_never_calls_mistake_engine_inline(quiz_env):
    """Structural proof of deferral: grade_answer itself does zero mistake work."""
    db, _, recorders = quiz_env()
    _seed_session(db)
    result = quiz_engine.grade_answer("quiz-1", "q1", "A", 2.0, USER, confidence="sure")
    assert result["is_correct"] is False
    assert recorders["explain_mistake"].calls == []
    assert recorders["seed_from_mistake"].calls == []
    # The deferred half is a separate function the router backgrounds.
    quiz_engine.followup_wrong_answer("quiz-1", "q1", "A", USER)
    assert len(recorders["explain_mistake"].calls) == 1
    assert len(recorders["seed_from_mistake"].calls) == 1


def test_answer_unknown_question_is_404(client, as_user, quiz_env):
    as_user(USER)
    db, _, _ = quiz_env()
    _seed_session(db)
    resp = client.post("/quiz/quiz-1/answer", data={"question_id": "q99", "selected": "A"})
    assert resp.status_code == 404


# ---- POST /quiz/{quiz_id}/submit: calibration math --------------------------

def test_submit_reports_calibration_buckets_and_confident_wrong(client, as_user, quiz_env):
    as_user(USER)
    db, _, _ = quiz_env()
    _seed_session(db, num_questions=4)

    answers = [
        ("q1", "B", "sure"),      # sure + correct
        ("q2", "A", "sure"),      # sure + WRONG -> confident_wrong
        ("q3", "B", "thinkso"),   # thinkso + correct
        ("q4", "C", "guessing"),  # guessing + wrong
    ]
    for qid, selected, conf in answers:
        r = client.post("/quiz/quiz-1/answer", data={
            "question_id": qid, "selected": selected, "confidence": conf,
        })
        assert r.status_code == 200

    resp = client.post("/quiz/quiz-1/submit")
    assert resp.status_code == 200
    body = resp.json()
    assert body["score"] == {"correct": 2, "total": 4, "pct": 50.0}
    assert body["calibration"] == {
        "sure": {"n": 2, "correct": 1},
        "thinkso": {"n": 1, "correct": 1},
        "guessing": {"n": 1, "correct": 0},
        "confident_wrong": 1,
    }
    [session] = db.tables["quiz_sessions"]
    assert session["status"] == "completed"


# ---- GET /quiz/in-progress + GET /quiz/{id}/responses: resume everywhere ----

def test_in_progress_requires_auth(client, quiz_env):
    quiz_env()
    assert client.get("/quiz/in-progress", params={"course_id": COURSE}).status_code == 401


def test_responses_requires_auth(client, quiz_env):
    db, _, _ = quiz_env()
    _seed_session(db, "quiz-1", USER)
    assert client.get("/quiz/quiz-1/responses").status_code == 401


def test_in_progress_requires_course_id(client, as_user, quiz_env):
    as_user(USER)
    quiz_env()
    missing = client.get("/quiz/in-progress")
    empty = client.get("/quiz/in-progress", params={"course_id": "  "})
    assert missing.status_code == empty.status_code == 400


def test_in_progress_lists_only_own_open_sessions_in_course(client, as_user, quiz_env):
    as_user(USER)
    db, _, _ = quiz_env()
    _seed_session(db, "quiz-mine", USER)
    _seed_session(db, "quiz-foreign", OTHER)                       # other user
    _seed_session(db, "quiz-elsewhere", USER, course_id="cs999")   # other course

    resp = client.get("/quiz/in-progress", params={"course_id": COURSE})
    assert resp.status_code == 200
    [session] = resp.json()["sessions"]
    assert session == {
        "quiz_id": "quiz-mine",
        "topic": "hashing",
        "difficulty": "medium",
        "num_requested": 2,
        "num_answered": 0,
        "num_available": 2,
        "generation_status": "ready",
        "created_at": "2026-07-01T00:00:00",
    }


def test_in_progress_excludes_completed_and_questionless_sessions(client, as_user, quiz_env):
    as_user(USER)
    db, _, _ = quiz_env()
    _seed_session(db, "quiz-open", USER)
    _seed_session(db, "quiz-done", USER, status="completed")
    _seed_session(db, "quiz-empty", USER, num_questions=0)  # zero questions

    body = client.get("/quiz/in-progress", params={"course_id": COURSE}).json()
    assert [s["quiz_id"] for s in body["sessions"]] == ["quiz-open"]


def test_in_progress_counts_distinct_answered_questions(client, as_user, quiz_env):
    as_user(USER)
    db, _, _ = quiz_env()
    _seed_session(db, "quiz-1", USER, num_questions=4)
    # q1 answered twice (re-answer) + q2 once -> 2 distinct.
    for qid, selected in [("q1", "A"), ("q1", "B"), ("q2", "B")]:
        r = client.post("/quiz/quiz-1/answer", data={"question_id": qid, "selected": selected})
        assert r.status_code == 200

    [session] = client.get("/quiz/in-progress", params={"course_id": COURSE}).json()["sessions"]
    assert session["num_answered"] == 2
    assert session["num_available"] == 4


def test_in_progress_is_newest_first_capped_at_three(client, as_user, quiz_env):
    as_user(USER)
    db, _, _ = quiz_env()
    for i in range(1, 6):  # 5 open sessions, quiz-5 newest
        _seed_session(db, f"quiz-{i}", USER, created_at=f"2026-07-0{i}T00:00:00")

    body = client.get("/quiz/in-progress", params={"course_id": COURSE}).json()
    assert [s["quiz_id"] for s in body["sessions"]] == ["quiz-5", "quiz-4", "quiz-3"]


def test_responses_returns_latest_answer_per_question_in_order(client, as_user, quiz_env):
    as_user(USER)
    db, _, _ = quiz_env()
    _seed_session(db, "quiz-1", USER, num_questions=3)
    # Answer q2 first, then q1 wrong, then q1 corrected -> latest wins, sorted q1,q2.
    for qid, selected, conf in [("q2", "C", None), ("q1", "A", "guessing"), ("q1", "B", "sure")]:
        data = {"question_id": qid, "selected": selected}
        if conf:
            data["confidence"] = conf
        assert client.post("/quiz/quiz-1/answer", data=data).status_code == 200

    resp = client.get("/quiz/quiz-1/responses")
    assert resp.status_code == 200
    assert resp.json() == {
        "quiz_id": "quiz-1",
        "responses": [
            {"question_id": "q1", "selected": "B", "is_correct": True, "confidence": "sure"},
            {"question_id": "q2", "selected": "C", "is_correct": False, "confidence": None},
        ],
    }


def test_responses_empty_when_nothing_answered(client, as_user, quiz_env):
    as_user(USER)
    db, _, _ = quiz_env()
    _seed_session(db, "quiz-1", USER)
    assert client.get("/quiz/quiz-1/responses").json() == {"quiz_id": "quiz-1", "responses": []}


def test_responses_foreign_and_unknown_quiz_404_identically(client, as_user, quiz_env):
    as_user(USER)
    db, _, _ = quiz_env()
    _seed_session(db, "quiz-foreign", OTHER)
    foreign = client.get("/quiz/quiz-foreign/responses")
    unknown = client.get("/quiz/does-not-exist/responses")
    assert foreign.status_code == unknown.status_code == 404
    assert foreign.json() == unknown.json() == {"detail": "Quiz not found"}


def test_submit_without_confidence_yields_empty_calibration(client, as_user, quiz_env):
    as_user(USER)
    db, _, _ = quiz_env()
    _seed_session(db, num_questions=1)
    client.post("/quiz/quiz-1/answer", data={"question_id": "q1", "selected": "B"})
    body = client.post("/quiz/quiz-1/submit").json()
    assert body["calibration"] == {
        "sure": {"n": 0, "correct": 0},
        "thinkso": {"n": 0, "correct": 0},
        "guessing": {"n": 0, "correct": 0},
        "confident_wrong": 0,
    }
