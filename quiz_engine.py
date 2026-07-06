"""Quiz engine — generate grounded MCQs, grade answers, score sessions.

Replaces the brittle ``quiz_assistant_engine`` heuristics for the quiz *runner*
flow (generate N questions -> answer one at a time -> instant feedback -> score).
Built on the shared Phase-3 foundations:

  - ``rag.retrieval.retrieve``  for hybrid + reranked grounding
  - ``providers.structured_call``  for schema-guaranteed question generation
  - ``learning_analytics``  for per-concept mastery tracking

V3 fast-start (workstream B): :func:`generate_quiz` is two-phase. Phase 1 makes
ONE retrieval + ONE small LLM call for the first few questions and returns
immediately; the router schedules :func:`generate_remaining_questions` on
FastAPI BackgroundTasks to append the rest (reusing the phase-1 retrieval
context). Clients poll :func:`get_quiz_questions` while
``generation_status == 'generating'``.

Persistence lives in three tables: ``quiz_sessions`` / ``quiz_questions`` /
``quiz_responses`` (see schema.sql + migrations/0013_quiz_v3.sql). All public
functions return plain dicts so the router can serialize them directly.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import create_client

from providers import structured_call
from rag.retrieval import retrieve

logger = logging.getLogger(__name__)

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
_supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_COMPLEX = os.getenv("MODEL_COMPLEX")  # smart model for generation; None -> provider default

# Fast-start tuning: phase 1 returns this many questions; the remainder is
# generated in the background (split into two batches for very large quizzes).
FIRST_BATCH_SIZE = 3
TWO_BATCH_THRESHOLD = 15  # num_requested >= this -> background work runs as 2 calls

# Confidence-calibration taps (workstream E). Stored on quiz_responses.
CONFIDENCE_LEVELS = ("sure", "thinkso", "guessing")

# Difficulty guidance mirrors practice_generator so the two modes stay coherent.
DIFFICULTY_SPECS = {
    "easy": "Basic recall, definitions, and single-concept understanding (Bloom: Remember/Understand).",
    "medium": "Application and analysis with multi-step reasoning across connected concepts (Bloom: Apply/Analyze).",
    "hard": "Synthesis, evaluation, edge cases, and trade-offs (Bloom: Evaluate/Create).",
}

QUIZ_QUESTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question stem."},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Exactly four options, each prefixed 'A) ', 'B) ', 'C) ', 'D) '.",
                    },
                    "correct_answer": {
                        "type": "string",
                        "enum": ["A", "B", "C", "D"],
                        "description": "Letter of the correct option.",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Why the answer is correct and why the others are wrong.",
                    },
                    "concept": {
                        "type": "string",
                        "description": "Short topic/concept label this question tests (for mastery tracking).",
                    },
                    "source_doc": {
                        "type": "string",
                        "description": "Document name from the provided sources this question is grounded in.",
                    },
                    "source_page": {
                        "type": ["integer", "null"],
                        "description": "Page/slide number from the source, or null if unknown.",
                    },
                },
                "required": ["question", "options", "correct_answer", "explanation", "concept"],
            },
        }
    },
    "required": ["questions"],
}


def _build_context(results: List[Dict[str, Any]], topic: str) -> str:
    """Render retrieved chunks into a numbered, source-attributed context block."""
    if not results:
        return f"No specific course materials were found for {topic}."
    parts = [f"COURSE MATERIALS RELATED TO {topic.upper()}:"]
    for i, row in enumerate(results[:10], 1):
        content = (row.get("content") or "").strip()
        doc = row.get("doc_name", "unknown")
        page = row.get("page") or row.get("slide")
        head = f"[Source {i}: {doc}" + (f", page {page}" if page else "") + "]"
        parts.append(f"\n{head}\n{content}\n---")
    return "\n".join(parts)


def _generate_questions(topic: str, context: str, difficulty: str, count: int,
                        avoid_questions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Generate ``count`` grounded MCQs via guaranteed-schema tool use.

    ``avoid_questions`` lists stems already in the quiz so background batches
    don't duplicate the phase-1 questions (or each other).
    """
    spec = DIFFICULTY_SPECS.get(difficulty, DIFFICULTY_SPECS["medium"])
    avoid_block = ""
    if avoid_questions:
        listed = "\n".join(f"- {q}" for q in avoid_questions if q)
        avoid_block = (
            "\nALREADY-ASKED QUESTIONS (do NOT repeat or trivially rephrase any of these; "
            f"cover different facts/angles):\n{listed}\n"
        )
    prompt = (
        f"Create {count} high-quality multiple-choice quiz questions about \"{topic}\" "
        f"at {difficulty.upper()} difficulty.\n\n"
        f"DIFFICULTY: {spec}\n\n"
        "RULES:\n"
        "- Base every question STRICTLY on the COURSE CONTENT below; use its terminology.\n"
        "- Exactly four options, each prefixed 'A) ', 'B) ', 'C) ', 'D) '.\n"
        "- Make distractors plausible but unambiguously wrong.\n"
        "- 'correct_answer' is the single letter (A/B/C/D).\n"
        "- 'concept' is a short label naming what the question tests.\n"
        "- 'source_doc'/'source_page' must come from one of the provided [Source N: ...] headers.\n"
        f"{avoid_block}\n"
        f"COURSE CONTENT:\n{context}"
    )
    out = structured_call(
        [{"role": "user", "content": prompt}],
        schema=QUIZ_QUESTION_SCHEMA,
        tool_name="quiz_questions",
        model=MODEL_COMPLEX,
        max_tokens=4000,
    )
    questions = out.get("questions") if isinstance(out, dict) else None
    return questions or []


def _normalize_question(raw: Dict[str, Any], index: int, topic: str, difficulty: str) -> Dict[str, Any]:
    """Coerce a model-produced question into the stable runner shape."""
    options = [str(o) for o in (raw.get("options") or []) if str(o).strip()][:4]
    letter = str(raw.get("correct_answer", "A")).strip().upper()[:1]
    if letter not in {"A", "B", "C", "D"}:
        letter = "A"
    return {
        "id": f"q{index}",
        "question": str(raw.get("question", "")).strip(),
        "options": options,
        "correct_answer": letter,
        "explanation": str(raw.get("explanation", "")).strip(),
        "concept": str(raw.get("concept") or topic).strip(),
        "difficulty": difficulty,
        "source": {
            "doc_name": (raw.get("source_doc") or None),
            "page": raw.get("source_page"),
        },
    }


def _normalize_batch(raw_questions: List[Dict[str, Any]], start_index: int,
                     topic: Optional[str], difficulty: str) -> List[Dict[str, Any]]:
    """Normalize a raw model batch, numbering ids from ``start_index``."""
    normalized = []
    index = start_index
    for raw in raw_questions:
        if not (raw.get("question") and len(raw.get("options") or []) >= 4):
            continue
        normalized.append(_normalize_question(raw, index, topic or "General", difficulty))
        index += 1
    return normalized


def _question_rows(quiz_id: str, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map normalized questions to quiz_questions rows."""
    return [{
        "quiz_id": quiz_id,
        "question_id": q["id"],
        "question": q["question"],
        "options": q["options"],
        "correct_answer": q["correct_answer"],
        "explanation": q["explanation"],
        "concept": q["concept"],
        "difficulty": q["difficulty"],
        "source_doc": q["source"]["doc_name"],
        "source_page": q["source"]["page"],
    } for q in questions]


def _client_question(q: Dict[str, Any]) -> Dict[str, Any]:
    """Client-safe view of a normalized question: hides the answer key."""
    return {
        "id": q["id"],
        "question": q["question"],
        "options": q["options"],
        "concept": q["concept"],
        "difficulty": q["difficulty"],
        "source": q["source"],
    }


def generate_quiz(course_id: str, topic: Optional[str] = None,
                  num_questions: int = 10, difficulty: str = "medium",
                  user_id: Optional[str] = None) -> Dict[str, Any]:
    """Phase 1 of two-phase generation: return a playable session FAST.

    Retrieves once, generates only the first ``min(FIRST_BATCH_SIZE, N)``
    questions, persists the session (owned by ``user_id``) + those questions,
    and returns immediately. When more questions were requested than generated
    here, the payload carries a private ``_background`` dict which the router
    pops and hands to :func:`generate_remaining_questions` via BackgroundTasks.

    The returned questions omit ``correct_answer``/``explanation`` so the answer
    key is never shipped to the client up front — those are revealed
    per-question by :func:`grade_answer`.
    """
    query = topic or "core concepts and key topics of this course"
    results = retrieve(query, course_id, top_k=12)
    context = _build_context(results, topic or "this course")

    first_count = min(FIRST_BATCH_SIZE, num_questions)
    raw_questions = _generate_questions(topic or "this course", context, difficulty, first_count)
    questions = _normalize_batch(raw_questions, 1, topic, difficulty)
    if not questions:
        raise RuntimeError("Quiz generation produced no valid questions.")

    remaining = num_questions - len(questions)
    generation_status = "generating" if remaining > 0 else "ready"

    quiz_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    _supabase.table("quiz_sessions").insert({
        "id": quiz_id,
        "course_id": course_id,
        "user_id": user_id,
        "topic": topic,
        "difficulty": difficulty,
        "num_questions": len(questions),
        "num_requested": num_questions,
        "generation_status": generation_status,
        "status": "active",
        "created_at": now,
    }).execute()
    _supabase.table("quiz_questions").insert(_question_rows(quiz_id, questions)).execute()

    payload: Dict[str, Any] = {
        "quiz_id": quiz_id,
        "difficulty": difficulty,
        "topic": topic,
        "num_questions": len(questions),
        "num_requested": num_questions,
        "generation_status": generation_status,
        "questions": [_client_question(q) for q in questions],
    }
    if remaining > 0:
        payload["_background"] = {
            "quiz_id": quiz_id,
            "topic": topic,
            "context": context,
            "difficulty": difficulty,
            "remaining": remaining,
            "existing_questions": [q["question"] for q in questions],
            "start_index": len(questions) + 1,
            "num_requested": num_questions,
        }
    return payload


def generate_remaining_questions(quiz_id: str, topic: Optional[str], context: str,
                                 difficulty: str, remaining: int,
                                 existing_questions: List[str], start_index: int,
                                 num_requested: int) -> None:
    """Background phase 2: append the remaining questions to a quiz.

    Reuses the phase-1 retrieval context (no second retrieve). Runs as one
    structured call, or two batches when the request is large
    (``num_requested >= TWO_BATCH_THRESHOLD``). Each batch's prompt lists every
    question generated so far, so batches don't duplicate phase 1 or each other.

    Failure policy: any error flips generation_status to 'partial' and logs —
    the session stays playable with whatever questions exist.
    """
    try:
        if num_requested >= TWO_BATCH_THRESHOLD:
            batch_sizes = [remaining - remaining // 2, remaining // 2]
        else:
            batch_sizes = [remaining]

        seen = list(existing_questions)
        next_index = start_index
        generated_total = 0
        for size in [s for s in batch_sizes if s > 0]:
            raw = _generate_questions(topic or "this course", context, difficulty, size,
                                      avoid_questions=seen)
            batch = _normalize_batch(raw, next_index, topic, difficulty)
            if not batch:
                raise RuntimeError("Background quiz batch produced no valid questions.")
            _supabase.table("quiz_questions").insert(_question_rows(quiz_id, batch)).execute()
            seen.extend(q["question"] for q in batch)
            next_index += len(batch)
            generated_total += len(batch)

        _supabase.table("quiz_sessions").update({
            "generation_status": "ready",
            "num_questions": (start_index - 1) + generated_total,
        }).eq("id", quiz_id).execute()
    except Exception:  # noqa: BLE001  background failure must not kill the session
        logger.warning("Background question generation failed for quiz %s "
                       "(requested=%d, remaining=%d); marking partial",
                       quiz_id, num_requested, remaining, exc_info=True)
        try:
            _supabase.table("quiz_sessions").update({
                "generation_status": "partial",
            }).eq("id", quiz_id).execute()
        except Exception:  # noqa: BLE001
            logger.warning("Couldn't mark quiz %s partial", quiz_id, exc_info=True)


def _fetch_session(quiz_id: str) -> Optional[Dict[str, Any]]:
    resp = _supabase.table("quiz_sessions").select("*").eq("id", quiz_id).limit(1).execute()
    return resp.data[0] if resp.data else None


def get_quiz_questions(quiz_id: str, user_id: str) -> Dict[str, Any]:
    """Sanitized questions + generation progress for polling clients.

    Ownership: the session must belong to ``user_id``. Legacy pre-0013 sessions
    (user_id NULL) are admitted. Unknown and foreign quizzes raise the SAME
    KeyError (-> 404) so ids can't be enumerated.
    """
    session = _fetch_session(quiz_id)
    if not session or session.get("user_id") not in (None, user_id):
        raise KeyError(f"Quiz {quiz_id} not found.")

    rows = _supabase.table("quiz_questions").select("*").eq("quiz_id", quiz_id).execute().data or []

    def _order(row: Dict[str, Any]) -> int:
        qid = str(row.get("question_id") or "q0")
        try:
            return int(qid.lstrip("q"))
        except ValueError:
            return 0

    questions = [_client_question({
        "id": row.get("question_id"),
        "question": row.get("question"),
        "options": row.get("options") or [],
        "concept": row.get("concept"),
        "difficulty": row.get("difficulty"),
        "source": {"doc_name": row.get("source_doc"), "page": row.get("source_page")},
    }) for row in sorted(rows, key=_order)]

    return {
        "quiz_id": quiz_id,
        "generation_status": session.get("generation_status") or "ready",
        "num_requested": session.get("num_requested") or len(questions),
        "num_questions": len(questions),
        "questions": questions,
    }


# Resume-everywhere (V4 workstream C): how many unfinished sessions to surface.
IN_PROGRESS_LIMIT = 3
FINISHED_STATUSES = ("completed", "submitted")


def _question_index(question_id: Any) -> int:
    """Numeric index of a 'qN' question id ('q7' -> 7); unparseable ids sort first."""
    try:
        return int(str(question_id or "q0").lstrip("q"))
    except ValueError:
        return 0


def get_in_progress_quizzes(course_id: str, user_id: str) -> List[Dict[str, Any]]:
    """Newest unfinished quiz sessions for (course, user) — resume everywhere.

    Excludes finished sessions (status in FINISHED_STATUSES) and sessions with
    zero stored questions (nothing to resume). Returns at most
    ``IN_PROGRESS_LIMIT`` items, newest first by created_at.
    """
    rows = (_supabase.table("quiz_sessions").select("*")
            .eq("course_id", course_id).eq("user_id", user_id)
            .execute().data or [])
    open_sessions = sorted(
        (r for r in rows if (r.get("status") or "active") not in FINISHED_STATUSES),
        key=lambda r: str(r.get("created_at") or ""),
        reverse=True,
    )

    items: List[Dict[str, Any]] = []
    for session in open_sessions:
        if len(items) >= IN_PROGRESS_LIMIT:
            break
        quiz_id = session.get("id")
        questions = (_supabase.table("quiz_questions").select("question_id")
                     .eq("quiz_id", quiz_id).execute().data or [])
        if not questions:
            continue  # nothing generated yet -> nothing to resume
        responses = (_supabase.table("quiz_responses").select("question_id")
                     .eq("quiz_id", quiz_id).eq("user_id", user_id)
                     .execute().data or [])
        items.append({
            "quiz_id": quiz_id,
            "topic": session.get("topic"),
            "difficulty": session.get("difficulty"),
            "num_requested": session.get("num_requested") or len(questions),
            "num_answered": len({r.get("question_id") for r in responses}),
            "num_available": len(questions),
            "generation_status": session.get("generation_status") or "ready",
            "created_at": session.get("created_at"),
        })
    return items


def get_quiz_responses(quiz_id: str, user_id: str) -> Dict[str, Any]:
    """The user's saved answers for a quiz — resume state for a picked session.

    Ownership mirrors :func:`get_quiz_questions`: legacy NULL-user sessions are
    admitted; unknown and foreign quizzes raise the SAME KeyError (-> 404) so
    ids can't be enumerated. The latest response per question wins (same
    "last wins" rule as :func:`submit_quiz`).
    """
    session = _fetch_session(quiz_id)
    if not session or session.get("user_id") not in (None, user_id):
        raise KeyError(f"Quiz {quiz_id} not found.")

    rows = (_supabase.table("quiz_responses").select("*")
            .eq("quiz_id", quiz_id).eq("user_id", user_id)
            .execute().data or [])
    latest: Dict[str, Dict[str, Any]] = {}
    for r in rows:  # insertion order == chronological; last response wins
        latest[r["question_id"]] = r

    responses = [{
        "question_id": question_id,
        "selected": r.get("selected"),
        "is_correct": r.get("is_correct"),
        "confidence": r.get("confidence"),
    } for question_id, r in sorted(latest.items(), key=lambda kv: _question_index(kv[0]))]
    return {"quiz_id": quiz_id, "responses": responses}


def _fetch_question(quiz_id: str, question_id: str) -> Optional[Dict[str, Any]]:
    resp = (_supabase.table("quiz_questions")
            .select("*")
            .eq("quiz_id", quiz_id)
            .eq("question_id", question_id)
            .limit(1)
            .execute())
    return resp.data[0] if resp.data else None


def grade_answer(quiz_id: str, question_id: str, selected: str,
                 time_taken: float = 0.0, user_id: str = "anonymous",
                 confidence: Optional[str] = None) -> Dict[str, Any]:
    """Grade a single answer, persist the response, and feed mastery analytics.

    Fast by design: wrong answers return the stored explanation immediately; the
    extra LLM work (grounded mistake explanation + review seeding) is deferred —
    the router schedules :func:`followup_wrong_answer` on BackgroundTasks so
    wrong-answer latency equals right-answer latency.

    ``confidence`` is the learner's pre-reveal tap ('sure'|'thinkso'|'guessing',
    already validated by the router) and is stored for calibration scoring.
    """
    question = _fetch_question(quiz_id, question_id)
    if not question:
        raise KeyError(f"Question {question_id} not found in quiz {quiz_id}.")

    selected_letter = str(selected or "").strip().upper()[:1]
    correct_letter = str(question["correct_answer"]).strip().upper()[:1]
    is_correct = selected_letter == correct_letter

    _supabase.table("quiz_responses").insert({
        "quiz_id": quiz_id,
        "question_id": question_id,
        "user_id": user_id,
        "selected": selected_letter,
        "is_correct": is_correct,
        "time_taken": float(time_taken or 0.0),
        "confidence": confidence if confidence in CONFIDENCE_LEVELS else None,
        "ts": datetime.utcnow().isoformat(),
    }).execute()

    # Feed per-concept mastery (confidence = 1.0 if correct else 0.0).
    course_id = _quiz_course_id(quiz_id)
    try:
        from deps import analytics_engine
        analytics_engine.track_quiz_answer(
            user_id=user_id,
            course_id=course_id or "",
            concept=question.get("concept") or "general",
            question=question.get("question") or "",
            is_correct=is_correct,
            time_taken=float(time_taken or 0.0),
        )
    except Exception:  # noqa: BLE001  analytics must never break grading
        logger.warning("Quiz analytics tracking failed (quiz=%s question=%s user=%s)",
                       quiz_id, question_id, user_id, exc_info=True)

    return {
        "is_correct": is_correct,
        "correct_answer": correct_letter,
        "explanation": question.get("explanation", ""),
        "concept": question.get("concept", ""),
        "source": {"doc_name": question.get("source_doc"), "page": question.get("source_page")},
        # Kept for response-shape compatibility; the grounded mistake explanation
        # is now produced in the background (followup_wrong_answer), not inline.
        "mistake_explanation": "",
        "mistake_source": {"doc_name": None, "page": None},
    }


def followup_wrong_answer(quiz_id: str, question_id: str, selected: str,
                          user_id: str = "anonymous") -> None:
    """Deferred wrong-answer enrichment (runs on BackgroundTasks after grading).

    Produces the grounded "explain my mistake" text and seeds a spaced-repetition
    review item from the miss. Every failure is logged and swallowed — this can
    never affect a response the user already received.
    """
    question = _fetch_question(quiz_id, question_id)
    if not question:
        logger.warning("followup_wrong_answer: question %s not found in quiz %s",
                       question_id, quiz_id)
        return

    course_id = _quiz_course_id(quiz_id)
    correct_letter = str(question.get("correct_answer") or "A").strip().upper()[:1]
    correct_text = _correct_option_text(question, correct_letter)
    selected_text = _correct_option_text(question, str(selected or "").strip().upper()[:1])

    mistake_explanation = ""
    try:
        import mistake_engine
        grounded = mistake_engine.explain_mistake(
            course_id or "", question.get("question") or "",
            question.get("concept") or "", selected_text, correct_text,
        )
        mistake_explanation = grounded.get("explanation") or ""
    except Exception:  # noqa: BLE001
        logger.warning("Quiz explain_mistake failed (quiz=%s question=%s)",
                       quiz_id, question_id, exc_info=True)

    try:
        import review_engine
        review_engine.seed_from_mistake(
            user_id=user_id,
            course_id=course_id or "",
            concept=question.get("concept") or "general",
            prompt=question.get("question") or "",
            answer=correct_text,
            # Prefer the grounded, cited explanation when we have one.
            explanation=mistake_explanation or question.get("explanation") or "",
            source="quiz",
        )
    except Exception:  # noqa: BLE001
        logger.warning("Quiz review seeding failed (quiz=%s question=%s user=%s)",
                       quiz_id, question_id, user_id, exc_info=True)


def _correct_option_text(question: Dict[str, Any], letter: str) -> str:
    """Resolve a correct-answer letter (A-D) to the full option text."""
    options = question.get("options") or []
    idx = "ABCD".find((letter or "A").upper()[:1])
    if 0 <= idx < len(options):
        return str(options[idx])
    return letter


def _quiz_course_id(quiz_id: str) -> Optional[str]:
    resp = _supabase.table("quiz_sessions").select("course_id").eq("id", quiz_id).limit(1).execute()
    return resp.data[0]["course_id"] if resp.data else None


def _calibration(latest_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calibration read-out: per-confidence-level accuracy + confident-wrong count."""
    buckets = {level: {"n": 0, "correct": 0} for level in CONFIDENCE_LEVELS}
    confident_wrong = 0
    for r in latest_responses:
        level = r.get("confidence")
        if level not in buckets:
            continue
        buckets[level]["n"] += 1
        if r.get("is_correct"):
            buckets[level]["correct"] += 1
        elif level == "sure":
            confident_wrong += 1
    return {**buckets, "confident_wrong": confident_wrong}


def submit_quiz(quiz_id: str, user_id: str = "anonymous") -> Dict[str, Any]:
    """Finalize a quiz: compute score, per-topic breakdown, weak areas, and the
    confidence-calibration read-out."""
    questions = _supabase.table("quiz_questions").select("*").eq("quiz_id", quiz_id).execute().data or []
    responses = (_supabase.table("quiz_responses")
                 .select("*").eq("quiz_id", quiz_id).eq("user_id", user_id)
                 .execute().data or [])

    # Last response per question wins (user may re-answer before submit).
    latest: Dict[str, Dict[str, Any]] = {}
    for r in responses:
        latest[r["question_id"]] = r

    total = len(questions)
    correct = sum(1 for r in latest.values() if r.get("is_correct"))

    by_concept: Dict[str, Dict[str, int]] = {}
    for q in questions:
        concept = q.get("concept") or "general"
        bucket = by_concept.setdefault(concept, {"correct": 0, "total": 0})
        bucket["total"] += 1
        r = latest.get(q["question_id"])
        if r and r.get("is_correct"):
            bucket["correct"] += 1

    by_topic = [{
        "topic": concept,
        "correct": b["correct"],
        "total": b["total"],
        "pct": round(100.0 * b["correct"] / b["total"], 1) if b["total"] else 0.0,
    } for concept, b in by_concept.items()]
    weak_areas = [t["topic"] for t in by_topic if t["pct"] < 70.0]

    pct = round(100.0 * correct / total, 1) if total else 0.0
    score = {"correct": correct, "total": total, "pct": pct}

    _supabase.table("quiz_sessions").update({
        "status": "completed",
        "score": score,
    }).eq("id", quiz_id).execute()

    return {"score": score, "by_topic": sorted(by_topic, key=lambda t: t["pct"]),
            "weak_areas": weak_areas, "calibration": _calibration(list(latest.values()))}
