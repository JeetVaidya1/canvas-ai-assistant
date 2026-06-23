"""Quiz engine — generate grounded MCQs, grade answers, score sessions.

Replaces the brittle ``quiz_assistant_engine`` heuristics for the quiz *runner*
flow (generate N questions -> answer one at a time -> instant feedback -> score).
Built on the shared Phase-3 foundations:

  - ``rag.retrieval.retrieve``  for hybrid + reranked grounding
  - ``providers.structured_call``  for schema-guaranteed question generation
  - ``learning_analytics``  for per-concept mastery tracking

Persistence lives in three tables: ``quiz_sessions`` / ``quiz_questions`` /
``quiz_responses`` (see schema.sql). All public functions return plain dicts so
the router can serialize them directly.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import create_client

from providers import structured_call
from rag.retrieval import retrieve

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
_supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_COMPLEX = os.getenv("MODEL_COMPLEX")  # smart model for generation; None -> provider default

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


def _generate_questions(topic: str, context: str, difficulty: str, count: int) -> List[Dict[str, Any]]:
    """Generate ``count`` grounded MCQs via guaranteed-schema tool use."""
    spec = DIFFICULTY_SPECS.get(difficulty, DIFFICULTY_SPECS["medium"])
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
        "- 'source_doc'/'source_page' must come from one of the provided [Source N: ...] headers.\n\n"
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


def generate_quiz(course_id: str, topic: Optional[str] = None,
                  num_questions: int = 10, difficulty: str = "medium") -> Dict[str, Any]:
    """Generate a quiz, persist it, and return ``{quiz_id, questions[...]}``.

    The returned questions omit ``correct_answer``/``explanation`` so the answer
    key is never shipped to the client up front — those are revealed per-question
    by :func:`grade_answer`.
    """
    query = topic or "core concepts and key topics of this course"
    results = retrieve(query, course_id, top_k=12)
    context = _build_context(results, topic or "this course")

    raw_questions = _generate_questions(topic or "this course", context, difficulty, num_questions)
    questions = [
        _normalize_question(raw, i, topic or "General", difficulty)
        for i, raw in enumerate(raw_questions, 1)
        if raw.get("question") and len(raw.get("options") or []) >= 4
    ]
    if not questions:
        raise RuntimeError("Quiz generation produced no valid questions.")

    quiz_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    _supabase.table("quiz_sessions").insert({
        "id": quiz_id,
        "course_id": course_id,
        "user_id": None,  # set per-response; sessions are anonymous-friendly
        "topic": topic,
        "difficulty": difficulty,
        "num_questions": len(questions),
        "status": "active",
        "created_at": now,
    }).execute()

    rows = [{
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
    _supabase.table("quiz_questions").insert(rows).execute()

    # Client-safe payload: hide the answer key.
    client_questions = [{
        "id": q["id"],
        "question": q["question"],
        "options": q["options"],
        "concept": q["concept"],
        "difficulty": q["difficulty"],
        "source": q["source"],
    } for q in questions]

    return {"quiz_id": quiz_id, "difficulty": difficulty, "topic": topic,
            "num_questions": len(questions), "questions": client_questions}


def _fetch_question(quiz_id: str, question_id: str) -> Optional[Dict[str, Any]]:
    resp = (_supabase.table("quiz_questions")
            .select("*")
            .eq("quiz_id", quiz_id)
            .eq("question_id", question_id)
            .limit(1)
            .execute())
    return resp.data[0] if resp.data else None


def grade_answer(quiz_id: str, question_id: str, selected: str,
                 time_taken: float = 0.0, user_id: str = "anonymous") -> Dict[str, Any]:
    """Grade a single answer, persist the response, and feed mastery analytics."""
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
    except Exception as e:  # noqa: BLE001  analytics must never break grading
        print(f"quiz analytics tracking failed: {e}")

    # Grounded "explain my mistake" + closed-loop review seeding (wrong only).
    mistake_explanation = ""
    mistake_source = {"doc_name": None, "page": None}
    if not is_correct:
        correct_text = _correct_option_text(question, correct_letter)
        selected_text = _correct_option_text(question, selected_letter)
        try:
            import mistake_engine
            grounded = mistake_engine.explain_mistake(
                course_id or "", question.get("question") or "",
                question.get("concept") or "", selected_text, correct_text,
            )
            mistake_explanation = grounded.get("explanation") or ""
            mistake_source = grounded.get("source") or mistake_source
        except Exception as e:  # noqa: BLE001
            print(f"quiz explain_mistake failed: {e}")
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
        except Exception as e:  # noqa: BLE001
            print(f"quiz review seeding failed: {e}")

    return {
        "is_correct": is_correct,
        "correct_answer": correct_letter,
        "explanation": question.get("explanation", ""),
        "concept": question.get("concept", ""),
        "source": {"doc_name": question.get("source_doc"), "page": question.get("source_page")},
        "mistake_explanation": mistake_explanation,
        "mistake_source": mistake_source,
    }


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


def submit_quiz(quiz_id: str, user_id: str = "anonymous") -> Dict[str, Any]:
    """Finalize a quiz: compute score, per-topic breakdown, and weak areas."""
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
            "weak_areas": weak_areas}
