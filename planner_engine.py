"""Study planner engine — generate and persist a day-by-day revision schedule.

Builds a grounded plan from a course's extracted topics, distributing them across
the available days with spaced-repetition spacing (first pass, then reviews at
+1 / +3 / +7 days). Uses ``structured_call`` for a schema-guaranteed plan and
persists it to the ``study_plans`` table.
"""
from __future__ import annotations

import os
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import create_client

from providers import structured_call

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
_supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

STUDY_PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "days": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer", "description": "1-based day index within the plan."},
                    "topics": {"type": "array", "items": {"type": "string"}},
                    "duration_minutes": {"type": "integer"},
                    "type": {"type": "string", "enum": ["review", "new", "practice"]},
                },
                "required": ["day", "topics", "duration_minutes", "type"],
            },
        }
    },
    "required": ["days"],
}


def _course_title(course_id: str) -> str:
    try:
        resp = _supabase.table("courses").select("title").eq("course_id", course_id).limit(1).execute()
        if resp.data and resp.data[0].get("title"):
            return resp.data[0]["title"]
    except Exception:  # noqa: BLE001
        pass
    return course_id


def _extract_topics(course_id: str) -> List[str]:
    """Course Brain topics keep the planner consistent with every other feature."""
    try:
        import course_brain
        topics = course_brain.topic_names(course_id, auto_generate=True)
        return [t for t in (topics or []) if isinstance(t, str) and t.strip()][:15]
    except Exception as e:  # noqa: BLE001
        print(f"Planner topic extraction failed: {e}")
        return []


def _weak_topics(course_id: str, user_id: str) -> List[str]:
    """Topics the student is weakest on (mastery ascending)."""
    rows = (_supabase.table("learning_progress")
            .select("topic, mastery_level")
            .eq("user_id", user_id).eq("course_id", course_id)
            .execute().data or [])
    rows = [r for r in rows if r.get("topic")]
    rows.sort(key=lambda r: float(r.get("mastery_level") or 0.0))
    return [r["topic"] for r in rows]


def _prereq_order(course_id: str) -> List[str]:
    """A prerequisite-first ordering of concepts (topological sort of the graph).
    Returns [] when no graph exists."""
    try:
        import concept_graph
        graph = concept_graph.get_graph(course_id)
    except Exception:  # noqa: BLE001
        graph = None
    if not graph:
        return []
    concepts = list(graph.get("concepts", []))
    edges = graph.get("edges", [])
    # Kahn's algorithm; prerequisite -> concept.
    indeg = {c: 0 for c in concepts}
    adj: Dict[str, List[str]] = {c: [] for c in concepts}
    for e in edges:
        pre, con = e.get("prerequisite"), e.get("concept")
        if pre in indeg and con in indeg:
            adj[pre].append(con)
            indeg[con] += 1
    queue = [c for c in concepts if indeg[c] == 0]
    order = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    # Append any leftovers (cycles) to stay total.
    order += [c for c in concepts if c not in order]
    return order


def _prioritized_topics(course_id: str, user_id: str) -> List[str]:
    """Weak topics first, but each topic preceded by its prerequisites.

    Orders by prerequisite chain (from the concept graph), then within that order
    bubbles the weakest topics toward the front so revision targets gaps while
    still respecting 'learn the foundation first'.
    """
    weak = _weak_topics(course_id, user_id)
    order = _prereq_order(course_id)
    base = order or _extract_topics(course_id) or weak
    weak_set = {w.lower() for w in weak}
    # Stable sort: weak topics first, otherwise keep prerequisite order.
    indexed = list(enumerate(base))
    indexed.sort(key=lambda iv: (iv[1].lower() not in weak_set, iv[0]))
    return [v for _, v in indexed]


def _resolve_horizon(days_available: Optional[int], exam_date: Optional[str]) -> (int, date):
    """Return (num_days, start_date). If an exam date is given, the plan runs from
    today up to the day before the exam; otherwise it runs for days_available days."""
    today = datetime.utcnow().date()
    exam_dt: Optional[date] = None
    if exam_date:
        try:
            exam_dt = datetime.fromisoformat(str(exam_date)[:10]).date()
        except Exception:  # noqa: BLE001
            exam_dt = None
    if exam_dt and exam_dt > today:
        span = (exam_dt - today).days
        num_days = days_available if days_available else span
        num_days = max(1, min(num_days, span))
    else:
        num_days = days_available or 10
    return max(1, min(num_days, 60)), today


def _generate_day_plan(course_id: str, topics: List[str], num_days: int,
                       minutes_per_day: int, mode: str = "balanced",
                       weak_topics: Optional[List[str]] = None,
                       due_reviews: int = 0) -> List[Dict[str, Any]]:
    """Ask the model for a spaced-repetition day plan over the given topics."""
    topic_list = "\n".join(f"- {t}" for t in topics) if topics else "(infer topics from the course)"
    weak_line = ""
    if mode == "weak_first" and weak_topics:
        weak_line = (
            "- PRIORITIZE the student's weak areas — front-load and repeat these: "
            f"{', '.join(weak_topics[:8])}.\n"
        )
    review_line = (
        f"- The student has {due_reviews} mistakes already due for review; fold a short "
        "'review' block into the first 1-2 days to clear them.\n"
        if due_reviews else ""
    )
    prompt = (
        f"Build a {num_days}-day study plan for the course covering these topics "
        f"(already ordered foundation-first):\n{topic_list}\n\n"
        f"RULES:\n"
        f"- Each day budgets about {minutes_per_day} minutes (set duration_minutes accordingly).\n"
        f"{weak_line}{review_line}"
        "- Introduce new topics on early days ('new'), then schedule spaced reviews of each "
        "topic at roughly +1, +3, and +7 days after it was introduced ('review').\n"
        "- Respect prerequisites: never schedule a topic before the topics it depends on.\n"
        "- Include 'practice' sessions before the end to consolidate.\n"
        "- Use the 1-based 'day' index from 1 to "
        f"{num_days}. Every day from 1 to {num_days} should appear exactly once."
    )
    out = structured_call(
        [{"role": "user", "content": prompt}],
        schema=STUDY_PLAN_SCHEMA,
        tool_name="study_plan",
        model=os.getenv("MODEL_COMPLEX"),
        max_tokens=2500,
    )
    days = out.get("days") if isinstance(out, dict) else None
    return days or []


def _normalize_days(raw_days: List[Dict[str, Any]], num_days: int, start: date,
                    minutes_per_day: int) -> List[Dict[str, Any]]:
    """Attach concrete dates, clamp durations, and ensure one entry per day."""
    by_index: Dict[int, Dict[str, Any]] = {}
    for d in raw_days:
        try:
            idx = int(d.get("day", 0))
        except Exception:  # noqa: BLE001
            continue
        if 1 <= idx <= num_days and idx not in by_index:
            by_index[idx] = d

    out: List[Dict[str, Any]] = []
    for idx in range(1, num_days + 1):
        d = by_index.get(idx, {})
        topics = [str(t).strip() for t in (d.get("topics") or []) if str(t).strip()]
        kind = d.get("type") if d.get("type") in ("review", "new", "practice") else "review"
        try:
            duration = int(d.get("duration_minutes") or minutes_per_day)
        except Exception:  # noqa: BLE001
            duration = minutes_per_day
        duration = max(15, min(duration, minutes_per_day))
        out.append({
            "date": (start + timedelta(days=idx - 1)).isoformat(),
            "topics": topics or ["Review previous material"],
            "duration_minutes": duration,
            "type": kind,
        })
    return out


def generate_study_plan(course_id: str, days_available: Optional[int] = None,
                        hours_per_day: Optional[float] = None,
                        exam_date: Optional[str] = None,
                        user_id: str = "anonymous",
                        mode: str = "balanced") -> Dict[str, Any]:
    """Generate, persist, and return a study plan for a course.

    ``mode='weak_first'`` re-prioritizes from the student's *current* state: weak
    topics, prerequisite order, and any reviews already due — i.e. a live replan.
    """
    num_days, start = _resolve_horizon(days_available, exam_date)
    minutes_per_day = int((hours_per_day or 2) * 60)

    weak = _weak_topics(course_id, user_id)
    if mode == "weak_first":
        topics = _prioritized_topics(course_id, user_id) or _extract_topics(course_id)
    else:
        topics = _extract_topics(course_id)

    due_reviews = 0
    try:
        import review_engine
        due_reviews = review_engine.due_count(course_id, user_id)
    except Exception as e:  # noqa: BLE001
        print(f"planner review-count lookup failed: {e}")

    raw_days = _generate_day_plan(course_id, topics, num_days, minutes_per_day,
                                  mode=mode, weak_topics=weak, due_reviews=due_reviews)
    days = _normalize_days(raw_days, num_days, start, minutes_per_day)
    if not days:
        raise RuntimeError("No study plan could be generated for this course.")

    plan_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    plan = {"id": plan_id, "course_id": course_id, "days": days,
            "created_at": created_at, "mode": mode}

    _supabase.table("study_plans").insert({
        "id": plan_id,
        "course_id": course_id,
        "user_id": user_id,
        "plan": plan,
        "created_at": created_at,
    }).execute()

    return plan


def get_latest_plan(course_id: str) -> Optional[Dict[str, Any]]:
    """Return the most recent persisted plan for a course, or None."""
    resp = (_supabase.table("study_plans")
            .select("plan")
            .eq("course_id", course_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute())
    if resp.data:
        return resp.data[0]["plan"]
    return None
