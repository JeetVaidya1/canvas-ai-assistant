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
    """Reuse the practice generator's topic extraction for consistency."""
    try:
        from deps import practice_generator
        topics = practice_generator.extract_topics_from_course(course_id)
        return [t for t in (topics or []) if isinstance(t, str) and t.strip()][:15]
    except Exception as e:  # noqa: BLE001
        print(f"Planner topic extraction failed: {e}")
        return []


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
                       minutes_per_day: int) -> List[Dict[str, Any]]:
    """Ask the model for a spaced-repetition day plan over the given topics."""
    topic_list = "\n".join(f"- {t}" for t in topics) if topics else "(infer topics from the course)"
    prompt = (
        f"Build a {num_days}-day study plan for the course covering these topics:\n"
        f"{topic_list}\n\n"
        f"RULES:\n"
        f"- Each day budgets about {minutes_per_day} minutes (set duration_minutes accordingly).\n"
        "- Introduce new topics on early days ('new'), then schedule spaced reviews of each "
        "topic at roughly +1, +3, and +7 days after it was introduced ('review').\n"
        "- Include 'practice' sessions before the end to consolidate.\n"
        "- Order topics pedagogically (foundational first).\n"
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
                        user_id: str = "anonymous") -> Dict[str, Any]:
    """Generate, persist, and return a study plan for a course."""
    num_days, start = _resolve_horizon(days_available, exam_date)
    minutes_per_day = int((hours_per_day or 2) * 60)
    topics = _extract_topics(course_id)

    raw_days = _generate_day_plan(course_id, topics, num_days, minutes_per_day)
    days = _normalize_days(raw_days, num_days, start, minutes_per_day)
    if not days:
        raise RuntimeError("No study plan could be generated for this course.")

    plan_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    plan = {"id": plan_id, "course_id": course_id, "days": days, "created_at": created_at}

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
