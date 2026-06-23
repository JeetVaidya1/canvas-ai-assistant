"""Exam-readiness engine — a single predicted-score number per course/user.

Combines the student's per-concept mastery (``learning_progress``) with how
heavily each topic is tested (frequency in past-paper analyses), so the score
reflects "how ready am I for the kind of exam this professor actually gives",
not a flat average. Falls back to equal-weighting the course's extracted topics
when no past papers have been analyzed.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
_supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

GAP_THRESHOLD = 0.6  # mastery below this counts as a gap


def _mastery_map(course_id: str, user_id: str) -> Dict[str, float]:
    rows = (_supabase.table("learning_progress")
            .select("topic, mastery_level")
            .eq("user_id", user_id).eq("course_id", course_id)
            .execute().data or [])
    return {(r["topic"] or "").strip().lower(): float(r.get("mastery_level") or 0.0)
            for r in rows if r.get("topic")}


def _tested_topic_weights(course_id: str) -> Dict[str, float]:
    """Topic -> exam frequency weight, from past-paper analyses. {} if none."""
    rows = (_supabase.table("past_paper_analyses")
            .select("analysis_data")
            .eq("course_id", course_id)
            .execute().data or [])
    weights: Dict[str, float] = {}
    for row in rows:
        data = row.get("analysis_data") or {}
        analysis = data.get("analysis") or data  # tolerate both shapes
        topics = analysis.get("topics_covered") or []
        for t in topics:
            key = str(t).strip()
            if key:
                weights[key] = weights.get(key, 0.0) + 1.0
    return weights


def _match_mastery(topic: str, mastery: Dict[str, float]) -> Optional[float]:
    """Best mastery match for a topic name (exact, then substring either way)."""
    t = topic.strip().lower()
    if t in mastery:
        return mastery[t]
    for k, v in mastery.items():
        if k and (k in t or t in k):
            return v
    return None


def get_readiness(course_id: str, user_id: str) -> Dict[str, Any]:
    """Return predicted exam readiness for a course/user."""
    mastery = _mastery_map(course_id, user_id)
    weights = _tested_topic_weights(course_id)
    has_past_papers = bool(weights)

    if not weights:
        # No past papers — equal-weight the course's extracted topics.
        try:
            from deps import practice_generator
            topics = practice_generator.extract_topics_from_course(course_id) or []
        except Exception as e:  # noqa: BLE001
            print(f"Readiness topic fallback failed: {e}")
            topics = list({k for k in mastery})
        weights = {t: 1.0 for t in topics if t}

    if not weights:
        return {"score_pct": 0.0, "by_topic": [], "gaps": [],
                "confidence": "low", "has_past_papers": has_past_papers,
                "message": "Add course materials or a past paper to get a readiness estimate."}

    by_topic: List[Dict[str, Any]] = []
    weighted_sum = 0.0
    total_weight = 0.0
    for topic, w in weights.items():
        m = _match_mastery(topic, mastery)
        has_data = m is not None
        m = m if has_data else 0.0
        weighted_sum += w * m
        total_weight += w
        by_topic.append({
            "topic": topic,
            "mastery_pct": round(m * 100, 1),
            "weight": w,
            "has_data": has_data,
        })

    score_pct = round(100.0 * weighted_sum / total_weight, 1) if total_weight else 0.0

    # Gaps: tested topics that are weak, ranked by weight * deficit.
    gaps = sorted(
        [t for t in by_topic if t["mastery_pct"] < GAP_THRESHOLD * 100],
        key=lambda t: t["weight"] * (100 - t["mastery_pct"]),
        reverse=True,
    )
    gap_topics = [g["topic"] for g in gaps[:5]]

    # Confidence from how much performance data we actually have.
    interactions = (_supabase.table("user_interactions")
                    .select("id", count="exact")
                    .eq("user_id", user_id).eq("course_id", course_id)
                    .execute())
    n = interactions.count or 0
    confidence = "high" if n >= 30 else "medium" if n >= 10 else "low"

    return {
        "score_pct": score_pct,
        "by_topic": sorted(by_topic, key=lambda t: t["weight"], reverse=True),
        "gaps": gap_topics,
        "confidence": confidence,
        "has_past_papers": has_past_papers,
        "data_points": n,
    }
