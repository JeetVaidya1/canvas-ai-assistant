"""Flashcard engine — persistence + SM-2 spaced repetition scheduling.

Cards live in ``flashcards`` (one row per Q/A, optionally tied to a note). Each
user's review state for a card lives in ``flashcard_reviews`` (ease, interval,
repetitions, due_date). Scheduling uses the classic SM-2 algorithm so cards the
user struggles with resurface sooner and mastered cards space out.
"""
from __future__ import annotations

import os
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
_supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

DEFAULT_EASE = 2.5
MIN_EASE = 1.3


def sm2(grade: int, ease: float, interval: int, repetitions: int) -> Dict[str, Any]:
    """One SM-2 step. ``grade`` is 0..5 (>=3 is a pass).

    Returns the new ease, interval (days), repetitions, and due_date (ISO date).
    """
    grade = max(0, min(5, int(grade)))
    if grade < 3:
        repetitions = 0
        interval = 1
    else:
        if repetitions == 0:
            interval = 1
        elif repetitions == 1:
            interval = 6
        else:
            interval = round(interval * ease)
        repetitions += 1

    ease = ease + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02))
    ease = max(MIN_EASE, ease)

    due = date.today() + timedelta(days=max(1, interval))
    return {
        "ease": round(ease, 3),
        "interval": int(interval),
        "repetitions": int(repetitions),
        "due_date": due.isoformat(),
    }


def save_cards(course_id: str, cards: List[Dict[str, str]],
               note_id: Optional[str] = None) -> Dict[str, Any]:
    """Persist a batch of Q/A cards for a course. Skips exact-duplicate fronts
    already present in the course deck. Returns {saved, skipped}."""
    existing = (_supabase.table("flashcards").select("q")
                .eq("course_id", course_id).execute().data or [])
    seen = {(r.get("q") or "").strip() for r in existing}

    rows = []
    skipped = 0
    now = datetime.utcnow().isoformat()
    for c in cards:
        q = str(c.get("q", "")).strip()
        a = str(c.get("a", "")).strip()
        if not q or not a:
            continue
        if q in seen:
            skipped += 1
            continue
        seen.add(q)
        rows.append({
            "id": str(uuid.uuid4()),
            "course_id": course_id,
            "note_id": note_id,
            "q": q,
            "a": a,
            "created_at": now,
        })

    if rows:
        _supabase.table("flashcards").insert(rows).execute()
    return {"saved": len(rows), "skipped": skipped}


def get_deck(course_id: str, user_id: str) -> Dict[str, Any]:
    """Return the course deck with per-user review state, due cards first.

    Each card: {id, q, a, due, ease, interval, repetitions, due_date}. ``due`` is
    True for new cards (never reviewed) and cards whose due_date has passed.
    """
    cards = (_supabase.table("flashcards").select("*")
             .eq("course_id", course_id).execute().data or [])
    reviews = (_supabase.table("flashcard_reviews").select("*")
               .eq("user_id", user_id).execute().data or [])
    review_by_card = {r["card_id"]: r for r in reviews}

    today = date.today().isoformat()
    out = []
    for c in cards:
        r = review_by_card.get(c["id"])
        due_date = r.get("due_date") if r else None
        is_due = (due_date is None) or (str(due_date)[:10] <= today)
        out.append({
            "id": c["id"],
            "q": c["q"],
            "a": c["a"],
            "due": is_due,
            "ease": (r.get("ease") if r else DEFAULT_EASE),
            "interval": (r.get("interval") if r else 0),
            "repetitions": (r.get("repetitions") if r else 0),
            "due_date": due_date,
        })

    # Due cards first; among due, the longest-overdue (or new) first.
    out.sort(key=lambda c: (not c["due"], c["due_date"] or ""))
    due_count = sum(1 for c in out if c["due"])
    return {"cards": out, "total": len(out), "due_count": due_count}


def review_card(card_id: str, user_id: str, grade: int) -> Dict[str, Any]:
    """Apply an SM-2 review to a card for a user; upsert the review state."""
    existing = (_supabase.table("flashcard_reviews").select("*")
                .eq("card_id", card_id).eq("user_id", user_id).limit(1).execute().data)
    if existing:
        prev = existing[0]
        state = sm2(grade, float(prev.get("ease") or DEFAULT_EASE),
                    int(prev.get("interval") or 0), int(prev.get("repetitions") or 0))
        _supabase.table("flashcard_reviews").update({
            **state,
            "last_reviewed": datetime.utcnow().isoformat(),
        }).eq("id", prev["id"]).execute()
    else:
        state = sm2(grade, DEFAULT_EASE, 0, 0)
        _supabase.table("flashcard_reviews").insert({
            "card_id": card_id,
            "user_id": user_id,
            **state,
            "last_reviewed": datetime.utcnow().isoformat(),
        }).execute()

    return {"card_id": card_id, **state}
