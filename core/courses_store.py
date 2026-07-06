"""Course records in Supabase — replaces the legacy local ``courses.json`` store.

The old store (``deps.load_courses`` / ``deps.save_courses``) kept a mutable
JSON file on the instance's disk: state was lost on redeploy and silently
diverged with more than one instance. The ``courses`` table (see schema.sql,
with ``owner_id``) plus the ``files`` table already hold everything the JSON
file held, so this module is a thin, testable query layer over them.

The Supabase client is created lazily so tests can swap ``_db`` for a fake
(same pattern as tests/test_auth_access.py) without any network I/O.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from supabase import create_client

from core.config import get_settings

logger = logging.getLogger(__name__)

_db = None  # lazily-created Supabase client; tests monkeypatch this


class CourseStoreError(RuntimeError):
    """A course-store query failed (connectivity, constraint, etc.)."""


def _get_db():
    global _db
    if _db is None:
        settings = get_settings()
        _db = create_client(settings.supabase_url, settings.supabase_key)
    return _db


def _require_course_id(course_id: str) -> str:
    if not isinstance(course_id, str) or not course_id.strip():
        raise CourseStoreError("course_id must be a non-empty string")
    return course_id.strip()


def course_exists(course_id: str) -> bool:
    """True if a course row exists for this id."""
    course_id = _require_course_id(course_id)
    try:
        rows = (
            _get_db().table("courses")
            .select("course_id")
            .eq("course_id", course_id)
            .limit(1)
            .execute()
            .data
        )
    except Exception as exc:
        logger.exception("Course existence lookup failed for %s", course_id)
        raise CourseStoreError(f"course lookup failed for {course_id}") from exc
    return bool(rows)


def create_course(course_id: str, title: str, owner_id: Optional[str]) -> Dict[str, Any]:
    """Insert a course row; returns the created record."""
    course_id = _require_course_id(course_id)
    if not isinstance(title, str) or not title.strip():
        raise CourseStoreError("title must be a non-empty string")
    record = {"course_id": course_id, "title": title.strip(), "owner_id": owner_id}
    try:
        result = _get_db().table("courses").insert(record).execute()
    except Exception as exc:
        logger.exception("Course insert failed for %s", course_id)
        raise CourseStoreError(f"course insert failed for {course_id}") from exc
    return result.data[0] if getattr(result, "data", None) else record


def list_courses() -> List[Dict[str, Any]]:
    """All courses, newest first, as [{course_id, title}, ...]."""
    try:
        rows = (
            _get_db().table("courses")
            .select("course_id, title")
            .order("created_at", desc=True)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        logger.exception("Course listing failed")
        raise CourseStoreError("course listing failed") from exc
    return [{"course_id": row["course_id"], "title": row.get("title")} for row in rows]


def list_courses_for_user(user_id: str) -> List[Dict[str, Any]]:
    """Courses the user owns or has joined, as [{course_id, title}, ...].

    Ownership is ``courses.owner_id``; membership is a row in
    ``course_memberships`` (the same tables auth.user_owns_or_member checks,
    so listing and access control can never disagree). Owned courses come
    first, then joined ones, without duplicates.
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise CourseStoreError("user_id must be a non-empty string")
    user_id = user_id.strip()
    try:
        db = _get_db()
        owned = (
            db.table("courses")
            .select("course_id, title")
            .eq("owner_id", user_id)
            .order("created_at", desc=True)
            .execute()
            .data
            or []
        )
        memberships = (
            db.table("course_memberships")
            .select("course_id")
            .eq("user_id", user_id)
            .execute()
            .data
            or []
        )
        seen = {row["course_id"] for row in owned}
        courses = [{"course_id": row["course_id"], "title": row.get("title")} for row in owned]
        for membership in memberships:
            course_id = membership.get("course_id")
            if not course_id or course_id in seen:
                continue
            seen.add(course_id)
            rows = (
                db.table("courses")
                .select("course_id, title")
                .eq("course_id", course_id)
                .limit(1)
                .execute()
                .data
                or []
            )
            if rows:
                courses.append({"course_id": rows[0]["course_id"], "title": rows[0].get("title")})
    except CourseStoreError:
        raise
    except Exception as exc:
        logger.exception("Course listing failed for user %s", user_id)
        raise CourseStoreError(f"course listing failed for user {user_id}") from exc
    return courses


def delete_course(course_id: str) -> None:
    """Delete a course row (files/embeddings cascade via FK in schema.sql)."""
    course_id = _require_course_id(course_id)
    try:
        _get_db().table("courses").delete().eq("course_id", course_id).execute()
    except Exception as exc:
        logger.exception("Course delete failed for %s", course_id)
        raise CourseStoreError(f"course delete failed for {course_id}") from exc
