"""Authentication + authorization for the multi-user app.

Identity comes from a Supabase Auth JWT, verified on every request via GoTrue
(``auth.get_user``). The client-supplied ``user_id`` is never trusted again — the
authenticated subject is the only source of identity. A short cache avoids a
GoTrue round-trip on every call.

Authorization for course-scoped data goes through :func:`require_course_access`:
a user may touch a course only if they own it or have joined it.
"""
from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import Header, HTTPException
from supabase import create_client

logger = logging.getLogger(__name__)

load_dotenv()
_URL = os.getenv("SUPABASE_URL")
_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
_SERVICE_KEY = os.getenv("SUPABASE_KEY")

# Anon-key client is enough to validate a user token via GoTrue.
_auth_client = create_client(_URL, _ANON_KEY) if (_URL and _ANON_KEY) else None
# Service-role client for ownership/membership lookups (bypasses RLS by design).
_db = create_client(_URL, _SERVICE_KEY) if (_URL and _SERVICE_KEY) else None

_CACHE_TTL = 60  # seconds
_CACHE_MAX_SIZE = 1000  # bound memory: evict least-recently-used beyond this
_cache: "OrderedDict[str, tuple]" = OrderedDict()  # token -> (user_dict, expires_at)


def _bearer(authorization: Optional[str]) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    return authorization.split(" ", 1)[1].strip()


def _cache_get(token: str, now: float) -> Optional[Dict[str, Any]]:
    """Return the cached user for a token, expiring stale entries on access."""
    hit = _cache.get(token)
    if hit is None:
        return None
    if hit[1] <= now:
        _cache.pop(token, None)
        return None
    _cache.move_to_end(token)  # LRU: recent hits are evicted last
    return hit[0]


def _cache_put(token: str, info: Dict[str, Any], expires_at: float) -> None:
    """Insert into the token cache, evicting the least-recently-used entry
    once the cap is reached so the cache can't grow without bound."""
    _cache[token] = (info, expires_at)
    _cache.move_to_end(token)
    while len(_cache) > _CACHE_MAX_SIZE:
        _cache.popitem(last=False)


def _verify(token: str) -> Dict[str, Any]:
    now = time.time()
    cached = _cache_get(token, now)
    if cached is not None:
        return cached
    if _auth_client is None:
        raise HTTPException(status_code=500, detail="Auth is not configured (SUPABASE_ANON_KEY missing)")
    try:
        resp = _auth_client.auth.get_user(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    user = getattr(resp, "user", None)
    if not user or not getattr(user, "id", None):
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    info = {"id": user.id, "email": getattr(user, "email", None)}
    _cache_put(token, info, now + _CACHE_TTL)
    return info


async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """FastAPI dependency: returns {id, email} for the authenticated user or 401."""
    return _verify(_bearer(authorization))


async def current_user_id(authorization: Optional[str] = Header(None)) -> str:
    """Dependency returning just the authenticated user id (drop-in for the old
    spoofable ``user_id`` form param)."""
    return _verify(_bearer(authorization))["id"]


async def get_optional_user(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """Like get_current_user but returns None instead of 401 when unauthenticated."""
    if not authorization:
        return None
    try:
        return _verify(_bearer(authorization))
    except HTTPException:
        return None


def user_owns_or_member(course_id: str, user_id: str) -> bool:
    """True if the user owns the course or has joined it.

    Unclaimed legacy courses (owner_id NULL) are NOT accessible here: they can
    only be reached by claiming them first via POST /api/claim-legacy-data
    (routers/auth_api.py), which assigns owner_id to the claiming user. Denying
    by default closes the hole where every authenticated user could read any
    unclaimed course.
    """
    if _db is None:
        return False
    try:
        course = _db.table("courses").select("owner_id").eq("course_id", course_id).limit(1).execute().data
    except Exception:
        logger.exception("Ownership lookup failed for course %s", course_id)
        course = None
    owner = course[0].get("owner_id") if course else None
    if owner and owner == user_id:
        return True
    try:
        member = (_db.table("course_memberships").select("id")
                  .eq("course_id", course_id).eq("user_id", user_id).limit(1).execute().data)
        if member:
            return True
    except Exception:
        logger.exception("Membership lookup failed for course %s", course_id)
    # No ownership and no membership -> deny. This includes unclaimed legacy
    # courses (owner_id NULL), which must go through the claim flow first.
    return False


def require_course_access(course_id: str, user: Dict[str, Any]) -> None:
    """Raise 403 unless the user may access this course."""
    if not user_owns_or_member(course_id, user["id"]):
        raise HTTPException(status_code=403, detail="You don't have access to this course")
