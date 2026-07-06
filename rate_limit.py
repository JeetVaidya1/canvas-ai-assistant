"""Per-user rate limiting for AI endpoints.

Protects the free-trial window: a single user (or a runaway client loop) can't
burn unbounded Claude spend. Sliding-window counters keyed by the authenticated
user id.

Storage is in-memory, which is correct for the single always-on Fly machine we
deploy on. If the backend is ever scaled to multiple instances, swap the limiter
store for Redis/Postgres — the public surface (``ai_rate_limit``) stays the same.
Daily-cap persistence (surviving restarts) also lands with the Phase 5 tiering.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

from fastapi import Depends, HTTPException

from auth import current_user_id
from core.config import get_settings


class SlidingWindowLimiter:
    """Allow at most ``max_requests`` per ``window_seconds`` per key."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> Tuple[bool, int]:
        """Return (allowed, retry_after_seconds). Records the hit when allowed."""
        now = time.monotonic() if now is None else now
        cutoff = now - self.window
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.max_requests:
                retry_after = int(hits[0] + self.window - now) + 1
                return False, max(retry_after, 1)
            hits.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


# Default AI limit: tune via AI_RATE_LIMIT_PER_MINUTE. Generous enough for real
# study sessions, tight enough that a runaway loop is capped. Phase 5 will vary
# this by subscription tier.
_AI_PER_MINUTE = get_settings().ai_rate_limit_per_minute
_ai_limiter = SlidingWindowLimiter(_AI_PER_MINUTE, 60.0)


def ai_rate_limit(user_id: str = Depends(current_user_id)) -> str:
    """FastAPI dependency: 429 if the user exceeds the AI call rate.

    Returns the user id so an endpoint can use it as ``Depends(ai_rate_limit)``
    in place of ``Depends(current_user_id)`` — one dependency, both jobs.
    """
    allowed, retry_after = _ai_limiter.check(user_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded — try again in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )
    return user_id
