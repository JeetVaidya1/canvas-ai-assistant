"""Per-request context propagated via contextvars.

Holds the short request ID assigned by ``core.middleware`` so any log record
emitted while handling a request (from any module, without plumbing arguments
through call stacks) can be correlated to that request.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    """The current request's short ID, or None outside a request."""
    return request_id_var.get()
