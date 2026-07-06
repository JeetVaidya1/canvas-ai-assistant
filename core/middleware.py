"""Request-ID assignment + structured access logging.

Every request gets a short ID (uuid4 hex prefix) stored in a contextvar so all
log records emitted while handling it carry ``request_id`` (see
``logging_config.JsonLineFormatter``), echoed back in an ``X-Request-ID``
response header, plus exactly one structured access-log line per request.
"""
from __future__ import annotations

import hashlib
import logging
import time
import uuid
from typing import Awaitable, Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from core.request_context import request_id_var

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_LENGTH = 8

_access_logger = logging.getLogger("access")


def _auth_fingerprint(request: Request) -> Optional[str]:
    """Short stable hash of the Authorization header.

    Correlates requests from the same session in logs without ever logging the
    token itself (and without paying for JWT verification in the middleware).
    """
    authorization = request.headers.get("authorization")
    if not authorization:
        return None
    return hashlib.sha256(authorization.encode("utf-8")).hexdigest()[:8]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request ID, echo it in the response, log one access line."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid.uuid4().hex[:_REQUEST_ID_LENGTH]
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status_code = 500  # what the client sees if call_next raises
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            _access_logger.info(
                "%s %s -> %s (%.1f ms)",
                request.method,
                request.url.path,
                status_code,
                duration_ms,
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "duration_ms": duration_ms,
                    "user": _auth_fingerprint(request),
                },
            )
            request_id_var.reset(token)
