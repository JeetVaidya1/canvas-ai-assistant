"""Central logging setup for the API process.

Single-line JSON-ish records (timestamp/level/logger/message) so log
aggregators can parse them. Level comes from Settings.log_level (LOG_LEVEL env
var, default INFO). Call :func:`setup_logging` once at process startup
(main.py does this before the routers import).

When a request is in flight, records carry a ``request_id`` field (set by
core.middleware via a contextvar) plus any structured access-log fields
(method/path/status/duration_ms/user) attached via ``extra=``.

Never log secrets or full JWTs — callers must redact before logging.
"""
from __future__ import annotations

import json
import logging
import sys

from core.config import get_settings
from core.request_context import get_request_id

# extra= fields the access log attaches; merged into the JSON payload when set.
_STRUCTURED_FIELDS = ("method", "path", "status", "duration_ms", "user")


class JsonLineFormatter(logging.Formatter):
    """Format each record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = get_request_id()
        if request_id:
            payload["request_id"] = request_id
        for field in _STRUCTURED_FIELDS:
            value = record.__dict__.get(field)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> None:
    """Configure the root logger once; safe to call repeatedly (idempotent)."""
    level_name = get_settings().log_level.upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLineFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    # Replace any pre-existing handlers so records aren't emitted twice.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
