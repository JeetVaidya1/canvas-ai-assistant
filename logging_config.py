"""Central logging setup for the API process.

Single-line JSON-ish records (timestamp/level/logger/message) so log
aggregators can parse them, stdlib only. Level comes from the LOG_LEVEL env
var (default INFO). Call :func:`setup_logging` once at process startup
(main.py does this before the routers import).

Never log secrets or full JWTs — callers must redact before logging.
"""
from __future__ import annotations

import json
import logging
import os
import sys

_DEFAULT_LEVEL = "INFO"


class JsonLineFormatter(logging.Formatter):
    """Format each record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> None:
    """Configure the root logger once; safe to call repeatedly (idempotent)."""
    level_name = os.getenv("LOG_LEVEL", _DEFAULT_LEVEL).upper()
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
