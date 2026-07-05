"""Request-ID middleware + access logging (core/middleware.py, logging_config.py).

Hermetic: uses a throwaway FastAPI app with one trivial route, plus the real
app's /system-status route (which touches no external service).
"""
import json as jsonlib
import logging
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.middleware import REQUEST_ID_HEADER, RequestContextMiddleware
from core.request_context import get_request_id, request_id_var
from logging_config import JsonLineFormatter

_ID_PATTERN = re.compile(r"^[0-9a-f]{8}$")


def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/ping")
    def ping():
        return {"request_id": get_request_id()}

    return app


@pytest.mark.unit
def test_response_carries_request_id_header():
    client = TestClient(_make_app())
    resp = client.get("/ping")
    assert resp.status_code == 200
    request_id = resp.headers.get(REQUEST_ID_HEADER)
    assert request_id is not None and _ID_PATTERN.match(request_id)
    # The handler saw the same id via the contextvar.
    assert resp.json()["request_id"] == request_id


@pytest.mark.unit
def test_request_ids_are_unique_per_request():
    client = TestClient(_make_app())
    ids = {client.get("/ping").headers[REQUEST_ID_HEADER] for _ in range(5)}
    assert len(ids) == 5


@pytest.mark.unit
def test_one_structured_access_log_line_per_request(caplog):
    client = TestClient(_make_app())
    with caplog.at_level(logging.INFO, logger="access"):
        client.get("/ping", headers={"Authorization": "Bearer fake-token"})
    records = [r for r in caplog.records if r.name == "access"]
    assert len(records) == 1
    record = records[0]
    assert record.method == "GET"
    assert record.path == "/ping"
    assert record.status == 200
    assert record.duration_ms >= 0
    # Auth fingerprint present, short, and never the raw token.
    assert record.user is not None
    assert len(record.user) == 8
    assert "fake-token" not in record.user


@pytest.mark.unit
def test_access_log_user_is_none_without_auth_header(caplog):
    client = TestClient(_make_app())
    with caplog.at_level(logging.INFO, logger="access"):
        client.get("/ping")
    record = [r for r in caplog.records if r.name == "access"][0]
    assert record.user is None


@pytest.mark.unit
def test_json_formatter_includes_request_id_and_access_fields():
    token = request_id_var.set("abcd1234")
    try:
        record = logging.LogRecord(
            "access", logging.INFO, __file__, 1, "GET /x -> 200", None, None
        )
        record.method = "GET"
        record.path = "/x"
        record.status = 200
        record.duration_ms = 1.2
        payload = jsonlib.loads(JsonLineFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert payload["request_id"] == "abcd1234"
    assert payload["method"] == "GET"
    assert payload["path"] == "/x"
    assert payload["status"] == 200
    assert payload["duration_ms"] == 1.2


@pytest.mark.unit
def test_json_formatter_omits_request_id_outside_requests():
    record = logging.LogRecord("app", logging.INFO, __file__, 1, "hello", None, None)
    payload = jsonlib.loads(JsonLineFormatter().format(record))
    assert "request_id" not in payload


@pytest.mark.unit
def test_real_app_system_status_has_request_id_header():
    import main

    client = TestClient(main.app)
    resp = client.get("/system-status")
    assert resp.status_code == 200
    assert _ID_PATTERN.match(resp.headers[REQUEST_ID_HEADER])
