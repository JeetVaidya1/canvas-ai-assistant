"""Global error handling (errors.install_error_handlers).

Verified on a throwaway app so the test is hermetic (no DB/AI) yet exercises the
real handler that main.py installs.
"""
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from errors import install_error_handlers


def _client() -> TestClient:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom")
    def boom():
        raise RuntimeError("secret internal detail that must not leak")

    @app.get("/deliberate")
    def deliberate():
        raise HTTPException(status_code=403, detail="forbidden")

    # raise_server_exceptions=False so the handler's response is returned, not re-raised.
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.unit
def test_unhandled_exception_returns_clean_500():
    resp = _client().get("/boom")
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "internal_server_error"
    assert "message" in body


@pytest.mark.unit
def test_internal_details_do_not_leak():
    resp = _client().get("/boom")
    assert "secret internal detail" not in resp.text


@pytest.mark.unit
def test_deliberate_http_exception_is_untouched():
    # HTTPException keeps FastAPI's default handling (not swallowed by our catch-all).
    resp = _client().get("/deliberate")
    assert resp.status_code == 403
    assert resp.json()["detail"] == "forbidden"
