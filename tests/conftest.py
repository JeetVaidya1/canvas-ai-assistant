"""tests/ conftest: fixtures for endpoint-level tests over the real app.

The root conftest.py (repo root) injects fake SUPABASE_/ANTHROPIC_ env vars
before any app code imports, so ``import main`` here is hermetic. This file
adds request-layer plumbing on top:

- ``client``: TestClient over ``main.app`` (raise_server_exceptions=False so
  the global error-envelope handler's 500 responses are observable).
- ``as_user``: installs dependency overrides for auth (current_user_id /
  get_current_user) and, by default, the AI rate limiter.
- An autouse fixture guarantees ``app.dependency_overrides`` is cleared after
  every test so overrides can never bleed across tests.
"""
import pytest
from fastapi.testclient import TestClient

import auth
import main
import rate_limit


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    """Never let one test's overrides leak into the next."""
    yield
    main.app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def as_user():
    """Install auth-dependency overrides for a chosen user id.

    Returns a callable: ``uid = as_user("user-1")``. By default the AI rate
    limiter is bypassed too (its Depends chain is what most tests want out of
    the way); pass ``bypass_rate_limit=False`` to exercise the real limiter.
    """
    def _install(user_id: str = "user-token", email: str = "user@test.dev",
                 bypass_rate_limit: bool = True) -> str:
        main.app.dependency_overrides[auth.current_user_id] = lambda: user_id
        main.app.dependency_overrides[auth.get_current_user] = lambda: {
            "id": user_id,
            "email": email,
        }
        if bypass_rate_limit:
            main.app.dependency_overrides[rate_limit.ai_rate_limit] = lambda: user_id
        return user_id

    return _install
