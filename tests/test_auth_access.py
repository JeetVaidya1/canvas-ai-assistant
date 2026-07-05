"""Course access control (auth.user_owns_or_member / require_course_access).

Hermetic: auth._db is replaced by a fake Supabase client, so no network I/O.
The key behavior under test: unclaimed legacy courses (owner_id NULL) are NOT
readable by arbitrary authenticated users — they must go through the
POST /api/claim-legacy-data flow, which assigns them an owner first.
"""
from typing import Any, Dict, List, Optional

import pytest
from fastapi import HTTPException

import auth


class _FakeResult:
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data


class _FakeQuery:
    """Chainable stand-in for supabase.table(...).select(...).eq(...).limit(...)."""

    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    def select(self, *_args, **_kwargs) -> "_FakeQuery":
        return self

    def eq(self, column: str, value: Any) -> "_FakeQuery":
        return _FakeQuery([r for r in self._rows if r.get(column) == value])

    def limit(self, n: int) -> "_FakeQuery":
        return _FakeQuery(self._rows[:n])

    def execute(self) -> _FakeResult:
        return _FakeResult(self._rows)


class _FakeDb:
    def __init__(self, courses: List[Dict[str, Any]], memberships: Optional[List[Dict[str, Any]]] = None):
        self._tables = {"courses": courses, "course_memberships": memberships or []}

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(list(self._tables.get(name, [])))


COURSE = "cs101"
OWNER = "user-owner"
STRANGER = "user-stranger"


def _with_db(monkeypatch, courses, memberships=None):
    monkeypatch.setattr(auth, "_db", _FakeDb(courses, memberships))


@pytest.mark.unit
def test_owner_has_access(monkeypatch):
    _with_db(monkeypatch, [{"course_id": COURSE, "owner_id": OWNER}])
    assert auth.user_owns_or_member(COURSE, OWNER) is True


@pytest.mark.unit
def test_non_owner_non_member_denied(monkeypatch):
    _with_db(monkeypatch, [{"course_id": COURSE, "owner_id": OWNER}])
    assert auth.user_owns_or_member(COURSE, STRANGER) is False


@pytest.mark.unit
def test_member_of_someone_elses_course_allowed(monkeypatch):
    _with_db(
        monkeypatch,
        [{"course_id": COURSE, "owner_id": OWNER}],
        [{"id": 1, "course_id": COURSE, "user_id": STRANGER}],
    )
    assert auth.user_owns_or_member(COURSE, STRANGER) is True


@pytest.mark.unit
def test_unclaimed_legacy_course_denied(monkeypatch):
    """The old behavior returned True for owner_id NULL — that hole is closed."""
    _with_db(monkeypatch, [{"course_id": COURSE, "owner_id": None}])
    assert auth.user_owns_or_member(COURSE, STRANGER) is False


@pytest.mark.unit
def test_unclaimed_legacy_course_accessible_after_claim(monkeypatch):
    """Claim flow (owner_id gets set to the claimer) restores access for them."""
    _with_db(monkeypatch, [{"course_id": COURSE, "owner_id": None}])
    assert auth.user_owns_or_member(COURSE, STRANGER) is False

    # Simulate POST /api/claim-legacy-data: unowned courses get owner_id set.
    _with_db(monkeypatch, [{"course_id": COURSE, "owner_id": STRANGER}])
    assert auth.user_owns_or_member(COURSE, STRANGER) is True


@pytest.mark.unit
def test_unknown_course_denied(monkeypatch):
    _with_db(monkeypatch, [])
    assert auth.user_owns_or_member("nope", STRANGER) is False


@pytest.mark.unit
def test_no_db_configured_denies(monkeypatch):
    monkeypatch.setattr(auth, "_db", None)
    assert auth.user_owns_or_member(COURSE, OWNER) is False


@pytest.mark.unit
def test_require_course_access_raises_403(monkeypatch):
    _with_db(monkeypatch, [{"course_id": COURSE, "owner_id": None}])
    with pytest.raises(HTTPException) as exc:
        auth.require_course_access(COURSE, {"id": STRANGER})
    assert exc.value.status_code == 403


@pytest.mark.unit
def test_require_course_access_allows_owner(monkeypatch):
    _with_db(monkeypatch, [{"course_id": COURSE, "owner_id": OWNER}])
    auth.require_course_access(COURSE, {"id": OWNER})  # must not raise
