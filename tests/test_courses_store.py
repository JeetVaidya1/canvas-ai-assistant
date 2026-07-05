"""Supabase-backed course store (core/courses_store.py).

Hermetic: ``courses_store._db`` is replaced by a fake Supabase client (same
pattern as tests/test_auth_access.py), so no network I/O ever happens.
"""
from typing import Any, Dict, List, Optional

import pytest

from core import courses_store
from core.courses_store import CourseStoreError


class _FakeResult:
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data


class _FakeQuery:
    """Chainable stand-in for supabase.table(...).select/insert/delete chains."""

    def __init__(self, db: "_FakeDb", table: str):
        self._db = db
        self._table = table
        self._rows = list(db.tables.get(table, []))
        self._op = "select"
        self._payload: Optional[Dict[str, Any]] = None
        self._filters: List[tuple] = []

    def select(self, *_args, **_kwargs) -> "_FakeQuery":
        return self

    def order(self, *_args, **_kwargs) -> "_FakeQuery":
        return self

    def limit(self, n: int) -> "_FakeQuery":
        self._rows = self._rows[:n]
        return self

    def eq(self, column: str, value: Any) -> "_FakeQuery":
        self._filters.append((column, value))
        self._rows = [r for r in self._rows if r.get(column) == value]
        return self

    def insert(self, record: Dict[str, Any]) -> "_FakeQuery":
        self._op = "insert"
        self._payload = record
        return self

    def delete(self) -> "_FakeQuery":
        self._op = "delete"
        return self

    def execute(self) -> _FakeResult:
        if self._db.fail:
            raise RuntimeError("supabase is down")
        if self._op == "insert":
            self._db.tables.setdefault(self._table, []).append(dict(self._payload))
            return _FakeResult([dict(self._payload)])
        if self._op == "delete":
            remaining = [
                r
                for r in self._db.tables.get(self._table, [])
                if not all(r.get(col) == val for col, val in self._filters)
            ]
            self._db.tables[self._table] = remaining
            return _FakeResult([])
        return _FakeResult(self._rows)


class _FakeDb:
    def __init__(self, courses: Optional[List[Dict[str, Any]]] = None, fail: bool = False):
        self.tables: Dict[str, List[Dict[str, Any]]] = {"courses": list(courses or [])}
        self.fail = fail

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self, name)


@pytest.fixture
def fake_db(monkeypatch):
    def _install(courses=None, fail=False):
        db = _FakeDb(courses, fail=fail)
        monkeypatch.setattr(courses_store, "_db", db)
        return db

    return _install


@pytest.mark.unit
def test_course_exists_true_and_false(fake_db):
    fake_db([{"course_id": "cs101", "title": "Algorithms", "owner_id": "u1"}])
    assert courses_store.course_exists("cs101") is True
    assert courses_store.course_exists("nope") is False


@pytest.mark.unit
def test_create_course_inserts_row_with_owner(fake_db):
    db = fake_db([])
    created = courses_store.create_course("cs101", "Algorithms", owner_id="u1")
    assert created == {"course_id": "cs101", "title": "Algorithms", "owner_id": "u1"}
    assert db.tables["courses"] == [
        {"course_id": "cs101", "title": "Algorithms", "owner_id": "u1"}
    ]


@pytest.mark.unit
def test_create_course_strips_whitespace(fake_db):
    db = fake_db([])
    courses_store.create_course("  cs101  ", "  Algorithms  ", owner_id="u1")
    assert db.tables["courses"][0]["course_id"] == "cs101"
    assert db.tables["courses"][0]["title"] == "Algorithms"


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["", "   ", None])
def test_create_course_rejects_invalid_course_id(fake_db, bad):
    fake_db([])
    with pytest.raises(CourseStoreError):
        courses_store.create_course(bad, "Title", owner_id="u1")


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["", "   ", None])
def test_create_course_rejects_invalid_title(fake_db, bad):
    fake_db([])
    with pytest.raises(CourseStoreError):
        courses_store.create_course("cs101", bad, owner_id="u1")


@pytest.mark.unit
def test_list_courses_returns_id_and_title_pairs(fake_db):
    fake_db(
        [
            {"course_id": "cs101", "title": "Algorithms", "owner_id": "u1"},
            {"course_id": "bio200", "title": None, "owner_id": "u2"},
        ]
    )
    assert courses_store.list_courses() == [
        {"course_id": "cs101", "title": "Algorithms"},
        {"course_id": "bio200", "title": None},
    ]


@pytest.mark.unit
def test_delete_course_removes_only_that_row(fake_db):
    db = fake_db(
        [
            {"course_id": "cs101", "title": "Algorithms"},
            {"course_id": "bio200", "title": "Cells"},
        ]
    )
    courses_store.delete_course("cs101")
    assert db.tables["courses"] == [{"course_id": "bio200", "title": "Cells"}]


@pytest.mark.unit
def test_db_failures_surface_as_course_store_error(fake_db):
    fake_db([], fail=True)
    with pytest.raises(CourseStoreError):
        courses_store.course_exists("cs101")
    with pytest.raises(CourseStoreError):
        courses_store.create_course("cs101", "Algorithms", owner_id="u1")
    with pytest.raises(CourseStoreError):
        courses_store.list_courses()
    with pytest.raises(CourseStoreError):
        courses_store.delete_course("cs101")
