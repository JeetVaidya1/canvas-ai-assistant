"""Shared fakes for endpoint-level tests (fastapi.testclient over main.app).

``FakeSupabase`` is a chainable stand-in for the supabase-py client, following
the established pattern in tests/test_courses_store.py, extended with the
operations the routers use (update / single / storage / rpc). It is a test
double: its internal ``tables`` dict is deliberately mutable state.

Failure mode: ``fail=True`` makes every ``execute()`` raise with a marker
string (``LEAK_MARKER``) that must NEVER appear in an HTTP response body —
tests use it to prove error responses don't leak internals.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# If this ever shows up in a response body, an endpoint leaked internals.
LEAK_MARKER = "password=hunter2"


class FakeResult:
    def __init__(self, data: Any):
        self.data = data


class FakeQuery:
    """Chainable stand-in for supabase.table(...) query builders."""

    def __init__(self, db: "FakeSupabase", table: str):
        self._db = db
        self._table = table
        self._op = "select"
        self._payload: Optional[Dict[str, Any]] = None
        self._filters: List[tuple] = []
        self._limit: Optional[int] = None
        self._single = False

    # -- chainable no-op/filter modifiers -------------------------------
    def select(self, *_args, **_kwargs) -> "FakeQuery":
        return self

    def order(self, *_args, **_kwargs) -> "FakeQuery":
        return self

    def limit(self, n: int) -> "FakeQuery":
        self._limit = n
        return self

    def single(self) -> "FakeQuery":
        self._single = True
        return self

    def eq(self, column: str, value: Any) -> "FakeQuery":
        self._filters.append((column, value))
        return self

    # -- operations ------------------------------------------------------
    def insert(self, record: Dict[str, Any]) -> "FakeQuery":
        self._op = "insert"
        self._payload = record
        return self

    def update(self, record: Dict[str, Any]) -> "FakeQuery":
        self._op = "update"
        self._payload = record
        return self

    def delete(self) -> "FakeQuery":
        self._op = "delete"
        return self

    # -- execution ---------------------------------------------------------
    def _matches(self, row: Dict[str, Any]) -> bool:
        return all(row.get(col) == val for col, val in self._filters)

    def execute(self) -> FakeResult:
        if self._db.fail:
            raise RuntimeError(f"supabase exploded: {LEAK_MARKER}")
        rows = self._db.tables.setdefault(self._table, [])
        if self._op == "insert":
            return self._execute_insert(rows)
        if self._op == "update":
            return self._execute_update(rows)
        if self._op == "delete":
            self._db.tables[self._table] = [r for r in rows if not self._matches(r)]
            return FakeResult([])
        return self._execute_select(rows)

    def _execute_insert(self, rows: List[Dict[str, Any]]) -> FakeResult:
        record = dict(self._payload or {})
        record.setdefault("id", f"{self._table}-row-{len(rows) + 1}")
        rows.append(record)
        return FakeResult([dict(record)])

    def _execute_update(self, rows: List[Dict[str, Any]]) -> FakeResult:
        updated = []
        for index, row in enumerate(rows):
            if self._matches(row):
                rows[index] = {**row, **(self._payload or {})}
                updated.append(dict(rows[index]))
        return FakeResult(updated)

    def _execute_select(self, rows: List[Dict[str, Any]]) -> FakeResult:
        out = [dict(r) for r in rows if self._matches(r)]
        if self._limit is not None:
            out = out[: self._limit]
        if self._single:
            if not out:
                raise RuntimeError("single(): no rows found")
            return FakeResult(out[0])
        return FakeResult(out)


class FakeStorageBucket:
    def remove(self, _paths: List[str]) -> Dict[str, Any]:
        return {}

    def download(self, _path: str) -> bytes:
        return b"fake-bytes"


class FakeStorage:
    def from_(self, _bucket: str) -> FakeStorageBucket:
        return FakeStorageBucket()


class FakeSupabase:
    """Fake supabase client backed by an in-memory ``tables`` dict."""

    def __init__(self, tables: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                 fail: bool = False):
        self.tables: Dict[str, List[Dict[str, Any]]] = {
            name: [dict(row) for row in rows] for name, rows in (tables or {}).items()
        }
        self.fail = fail
        self.storage = FakeStorage()

    def table(self, name: str) -> FakeQuery:
        return FakeQuery(self, name)

    def rpc(self, *_args, **_kwargs) -> FakeQuery:
        return FakeQuery(self, "_rpc")
