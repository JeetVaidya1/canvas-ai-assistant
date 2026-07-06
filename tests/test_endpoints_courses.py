"""Course endpoints: create / list / delete over the real app.

Hermetic: core.courses_store._db and auth._db are swapped for FakeSupabase,
and routers.courses' own supabase reference is patched for the delete path.
Local scratch dirs are redirected to tmp_path via monkeypatch.chdir.
"""
import pytest

import auth
import routers.courses as courses_router
from core import courses_store
from fakes_endpoints import LEAK_MARKER, FakeSupabase

pytestmark = pytest.mark.unit

OWNER = "user-owner"
STRANGER = "user-stranger"
COURSE = "cs101"


@pytest.fixture
def course_store_db(monkeypatch):
    def _install(courses=None, fail=False):
        db = FakeSupabase({"courses": courses or []}, fail=fail)
        monkeypatch.setattr(courses_store, "_db", db)
        return db

    return _install


@pytest.fixture
def auth_db(monkeypatch):
    def _install(courses=None, memberships=None):
        db = FakeSupabase({
            "courses": courses or [],
            "course_memberships": memberships or [],
        })
        monkeypatch.setattr(auth, "_db", db)
        return db

    return _install


# ---- POST /create-course -------------------------------------------------

def test_create_course_success(client, as_user, course_store_db, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    as_user(OWNER)
    db = course_store_db([])
    resp = client.post("/create-course", data={"course_id": COURSE, "title": "Algorithms"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "message": "Created course Algorithms"}
    [row] = db.tables["courses"]
    assert row["course_id"] == COURSE
    assert row["title"] == "Algorithms"
    assert row["owner_id"] == OWNER


def test_create_course_owner_comes_from_token_not_form(client, as_user, course_store_db,
                                                       monkeypatch, tmp_path):
    """A spoofed user_id form field must be ignored; owner is the token subject."""
    monkeypatch.chdir(tmp_path)
    as_user(OWNER)
    db = course_store_db([])
    resp = client.post(
        "/create-course",
        data={"course_id": COURSE, "title": "Algorithms", "user_id": "someone-else"},
    )
    assert resp.status_code == 200
    assert db.tables["courses"][0]["owner_id"] == OWNER


def test_create_course_duplicate_is_400(client, as_user, course_store_db, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    as_user(OWNER)
    course_store_db([{"course_id": COURSE, "title": "Algorithms", "owner_id": OWNER}])
    resp = client.post("/create-course", data={"course_id": COURSE, "title": "Again"})
    assert resp.status_code == 400
    assert resp.json() == {"detail": "Course already exists"}


def test_create_course_store_failure_is_honest_500(client, as_user, course_store_db,
                                                   monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    as_user(OWNER)
    course_store_db([], fail=True)
    resp = client.post("/create-course", data={"course_id": COURSE, "title": "Algorithms"})
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Failed to create course"}
    assert LEAK_MARKER not in resp.text


@pytest.mark.parametrize("data", [
    {},
    {"course_id": COURSE},
    {"title": "Algorithms"},
])
def test_create_course_missing_fields_is_422(client, as_user, data):
    as_user(OWNER)
    resp = client.post("/create-course", data=data)
    assert resp.status_code == 422


def test_create_course_unauthenticated_is_401(client):
    resp = client.post("/create-course", data={"course_id": COURSE, "title": "X"})
    assert resp.status_code == 401


# ---- GET /list-courses -----------------------------------------------------
# Multi-tenant rule: only courses the token user owns or has joined come back.

def test_list_courses_requires_auth(client, course_store_db):
    course_store_db([{"course_id": COURSE, "title": "Algorithms", "owner_id": OWNER}])
    resp = client.get("/list-courses")
    assert resp.status_code == 401


def test_list_courses_returns_only_token_users_courses(client, as_user, course_store_db):
    as_user(OWNER)
    course_store_db([
        {"course_id": "cs101", "title": "Algorithms", "owner_id": OWNER},
        {"course_id": "bio200", "title": "Cells", "owner_id": STRANGER},
    ])
    resp = client.get("/list-courses")
    assert resp.status_code == 200
    assert resp.json() == {"courses": [
        {"course_id": "cs101", "title": "Algorithms"},
    ]}


def test_list_courses_includes_joined_courses(client, as_user, course_store_db):
    as_user(OWNER)
    db = course_store_db([
        {"course_id": "cs101", "title": "Algorithms", "owner_id": OWNER},
        {"course_id": "bio200", "title": "Cells", "owner_id": STRANGER},
        {"course_id": "chem300", "title": "Orgo", "owner_id": STRANGER},
    ])
    db.tables["course_memberships"] = [
        {"id": 1, "course_id": "bio200", "user_id": OWNER},
    ]
    resp = client.get("/list-courses")
    assert resp.status_code == 200
    assert resp.json() == {"courses": [
        {"course_id": "cs101", "title": "Algorithms"},
        {"course_id": "bio200", "title": "Cells"},
    ]}


def test_list_courses_store_failure_is_honest_500(client, as_user, course_store_db):
    as_user(OWNER)
    course_store_db([], fail=True)
    resp = client.get("/list-courses")
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Failed to list courses"}
    assert LEAK_MARKER not in resp.text


# ---- GET /list-files -------------------------------------------------------

def test_list_files_requires_auth(client):
    resp = client.get("/list-files", params={"course_id": COURSE})
    assert resp.status_code == 401


def test_list_files_non_member_is_403(client, as_user, auth_db, monkeypatch):
    as_user(STRANGER)
    auth_db([{"course_id": COURSE, "owner_id": OWNER}])
    monkeypatch.setattr(courses_router, "supabase", FakeSupabase({
        "files": [{"course_id": COURSE, "filename": "week1.pdf"}],
    }))
    resp = client.get("/list-files", params={"course_id": COURSE})
    assert resp.status_code == 403
    assert resp.json() == {"detail": "You don't have access to this course"}
    assert "week1.pdf" not in resp.text


def test_list_files_owner_gets_course_files(client, as_user, auth_db, monkeypatch):
    as_user(OWNER)
    auth_db([{"course_id": COURSE, "owner_id": OWNER}])
    monkeypatch.setattr(courses_router, "supabase", FakeSupabase({
        "files": [
            {"course_id": COURSE, "filename": "week1.pdf"},
            {"course_id": "bio200", "filename": "other.pdf"},
        ],
    }))
    resp = client.get("/list-files", params={"course_id": COURSE})
    assert resp.status_code == 200
    assert resp.json() == {"files": ["week1.pdf"]}


# ---- legacy POST /upload/{course_id} ----------------------------------------

def test_legacy_single_file_upload_route_is_gone(client):
    """The unauthenticated legacy route was deleted (frontend never called it)."""
    resp = client.post(f"/upload/{COURSE}", files={"file": ("a.pdf", b"%PDF-fake")})
    assert resp.status_code == 404


# ---- POST /delete-course ---------------------------------------------------

def _install_delete_path_fakes(monkeypatch, course_store_db, tmp_path):
    """Everything the owner-success delete path touches, faked."""
    monkeypatch.chdir(tmp_path)
    course_store_db([{"course_id": COURSE, "title": "Algorithms", "owner_id": OWNER}])
    monkeypatch.setattr(courses_router, "supabase", FakeSupabase({"files": []}))
    monkeypatch.setattr(courses_router, "delete_course", lambda _cid: True)


def test_delete_course_non_owner_is_403(client, as_user, auth_db):
    as_user(STRANGER)
    auth_db([{"course_id": COURSE, "owner_id": OWNER}])
    resp = client.post("/delete-course", data={"course_id": COURSE})
    assert resp.status_code == 403
    assert resp.json() == {"detail": "You don't have access to this course"}


def test_delete_course_unauthenticated_is_401(client):
    resp = client.post("/delete-course", data={"course_id": COURSE})
    assert resp.status_code == 401


def test_delete_course_owner_succeeds(client, as_user, auth_db, course_store_db,
                                      monkeypatch, tmp_path):
    as_user(OWNER)
    auth_db([{"course_id": COURSE, "owner_id": OWNER}])
    _install_delete_path_fakes(monkeypatch, course_store_db, tmp_path)
    resp = client.post("/delete-course", data={"course_id": COURSE})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "message": f"Deleted course {COURSE}"}
    assert courses_store._db.tables["courses"] == []


def test_delete_course_member_succeeds(client, as_user, auth_db, course_store_db,
                                       monkeypatch, tmp_path):
    """A joined member (not owner) may also delete per require_course_access."""
    as_user(STRANGER)
    auth_db(
        [{"course_id": COURSE, "owner_id": OWNER}],
        memberships=[{"id": 1, "course_id": COURSE, "user_id": STRANGER}],
    )
    _install_delete_path_fakes(monkeypatch, course_store_db, tmp_path)
    resp = client.post("/delete-course", data={"course_id": COURSE})
    assert resp.status_code == 200
