"""Course Brain topic endpoints over the real app (hermetic).

Covers: /practice-topics served from the course_topics table, auth on the
regenerate route, and the new full-object /api/courses/{id}/topics routes.
"""
import pytest

import auth
import course_brain
import deps
from fakes_endpoints import FakeSupabase

pytestmark = pytest.mark.unit

OWNER = "user-owner"
STRANGER = "user-stranger"
COURSE = "cs101"

TOPIC_ROWS = [
    {"course_id": COURSE, "slug": "hash-tables", "name": "Hash Tables",
     "description": "Hashing maps keys to buckets.", "doc_coverage": [{"doc": "hashing.pdf", "pages": [1, 4]}],
     "prereq_slugs": ["binary-search-trees"], "position": 1},
    {"course_id": COURSE, "slug": "binary-search-trees", "name": "Binary Search Trees",
     "description": "Ordered trees with log-time search.", "doc_coverage": [{"doc": "trees.pdf", "pages": [2, 9]}],
     "prereq_slugs": [], "position": 0},
]


@pytest.fixture
def topics_env(monkeypatch):
    """Wire auth ownership, deps validation data, and stored course topics."""
    def _install(course_topics=None):
        monkeypatch.setattr(auth, "_db", FakeSupabase({
            "courses": [{"course_id": COURSE, "owner_id": OWNER}],
            "course_memberships": [],
        }))
        monkeypatch.setattr(deps, "supabase", FakeSupabase({
            "courses": [{"course_id": COURSE, "title": "Algorithms", "owner_id": OWNER}],
            "files": [{"course_id": COURSE, "filename": "trees.pdf"},
                      {"course_id": COURSE, "filename": "hashing.pdf"}],
            "embeddings": [{"course_id": COURSE, "id": 1}],
        }))
        db = FakeSupabase({"course_topics": course_topics if course_topics is not None else TOPIC_ROWS})
        monkeypatch.setattr(course_brain, "_supabase", db)
        monkeypatch.setattr(course_brain, "structured_call",
                            lambda *a, **k: pytest.fail("LLM must not be called"))
        return db

    return _install


# ---- GET /practice-topics/{course_id} ---------------------------------------

def test_practice_topics_requires_auth(client):
    resp = client.get(f"/practice-topics/{COURSE}")
    assert resp.status_code == 401


def test_practice_topics_non_member_is_403(client, as_user, topics_env):
    as_user(STRANGER)
    topics_env()
    resp = client.get(f"/practice-topics/{COURSE}")
    assert resp.status_code == 403


def test_practice_topics_served_from_table_in_order(client, as_user, topics_env):
    """Stored Course Brain topics are served (no LLM), ordered by position."""
    as_user(OWNER)
    topics_env()
    resp = client.get(f"/practice-topics/{COURSE}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["topics"] == ["Binary Search Trees", "Hash Tables"]
    assert body["status"] == "success"
    assert body["extraction_method"] == "course_brain"
    assert body["course_files_count"] == 2


def test_practice_topics_unknown_course_returns_validation_error(client, as_user, topics_env, monkeypatch):
    as_user(OWNER)
    topics_env()
    monkeypatch.setattr(auth, "_db", FakeSupabase({
        "courses": [{"course_id": "ghost", "owner_id": OWNER}],
        "course_memberships": [],
    }))
    resp = client.get("/practice-topics/ghost")
    assert resp.status_code == 200
    assert resp.json()["status"] == "error"           # legacy error envelope kept


# ---- POST /regenerate-practice-topics ---------------------------------------

def test_regenerate_topics_requires_auth(client):
    resp = client.post("/regenerate-practice-topics", data={"course_id": COURSE})
    assert resp.status_code == 401


def test_regenerate_topics_non_member_is_403(client, as_user, topics_env):
    as_user(STRANGER)
    topics_env()
    resp = client.post("/regenerate-practice-topics", data={"course_id": COURSE})
    assert resp.status_code == 403


def test_regenerate_topics_resynthesizes(client, as_user, topics_env, monkeypatch):
    as_user(OWNER)
    topics_env()
    monkeypatch.setattr(course_brain, "synthesize_topics", lambda cid: [
        course_brain.Topic(slug="fresh-topic", name="Fresh Topic", position=0),
    ])
    resp = client.post("/regenerate-practice-topics", data={"course_id": COURSE})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["topics"] == ["Fresh Topic"]


# ---- GET /api/courses/{course_id}/topics ------------------------------------

def test_course_topics_requires_auth(client):
    resp = client.get(f"/api/courses/{COURSE}/topics")
    assert resp.status_code == 401


def test_course_topics_non_member_is_403(client, as_user, topics_env):
    as_user(STRANGER)
    topics_env()
    resp = client.get(f"/api/courses/{COURSE}/topics")
    assert resp.status_code == 403


def test_course_topics_returns_full_objects(client, as_user, topics_env):
    as_user(OWNER)
    topics_env()
    resp = client.get(f"/api/courses/{COURSE}/topics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["course_id"] == COURSE
    assert body["count"] == 2
    first = body["topics"][0]
    assert first == {
        "slug": "binary-search-trees",
        "name": "Binary Search Trees",
        "description": "Ordered trees with log-time search.",
        "doc_coverage": [{"doc": "trees.pdf", "pages": [2, 9]}],
        "prereq_slugs": [],
        "position": 0,
    }
    assert body["topics"][1]["prereq_slugs"] == ["binary-search-trees"]


# ---- POST /api/courses/{course_id}/topics/rebuild ----------------------------

def test_rebuild_topics_requires_auth(client):
    resp = client.post(f"/api/courses/{COURSE}/topics/rebuild")
    assert resp.status_code == 401


def test_rebuild_topics_non_member_is_403(client, as_user, topics_env):
    as_user(STRANGER)
    topics_env()
    resp = client.post(f"/api/courses/{COURSE}/topics/rebuild")
    assert resp.status_code == 403


def test_rebuild_topics_owner_gets_fresh_synthesis(client, as_user, topics_env, monkeypatch):
    as_user(OWNER)
    topics_env()
    monkeypatch.setattr(course_brain, "synthesize_topics", lambda cid: [
        course_brain.Topic(slug="fresh-topic", name="Fresh Topic",
                           description="New.", position=0),
    ])
    resp = client.post(f"/api/courses/{COURSE}/topics/rebuild")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["topics"][0]["slug"] == "fresh-topic"


def test_rebuild_topics_empty_course_is_409(client, as_user, topics_env, monkeypatch):
    as_user(OWNER)
    topics_env()
    monkeypatch.setattr(course_brain, "synthesize_topics", lambda cid: [])
    resp = client.post(f"/api/courses/{COURSE}/topics/rebuild")
    assert resp.status_code == 409


def test_rebuild_topics_synthesis_failure_is_honest_500(client, as_user, topics_env, monkeypatch):
    as_user(OWNER)
    topics_env()

    def boom(_cid):
        raise RuntimeError("llm exploded: secret-internal-detail")

    monkeypatch.setattr(course_brain, "synthesize_topics", boom)
    resp = client.post(f"/api/courses/{COURSE}/topics/rebuild")
    assert resp.status_code == 500
    assert "secret-internal-detail" not in resp.text
