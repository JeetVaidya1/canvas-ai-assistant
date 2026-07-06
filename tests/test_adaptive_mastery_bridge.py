"""Adaptive difficulty mastery lookup — legacy-label bridging (hermetic).

lookup_topic_mastery no longer requires an exact ``topic`` string match:
rows written under old filename-era labels still route difficulty for the
new Course Brain topic names.
"""
import pytest

from deps import practice_generator
from fakes_endpoints import FakeSupabase

pytestmark = pytest.mark.unit

COURSE = "cs101"
USER = "user-1"


@pytest.fixture
def mastery_db(monkeypatch):
    def _install(rows):
        db = FakeSupabase({"learning_progress": rows})
        monkeypatch.setattr(practice_generator, "_supabase", db)
        return db

    return _install


def test_lookup_bridges_legacy_label(mastery_db):
    mastery_db([
        {"user_id": USER, "course_id": COURSE, "topic": "301 3 Excel", "mastery_level": 0.9},
    ])
    assert practice_generator.lookup_topic_mastery(COURSE, "Excel Formulas", USER) == 0.9


def test_lookup_exact_label_still_wins(mastery_db):
    mastery_db([
        {"user_id": USER, "course_id": COURSE, "topic": "Hash Tables", "mastery_level": 0.7},
        {"user_id": USER, "course_id": COURSE, "topic": "Hashing Extras", "mastery_level": 0.1},
    ])
    assert practice_generator.lookup_topic_mastery(COURSE, "hash tables", USER) == 0.7


def test_lookup_scopes_to_user_and_course(mastery_db):
    mastery_db([
        {"user_id": "someone-else", "course_id": COURSE, "topic": "Hash Tables", "mastery_level": 0.9},
        {"user_id": USER, "course_id": "other-course", "topic": "Hash Tables", "mastery_level": 0.9},
    ])
    assert practice_generator.lookup_topic_mastery(COURSE, "Hash Tables", USER) == 0.5


def test_lookup_defaults_to_half_when_unknown(mastery_db):
    mastery_db([])
    assert practice_generator.lookup_topic_mastery(COURSE, "Anything", USER) == 0.5


def test_lookup_defaults_to_half_on_db_failure(mastery_db):
    db = mastery_db([])
    db.fail = True
    assert practice_generator.lookup_topic_mastery(COURSE, "Anything", USER) == 0.5
